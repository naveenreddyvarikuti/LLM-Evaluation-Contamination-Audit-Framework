"""Head-to-head comparison of two model checkpoints on the open-ended private
benchmark, using the LLM judge (with swap trick) and Elo ratings directly —
without running the full contamination + Pass@k pipeline.

Usage:
    python scripts/compare_models.py --model-a Qwen/Qwen2.5-Coder-1.5B \
                                      --model-b Qwen/Qwen2.5-Coder-0.5B

Requires `OPENAI_API_KEY` in the environment for the LLM judge.
"""

from __future__ import annotations

import argparse
import os

from llm_eval_audit.benchmark.manager import BenchmarkManager
from llm_eval_audit.core.logging_config import get_logger
from llm_eval_audit.core.types import PairwiseComparison
from llm_eval_audit.analysis.model_comparator import ModelComparator
from llm_eval_audit.evaluation.llm_judge import LLMJudge, OpenAIChatClient

logger = get_logger(__name__)


def _make_answer_fn(model_name: str):
    """Loads a HF causal LM and returns a greedy single-completion answer function."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    def answer(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_ids, skip_special_tokens=True)

    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", default="Qwen/Qwen2.5-Coder-1.5B")
    parser.add_argument("--model-b", default="Qwen/Qwen2.5-Coder-0.5B")
    parser.add_argument("--benchmark-version", default="v1")
    args = parser.parse_args()

    bench = BenchmarkManager().load_version(args.benchmark_version)
    open_examples = [e for e in bench.active_examples() if not e.test_cases]

    answer_a = _make_answer_fn(args.model_a)
    answer_b = _make_answer_fn(args.model_b)
    judge = LLMJudge(client=OpenAIChatClient(api_key=os.environ.get("OPENAI_API_KEY")))

    comparisons: list[PairwiseComparison] = []
    for example in open_examples:
        response_a = answer_a(example.question)
        response_b = answer_b(example.question)
        score = judge.compare_with_swap(example, response_a, response_b)
        comparisons.append(
            PairwiseComparison(
                example_id=example.id,
                model_a=args.model_a,
                model_b=args.model_b,
                verdict=score.verdict,
                dimension_scores_a=score.dimension_scores_a,
                dimension_scores_b=score.dimension_scores_b,
            )
        )

    comparator = ModelComparator()
    elo_ratings = comparator.bootstrap_elo_ci(comparisons)
    win_rates = ModelComparator.win_rates(comparisons)

    print(f"\n=== {args.model_a} vs {args.model_b} ===")
    for name in (args.model_a, args.model_b):
        rating = elo_ratings[name]
        print(f"{name}: Elo {rating.rating:.1f} [{rating.ci_lower:.1f}, {rating.ci_upper:.1f}] "
              f"| win rate {win_rates[name]:.1%}")

    distinguishable = ModelComparator.is_distinguishable(elo_ratings[args.model_a], elo_ratings[args.model_b])
    print(f"Statistically distinguishable: {'Yes' if distinguishable else 'No'}")


if __name__ == "__main__":
    main()
