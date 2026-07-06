"""Runs the full evaluation pipeline on a candidate model vs. a registered baseline
and writes a markdown report.

Usage:
    python scripts/run_evaluation.py

Requires:
    - `OPENAI_API_KEY` set in the environment (used by the LLM judge).
    - A GPU is recommended for the HuggingFace models but not required for
      small models like Qwen2.5-Coder-0.5B/1.5B on CPU (just slower).
    - `data/benchmarks/v1.jsonl` populated (`python scripts/seed_benchmark.py`).

This script wires together real HuggingFace models and a real OpenAI client
as the `PipelineDependencies` the orchestrator needs; the orchestrator
itself has no idea these are "real" vs. fakes used in tests.
"""

from __future__ import annotations

import os

from llm_eval_audit.benchmark.humaneval_loader import load_humaneval
from llm_eval_audit.contamination.ngram_detector import build_ngram_hash_set
from llm_eval_audit.core.config import load_config
from llm_eval_audit.core.logging_config import get_logger
from llm_eval_audit.evaluation.llm_judge import OpenAIChatClient
from llm_eval_audit.pipeline.orchestrator import EvaluationPipeline, PipelineDependencies
from llm_eval_audit.reporting.report_generator import generate_report

logger = get_logger(__name__)


def _load_hf_causal_lm(model_name: str):
    """Load a HuggingFace causal LM + tokenizer onto the best available device."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer, device


def _make_code_generate_fn(model, tokenizer, device: str):
    """Adapts a HF model into the `(prompt, n, temperature) -> list[str]` shape Pass@k expects."""
    import torch

    def generate(prompt: str, n: int, temperature: float) -> list[str]:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                num_return_sequences=n,
                pad_token_id=tokenizer.eos_token_id,
            )
        completions = []
        for ids in output_ids:
            new_ids = ids[inputs["input_ids"].shape[1]:]
            completions.append(tokenizer.decode(new_ids, skip_special_tokens=True))
        return completions

    return generate


def _make_answer_fn(model, tokenizer, device: str):
    """Single-completion greedy generation, used for open-ended LLM-judge examples."""
    generate_fn = _make_code_generate_fn(model, tokenizer, device)

    def answer(prompt: str) -> str:
        return generate_fn(prompt, n=1, temperature=0.0)[0]

    return answer


def main() -> None:
    config = load_config()

    candidate_model, candidate_tok, candidate_device = _load_hf_causal_lm(config["models"]["candidate"])
    baseline_model, baseline_tok, baseline_device = _load_hf_causal_lm(config["models"]["baseline"])

    # A real deployment would build this from an actual training-data snapshot;
    # here we use the HumanEval canonical solutions themselves as a stand-in
    # "training corpus" to demonstrate the detector catching a known-contaminated
    # public benchmark (see README's "Design Decisions" section).
    humaneval_examples = load_humaneval(limit=20)
    corpus_texts = [e.reference_answer for e in humaneval_examples]
    corpus_hashes = build_ngram_hash_set(corpus_texts, n=config["contamination"]["ngram_n"])

    judge_client = OpenAIChatClient(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=config["models"].get("judge_api_base"),
    )

    deps = PipelineDependencies(
        candidate_code_generate_fn=_make_code_generate_fn(candidate_model, candidate_tok, candidate_device),
        baseline_code_generate_fn=_make_code_generate_fn(baseline_model, baseline_tok, baseline_device),
        candidate_answer_fn=_make_answer_fn(candidate_model, candidate_tok, candidate_device),
        baseline_answer_fn=_make_answer_fn(baseline_model, baseline_tok, baseline_device),
        judge_client=judge_client,
        corpus_hashes=corpus_hashes,
    )

    pipeline = EvaluationPipeline(
        deps=deps,
        candidate_name=config["models"]["candidate"],
        baseline_name=config["models"]["baseline"],
    )
    result = pipeline.run()

    report_path = f"{config['reporting']['output_dir']}/eval_report_{result.benchmark_version}.md"
    generate_report(result, output_path=report_path)

    logger.info("Evaluation %s. Report written to %s", "PASSED" if result.passed else "FAILED", report_path)
    if not result.passed:
        logger.warning("Fail reasons: %s", result.fail_reasons)


if __name__ == "__main__":
    main()
