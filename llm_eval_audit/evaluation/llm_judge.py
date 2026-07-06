"""LLM-as-Judge Evaluator.

Evaluates open-ended model responses (code explanation, refactoring
rationale, etc. — anything that isn't unit-testable like Pass@k) by having
a strong LLM (e.g. GPT-4o-mini) judge two candidate answers pairwise.

Bias mitigations implemented:
  - Swap trick: every comparison is run twice, with A/B order swapped the
    second time. If the verdict flips between runs, the two candidates are
    judged indistinguishable and recorded as a TIE rather than trusting
    whichever run happened to run first. This directly targets position
    bias (LLM judges systematically favor whichever answer appears first).
  - Explicit anti-bias instructions in the prompt: judges are told not to
    reward verbosity for its own sake and to normalize for response length.
  - Chain-of-thought before scoring: the judge must write out its reasoning
    BEFORE assigning numeric scores, which empirically produces more
    consistent, better-calibrated scores than scoring first.
  - Structured, multi-dimensional scoring (not a single scalar) so a verbose
    but wrong answer can't hide behind a single "quality" number.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from llm_eval_audit.core.logging_config import get_logger
from llm_eval_audit.core.types import Example, JudgeScore, Verdict

logger = get_logger(__name__)

DEFAULT_DIMENSIONS = ["correctness", "completeness", "code_quality", "explanation_clarity"]

_SYSTEM_PROMPT = """You are an expert, impartial judge evaluating two AI-generated responses \
to the same coding-related question. You must be rigorous and unbiased.

Anti-bias rules you MUST follow:
- Do NOT favor a response merely because it is longer or more verbose. Judge substance, not length.
- Do NOT let the order of presentation (Response A vs Response B) influence your judgment. \
Evaluate each response strictly on its own merits.
- If the responses are of genuinely similar quality, it is correct and expected to say so — \
do not force a winner where none clearly exists.

You must first reason step by step about the strengths and weaknesses of each response \
(chain-of-thought), and only THEN assign numeric scores. Respond with a single JSON object \
and nothing else, in exactly this schema:

{
  "reasoning": "<your step-by-step comparative analysis>",
  "scores_a": {"<dimension>": <1-5 int>, ...},
  "scores_b": {"<dimension>": <1-5 int>, ...},
  "verdict": "A" | "B" | "tie"
}
"""


def _build_user_prompt(question: str, answer_a: str, answer_b: str, dimensions: list[str]) -> str:
    dims_str = ", ".join(dimensions)
    return f"""Question:
{question}

Response A:
{answer_a}

Response B:
{answer_b}

Score each response on these dimensions (1-5 each): {dims_str}.
Then give your verdict: "A" if Response A is clearly better, "B" if Response B is clearly \
better, or "tie" if they are comparable in quality."""


class ChatClient(Protocol):
    """Structural type matching the OpenAI SDK's `client.chat.completions.create` surface.

    Any OpenAI-compatible client (official SDK pointed at a custom base_url,
    a local vLLM OpenAI-compatible server, etc.) satisfies this without any
    adapter code — the caller just constructs the client with the desired
    base_url and api_key and passes it in here.
    """

    def chat_completion(self, model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        """Return the raw text content of the model's response."""
        ...


class OpenAIChatClient:
    """Thin wrapper around `openai.OpenAI` implementing the `ChatClient` protocol."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        from openai import OpenAI  # deferred import: only needed if this adapter is used

        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def chat_completion(self, model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


class LLMJudge:
    """Pairwise LLM-as-judge evaluator with swap-trick position-bias mitigation."""

    def __init__(
        self,
        client: ChatClient,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        dimensions: list[str] | None = None,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.dimensions = dimensions or DEFAULT_DIMENSIONS

    def _call_judge(self, question: str, answer_a: str, answer_b: str) -> dict[str, Any]:
        """One raw judge call comparing `answer_a` vs `answer_b` in that literal order."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(question, answer_a, answer_b, self.dimensions)},
        ]
        raw = self.client.chat_completion(
            model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error("Judge returned non-JSON output: %s", raw[:300])
            raise ValueError(f"Judge response was not valid JSON: {e}") from e

    def compare_once(self, example: Example, answer_a: str, answer_b: str, swapped: bool = False) -> JudgeScore:
        """Run a single (non-swap-corrected) pairwise comparison."""
        parsed = self._call_judge(example.question, answer_a, answer_b)
        verdict = Verdict(parsed["verdict"].lower() if parsed["verdict"].lower() == "tie" else parsed["verdict"].upper())
        return JudgeScore(
            example_id=example.id,
            dimension_scores_a=parsed["scores_a"],
            dimension_scores_b=parsed["scores_b"],
            verdict=verdict,
            reasoning=parsed["reasoning"],
            swapped=swapped,
        )

    def compare_with_swap(self, example: Example, answer_a: str, answer_b: str) -> JudgeScore:
        """Run the comparison twice (A/B and B/A) and reconcile via the swap trick.

        If both runs agree on which underlying model won (accounting for the
        swap), the verdict stands. If they disagree, the judge is
        inconsistent under position swap and we record a TIE — this is the
        core position-bias mitigation.
        """
        first = self.compare_once(example, answer_a, answer_b, swapped=False)
        second_raw = self.compare_once(example, answer_b, answer_a, swapped=True)

        # Re-express the swapped run's verdict in terms of the ORIGINAL a/b labeling.
        remapped_verdict = {
            Verdict.A: Verdict.B,
            Verdict.B: Verdict.A,
            Verdict.TIE: Verdict.TIE,
        }[second_raw.verdict]

        if first.verdict == remapped_verdict:
            final_verdict = first.verdict
        else:
            final_verdict = Verdict.TIE
            logger.info(
                "Example '%s': swap trick detected inconsistency (%s vs %s) -> recording TIE",
                example.id, first.verdict.value, remapped_verdict.value,
            )

        # Average per-dimension scores across both orderings for stability.
        avg_a = {
            dim: (first.dimension_scores_a[dim] + second_raw.dimension_scores_b[dim]) / 2
            for dim in self.dimensions
        }
        avg_b = {
            dim: (first.dimension_scores_b[dim] + second_raw.dimension_scores_a[dim]) / 2
            for dim in self.dimensions
        }

        return JudgeScore(
            example_id=example.id,
            dimension_scores_a=avg_a,
            dimension_scores_b=avg_b,
            verdict=final_verdict,
            reasoning=f"[Run 1 (A,B)]: {first.reasoning}\n[Run 2 (B,A)]: {second_raw.reasoning}",
            swapped=False,
        )
