"""Failure Taxonomy Classifier.

Classifies individual model failures (wrong Pass@k outputs, low-scoring
LLM-judge responses, etc.) into a fixed taxonomy using an LLM judge, so that
"the model failed 30% of examples" becomes "18% reasoning errors, 8% format
errors, 4% refusals" — actionable, not just a number.

Categories mirror common code-model failure modes:
  - factual_error: states something incorrect (wrong API behavior, wrong fact)
  - reasoning_error: logic/algorithm is flawed even if syntactically valid
  - format_error: right idea, wrong output shape (e.g. ignored required signature)
  - compilation_error: doesn't parse/run at all
  - refusal_error: model declined to answer or produced a non-answer
  - other: doesn't fit the above
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Optional

from llm_eval_audit.core.logging_config import get_logger
from llm_eval_audit.core.types import Example, FailureCategory, FailureRecord, Severity
from llm_eval_audit.evaluation.llm_judge import ChatClient

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are an expert code reviewer classifying WHY a model's response to a \
coding question failed or fell short. Choose exactly one category from this fixed list:

- factual_error: the response asserts something incorrect (wrong API/library behavior, wrong fact)
- reasoning_error: the underlying logic or algorithm is flawed, even if the code runs
- format_error: the content is broadly right but violates the required output shape/signature
- compilation_error: the code does not parse or run at all (syntax/import/runtime crash)
- refusal_error: the model declined to answer, hedged without answering, or gave a non-answer
- other: does not clearly fit any category above

Respond with a single JSON object and nothing else:
{"category": "<one of the categories above>", "explanation": "<one sentence, specific>", \
"severity": "high" | "medium" | "low"}

Severity guidance: "high" = would mislead or break a real user's code; "medium" = incorrect but \
low-impact or easily caught; "low" = cosmetic/minor issue."""


# Static, non-LLM recommended fix directions per category — kept as a lightweight
# lookup table rather than another model call, since the mapping from category to
# fix strategy is fairly stable domain knowledge.
_RECOMMENDED_FIXES: dict[FailureCategory, str] = {
    FailureCategory.FACTUAL_ERROR: "Augment fine-tuning data with more accurate reference material; consider retrieval-augmented generation for fact-heavy queries.",
    FailureCategory.REASONING_ERROR: "Add chain-of-thought / self-consistency prompting; fine-tune on harder algorithmic examples with step-by-step solutions.",
    FailureCategory.FORMAT_ERROR: "Tighten prompt instructions on required output format; add few-shot examples showing exact expected shape.",
    FailureCategory.COMPILATION_ERROR: "Add a syntax-validation self-check step before returning output; fine-tune with more executable-code examples.",
    FailureCategory.REFUSAL_ERROR: "Review safety/refusal training data for over-triggering on benign coding requests; adjust system prompt.",
    FailureCategory.OTHER: "Manually review a sample of these cases; category may need to be split further.",
}


def _build_user_prompt(question: str, model_output: str, error_context: Optional[str]) -> str:
    context_block = f"\n\nExecution/error context:\n{error_context}" if error_context else ""
    return f"""Question:
{question}

Model's response:
{model_output}{context_block}"""


class FailureTaxonomyClassifier:
    """Classifies model failures into a fixed taxonomy using an LLM judge."""

    def __init__(self, client: ChatClient, model: str = "gpt-4o-mini", temperature: float = 0.0, max_tokens: int = 512):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def classify_failure(
        self,
        example: Example,
        model_output: str,
        error_context: Optional[str] = None,
    ) -> FailureRecord:
        """Classify a single failure into the taxonomy."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(example.question, model_output, error_context)},
        ]
        raw = self.client.chat_completion(
            model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens
        )
        try:
            parsed = json.loads(raw)
            category = FailureCategory(parsed["category"])
            severity = Severity(parsed["severity"])
            explanation = parsed["explanation"]
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Failed to parse taxonomy classification for '%s': %s", example.id, raw[:300])
            category, severity, explanation = FailureCategory.OTHER, Severity.LOW, f"Unparseable judge output: {e}"

        return FailureRecord(
            example_id=example.id, category=category, explanation=explanation, severity=severity
        )

    def classify_batch(
        self, failures: list[tuple[Example, str, Optional[str]]]
    ) -> list[FailureRecord]:
        """Classify a batch of (example, model_output, error_context) failures."""
        records = [
            self.classify_failure(example, model_output, error_context)
            for example, model_output, error_context in failures
        ]
        logger.info("Classified %d failures", len(records))
        return records

    # ------------------------------------------------------------------
    # Aggregation / reporting helpers — pure functions over FailureRecords,
    # no LLM calls, safe to run repeatedly for reporting.
    # ------------------------------------------------------------------
    @staticmethod
    def failure_distribution(records: list[FailureRecord]) -> dict[str, int]:
        """Count of failures per category."""
        return dict(Counter(r.category.value for r in records))

    @staticmethod
    def priority_matrix(records: list[FailureRecord]) -> dict[str, dict[str, int]]:
        """Frequency x severity matrix: {category: {severity: count}}.

        This is the actionable artifact — a category with high frequency AND
        high severity is the top engineering priority, regardless of how it
        ranks on frequency alone.
        """
        matrix: dict[str, dict[str, int]] = defaultdict(lambda: {s.value: 0 for s in Severity})
        for r in records:
            matrix[r.category.value][r.severity.value] += 1
        return dict(matrix)

    @staticmethod
    def top_patterns(records: list[FailureRecord], top_n: int = 5) -> list[tuple[str, int]]:
        """Most common (category, count) pairs, most frequent first."""
        return Counter(r.category.value for r in records).most_common(top_n)

    @staticmethod
    def recommended_fixes(records: list[FailureRecord]) -> dict[str, str]:
        """Recommended fix direction for each category actually observed in `records`."""
        seen_categories = {r.category for r in records}
        return {cat.value: _RECOMMENDED_FIXES[cat] for cat in seen_categories}
