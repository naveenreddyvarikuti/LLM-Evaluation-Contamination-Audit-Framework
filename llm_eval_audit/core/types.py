"""Shared dataclasses and enums used across every component in the framework.

Centralizing these types avoids circular imports (e.g. the contamination
detector and the Pass@k evaluator both need to know what an `Example` looks
like) and keeps the on-disk JSONL schema in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class Difficulty(str, Enum):
    """Difficulty tier assigned to a benchmark example."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class CapabilityCategory(str, Enum):
    """What kind of capability an example is probing.

    The four custom categories map directly to the 20 private examples
    described in the project brief. CODE_GENERATION is used for public
    benchmarks like HumanEval that don't fit the private taxonomy.
    """

    FUNCTION_COMPLETION = "function_completion"
    BUG_FIXING = "bug_fixing"
    CODE_EXPLANATION = "code_explanation"
    REFACTORING = "refactoring"
    CODE_GENERATION = "code_generation"
    OTHER = "other"


class FailureCategory(str, Enum):
    """Taxonomy used by the failure classifier (Step 8)."""

    FACTUAL_ERROR = "factual_error"
    REASONING_ERROR = "reasoning_error"
    FORMAT_ERROR = "format_error"
    COMPILATION_ERROR = "compilation_error"
    REFUSAL_ERROR = "refusal_error"
    OTHER = "other"


class Severity(str, Enum):
    """Severity assigned to a classified failure."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Verdict(str, Enum):
    """Outcome of a pairwise LLM-as-judge comparison."""

    A = "A"
    B = "B"
    TIE = "tie"


def _utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (used as a default_factory)."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Example:
    """A single benchmark example (private or public).

    `test_cases` and `entry_point` are populated for code-generation-style
    examples (e.g. HumanEval) so the Pass@k evaluator can execute them;
    they are `None` for open-ended examples that go through the LLM judge.
    """

    id: str
    question: str
    reference_answer: str
    capability_category: CapabilityCategory
    difficulty: Difficulty
    metadata: dict[str, Any] = field(default_factory=dict)
    test_cases: Optional[list[str]] = None
    entry_point: Optional[str] = None
    canary: Optional[str] = None
    retired: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "capability_category": self.capability_category.value,
            "difficulty": self.difficulty.value,
            "metadata": self.metadata,
            "test_cases": self.test_cases,
            "entry_point": self.entry_point,
            "canary": self.canary,
            "retired": self.retired,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Example":
        return cls(
            id=d["id"],
            question=d["question"],
            reference_answer=d["reference_answer"],
            capability_category=CapabilityCategory(d["capability_category"]),
            difficulty=Difficulty(d["difficulty"]),
            metadata=d.get("metadata", {}),
            test_cases=d.get("test_cases"),
            entry_point=d.get("entry_point"),
            canary=d.get("canary"),
            retired=d.get("retired", False),
        )


@dataclass
class BenchmarkVersion:
    """A named, timestamped snapshot of a benchmark's examples and canaries."""

    version: str
    examples: list[Example]
    canary_strings: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utcnow_iso)
    notes: str = ""

    def active_examples(self) -> list[Example]:
        """Examples that have not been retired."""
        return [e for e in self.examples if not e.retired]


@dataclass
class NgramOverlapResult:
    """Contamination score for a single example against a training corpus."""

    example_id: str
    side: str  # "question" or "answer"
    overlap_score: float  # fraction of n-grams found in the corpus, 0..1
    matched_ngrams: int
    total_ngrams: int
    flagged: bool
    risk_level: str  # "clean" | "flagged" | "high_risk"


@dataclass
class ContaminationReport:
    """Aggregate contamination results for a benchmark version."""

    benchmark_version: str
    n: int
    threshold: float
    results: list[NgramOverlapResult]
    clean_example_ids: list[str]
    flagged_example_ids: list[str]
    high_risk_example_ids: list[str]
    generated_at: str = field(default_factory=_utcnow_iso)

    @property
    def total_examples(self) -> int:
        return len(self.clean_example_ids) + len(self.flagged_example_ids) + len(
            self.high_risk_example_ids
        )


@dataclass
class CanaryTestResult:
    """Result of testing whether one canary string was memorized by a model."""

    canary: str
    frequency: int  # how many times the canary was injected during training (0 if unknown)
    extracted: bool
    extraction_rate: float  # fraction of generations that reproduced the canary
    num_attempts: int


@dataclass
class VerbatimResult:
    """Result of testing verbatim recall of a benchmark example."""

    example_id: str
    overlap_pct: float
    memorized: bool
    prefix_used: str
    generated_continuation: str
    reference_continuation: str


@dataclass
class PassAtKResult:
    """Pass@k estimates for a single problem."""

    problem_id: str
    difficulty: Difficulty
    n_samples: int
    n_correct: int
    estimates: dict[int, float]  # k -> pass@k estimate


@dataclass
class JudgeScore:
    """Structured output of a single LLM-as-judge pairwise comparison."""

    example_id: str
    dimension_scores_a: dict[str, float]
    dimension_scores_b: dict[str, float]
    verdict: Verdict
    reasoning: str
    swapped: bool  # whether A/B order was swapped for this particular call


@dataclass
class FailureRecord:
    """A single classified failure produced by the taxonomy classifier."""

    example_id: str
    category: FailureCategory
    explanation: str
    severity: Severity


@dataclass
class PairwiseComparison:
    """One pairwise judge comparison between two named models, for Elo/win-rate analysis."""

    example_id: str
    model_a: str
    model_b: str
    verdict: Verdict
    dimension_scores_a: dict[str, float] = field(default_factory=dict)
    dimension_scores_b: dict[str, float] = field(default_factory=dict)


@dataclass
class EloRating:
    """Elo rating for a model with a bootstrap confidence interval."""

    model_name: str
    rating: float
    ci_lower: float
    ci_upper: float
    num_comparisons: int


@dataclass
class GateResult:
    """Outcome of a single regression gate check."""

    gate_name: str
    passed: bool
    actual_value: float
    threshold: float
    comparison: str  # e.g. ">=", "<="


@dataclass
class PipelineResult:
    """Aggregate output of a full evaluation pipeline run (Step 10).

    This is the single object the Report Generator (Step 11) consumes to
    render a markdown report, and what `run_evaluation.py` inspects to
    decide whether to promote a model.
    """

    model_name: str
    baseline_name: str
    benchmark_version: str
    contamination_report: ContaminationReport
    used_clean_subset: bool
    pass_at_k_results: list[PassAtKResult] = field(default_factory=list)
    pass_at_k_aggregate: dict[int, float] = field(default_factory=dict)
    pairwise_comparisons: list[PairwiseComparison] = field(default_factory=list)
    elo_ratings: dict[str, EloRating] = field(default_factory=dict)
    failure_records: list[FailureRecord] = field(default_factory=list)
    gate_results: list[GateResult] = field(default_factory=list)
    passed: bool = False
    fail_reasons: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=_utcnow_iso)
