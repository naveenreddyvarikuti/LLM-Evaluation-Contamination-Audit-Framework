from llm_eval_audit.analysis.failure_taxonomy import FailureTaxonomyClassifier
from llm_eval_audit.core.types import FailureCategory, FailureRecord, Severity


def _records():
    return [
        FailureRecord("e1", FailureCategory.REASONING_ERROR, "logic wrong", Severity.HIGH),
        FailureRecord("e2", FailureCategory.REASONING_ERROR, "logic wrong again", Severity.MEDIUM),
        FailureRecord("e3", FailureCategory.FORMAT_ERROR, "bad shape", Severity.LOW),
        FailureRecord("e4", FailureCategory.REFUSAL_ERROR, "declined", Severity.HIGH),
    ]


def test_failure_distribution_counts_by_category():
    dist = FailureTaxonomyClassifier.failure_distribution(_records())
    assert dist["reasoning_error"] == 2
    assert dist["format_error"] == 1
    assert dist["refusal_error"] == 1


def test_priority_matrix_breaks_down_by_severity():
    matrix = FailureTaxonomyClassifier.priority_matrix(_records())
    assert matrix["reasoning_error"]["high"] == 1
    assert matrix["reasoning_error"]["medium"] == 1
    assert matrix["reasoning_error"]["low"] == 0


def test_top_patterns_orders_by_frequency():
    top = FailureTaxonomyClassifier.top_patterns(_records(), top_n=1)
    assert top[0][0] == "reasoning_error"
    assert top[0][1] == 2


def test_recommended_fixes_only_covers_observed_categories():
    fixes = FailureTaxonomyClassifier.recommended_fixes(_records())
    assert set(fixes.keys()) == {"reasoning_error", "format_error", "refusal_error"}
