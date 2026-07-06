from llm_eval_audit.contamination.ngram_detector import (
    NgramContaminationDetector,
    build_ngram_hash_set,
    normalize_code,
)
from llm_eval_audit.core.types import BenchmarkVersion, CapabilityCategory, Difficulty, Example


def test_normalize_code_strips_comments_and_whitespace():
    code = "def f(x):  # a comment\n    return x   +   1\n"
    normalized = normalize_code(code)
    assert "#" not in normalized
    assert "comment" not in normalized
    assert "  " not in normalized


def test_detector_flags_exact_copy_as_high_risk():
    solution = " ".join(f"line_{i} = {i}" for i in range(20))
    corpus_hashes = build_ngram_hash_set([solution], n=5)

    example = Example(
        id="ex1",
        question="irrelevant question text with no overlap at all whatsoever here",
        reference_answer=solution,
        capability_category=CapabilityCategory.FUNCTION_COMPLETION,
        difficulty=Difficulty.EASY,
    )
    bench = BenchmarkVersion(version="test", examples=[example])

    detector = NgramContaminationDetector(n=5, flag_threshold=0.2, high_risk_threshold=0.5)
    report = detector.scan(bench, corpus_hashes)

    assert "ex1" in report.high_risk_example_ids


def test_detector_marks_unrelated_text_as_clean():
    corpus_hashes = build_ngram_hash_set(["completely unrelated corpus text " * 5], n=13)
    example = Example(
        id="ex2",
        question="a totally different question about something else entirely, unrelated",
        reference_answer="a totally different answer with different words and structure, unrelated",
        capability_category=CapabilityCategory.FUNCTION_COMPLETION,
        difficulty=Difficulty.EASY,
    )
    bench = BenchmarkVersion(version="test", examples=[example])
    detector = NgramContaminationDetector(n=13)
    report = detector.scan(bench, corpus_hashes)

    assert "ex2" in report.clean_example_ids
