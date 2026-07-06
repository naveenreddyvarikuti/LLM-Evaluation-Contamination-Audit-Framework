"""N-gram Contamination Detector.

Detects whether benchmark examples (question and/or reference answer) leak
into a training corpus by comparing hashed n-grams. Built for code text:
normalization strips comments/whitespace noise so that trivial reformatting
doesn't hide a real match, and doesn't create false matches out of shared
boilerplate.

Why N=13 for code: short n-grams (n=5-8) match on extremely common
boilerplate (`def __init__(self):`, `for i in range(`), producing high false
positive rates. N=13 tokens is long enough that a match is very unlikely to
occur by chance in unrelated code, while still being short enough to catch
a benchmark solution copy-pasted (even with minor edits) into training data.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from llm_eval_audit.core.logging_config import get_logger
from llm_eval_audit.core.types import BenchmarkVersion, ContaminationReport, NgramOverlapResult

logger = get_logger(__name__)

_COMMENT_LINE_RE = re.compile(r"#.*$", re.MULTILINE)
_COMMENT_BLOCK_RE = re.compile(r'("""|\'\'\')(?:(?!\1).)*\1', re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_code(text: str) -> str:
    """Normalize code text before hashing: lowercase, strip comments/docstrings,
    and collapse all whitespace to single spaces.

    This is deliberately aggressive — the goal of contamination detection is
    to catch near-verbatim reuse, and comment/whitespace differences are the
    most common way a copy-paste is trivially "changed" without changing the
    actual logic.
    """
    text = _COMMENT_BLOCK_RE.sub(" ", text)
    text = _COMMENT_LINE_RE.sub(" ", text)
    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _tokenize(text: str) -> list[str]:
    """Split normalized text into whitespace tokens for n-gram construction."""
    return text.split(" ") if text else []


def _hash_ngram(ngram: tuple[str, ...]) -> str:
    """Hash an n-gram tuple to a fixed-size digest for O(1) set membership."""
    joined = " ".join(ngram)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_ngram_hash_set(corpus_texts: list[str], n: int) -> set[str]:
    """Build a set of hashed n-grams from a training corpus.

    This is the expensive one-time cost (O(corpus size)); every subsequent
    membership check against it is O(1) thanks to Python's hash set.
    """
    hashes: set[str] = set()
    for text in corpus_texts:
        tokens = _tokenize(normalize_code(text))
        for i in range(len(tokens) - n + 1):
            hashes.add(_hash_ngram(tuple(tokens[i : i + n])))
    logger.info("Built corpus n-gram hash set: %d unique %d-grams from %d documents", len(hashes), n, len(corpus_texts))
    return hashes


@dataclass
class NgramMatchDetail:
    """Fraction of an example's n-grams found in the corpus hash set."""

    matched: int
    total: int

    @property
    def score(self) -> float:
        return self.matched / self.total if self.total > 0 else 0.0


def _score_text(text: str, n: int, corpus_hashes: set[str]) -> NgramMatchDetail:
    tokens = _tokenize(normalize_code(text))
    total = max(len(tokens) - n + 1, 0)
    if total == 0:
        # Text shorter than n tokens: nothing to hash, treat as clean (0 matches / 0 total).
        return NgramMatchDetail(matched=0, total=0)
    matched = sum(
        1 for i in range(total) if _hash_ngram(tuple(tokens[i : i + n])) in corpus_hashes
    )
    return NgramMatchDetail(matched=matched, total=total)


class NgramContaminationDetector:
    """Compares benchmark examples against a training corpus via hashed n-gram overlap."""

    def __init__(
        self,
        n: int = 13,
        flag_threshold: float = 0.20,
        high_risk_threshold: float = 0.50,
        check_question_side: bool = True,
        check_answer_side: bool = True,
    ):
        if not (0.0 <= flag_threshold <= high_risk_threshold <= 1.0):
            raise ValueError("Require 0 <= flag_threshold <= high_risk_threshold <= 1")
        self.n = n
        self.flag_threshold = flag_threshold
        self.high_risk_threshold = high_risk_threshold
        self.check_question_side = check_question_side
        self.check_answer_side = check_answer_side

    def _risk_level(self, score: float) -> str:
        if score >= self.high_risk_threshold:
            return "high_risk"
        if score >= self.flag_threshold:
            return "flagged"
        return "clean"

    def scan(
        self,
        bench: BenchmarkVersion,
        corpus_hashes: set[str],
    ) -> ContaminationReport:
        """Scan every active example in a benchmark version against a corpus hash set.

        Returns a ContaminationReport with per-example results plus clean /
        flagged / high-risk breakdowns, so downstream code (e.g. the
        pipeline orchestrator) can decide whether to fall back to a clean
        subset of the benchmark.
        """
        results: list[NgramOverlapResult] = []
        clean_ids, flagged_ids, high_risk_ids = [], [], []

        for example in bench.active_examples():
            sides_to_check: list[tuple[str, str]] = []
            if self.check_question_side:
                sides_to_check.append(("question", example.question))
            if self.check_answer_side:
                sides_to_check.append(("answer", example.reference_answer))

            worst_score = 0.0
            for side_name, side_text in sides_to_check:
                detail = _score_text(side_text, self.n, corpus_hashes)
                risk = self._risk_level(detail.score)
                results.append(
                    NgramOverlapResult(
                        example_id=example.id,
                        side=side_name,
                        overlap_score=detail.score,
                        matched_ngrams=detail.matched,
                        total_ngrams=detail.total,
                        flagged=risk != "clean",
                        risk_level=risk,
                    )
                )
                worst_score = max(worst_score, detail.score)

            overall_risk = self._risk_level(worst_score)
            if overall_risk == "high_risk":
                high_risk_ids.append(example.id)
            elif overall_risk == "flagged":
                flagged_ids.append(example.id)
            else:
                clean_ids.append(example.id)

        report = ContaminationReport(
            benchmark_version=bench.version,
            n=self.n,
            threshold=self.flag_threshold,
            results=results,
            clean_example_ids=clean_ids,
            flagged_example_ids=flagged_ids,
            high_risk_example_ids=high_risk_ids,
        )
        logger.info(
            "Contamination scan for '%s': %d clean, %d flagged, %d high-risk (of %d)",
            bench.version, len(clean_ids), len(flagged_ids), len(high_risk_ids), report.total_examples,
        )
        return report
