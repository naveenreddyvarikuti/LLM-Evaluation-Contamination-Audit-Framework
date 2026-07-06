"""Model Comparator with Elo ratings.

Turns a list of pairwise LLM-judge verdicts (from Step 7) between named
models into Elo ratings, analogous to how chess ratings are derived from
game outcomes. Elo is preferred over a simple win-rate here because it
accounts for strength of opposition — beating a strong baseline should move
the rating more than beating a weak one — and composes across more than two
models if the framework grows to compare several checkpoints at once.

Confidence intervals come from bootstrap resampling: since Elo is a
non-linear, order-dependent function of the comparison sequence, there's no
simple closed-form standard error, so we resample the comparison list with
replacement many times, recompute Elo each time, and take percentiles of the
resulting distribution.
"""

from __future__ import annotations

import random
from collections import defaultdict

from llm_eval_audit.core.logging_config import get_logger
from llm_eval_audit.core.types import EloRating, PairwiseComparison, Verdict

logger = get_logger(__name__)


def _expected_score(rating_a: float, rating_b: float) -> float:
    """Standard Elo expected-score formula: probability A beats B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def _verdict_score(verdict: Verdict) -> float:
    """Map a verdict to a numeric score for model A: win=1, tie=0.5, loss=0."""
    return {Verdict.A: 1.0, Verdict.TIE: 0.5, Verdict.B: 0.0}[verdict]


class ModelComparator:
    """Computes Elo ratings, bootstrap confidence intervals, and win rates from pairwise comparisons."""

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 32.0,
        bootstrap_iterations: int = 1000,
        confidence_level: float = 0.95,
        random_seed: int | None = None,
    ):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.bootstrap_iterations = bootstrap_iterations
        self.confidence_level = confidence_level
        self._rng = random.Random(random_seed)

    def compute_elo(self, comparisons: list[PairwiseComparison]) -> dict[str, float]:
        """Single-pass Elo computation over comparisons, processed in the given order.

        Every model seen starts at `initial_rating`. Each comparison updates
        both participants' ratings via the standard Elo update rule.
        """
        ratings: dict[str, float] = defaultdict(lambda: self.initial_rating)
        for comp in comparisons:
            ra, rb = ratings[comp.model_a], ratings[comp.model_b]
            expected_a = _expected_score(ra, rb)
            score_a = _verdict_score(comp.verdict)
            ratings[comp.model_a] = ra + self.k_factor * (score_a - expected_a)
            ratings[comp.model_b] = rb + self.k_factor * ((1 - score_a) - (1 - expected_a))
        return dict(ratings)

    def bootstrap_elo_ci(self, comparisons: list[PairwiseComparison]) -> dict[str, EloRating]:
        """Bootstrap-resample the comparison list to compute Elo ratings with a confidence interval.

        Each of `bootstrap_iterations` resamples draws len(comparisons) comparisons
        with replacement AND shuffles them before computing Elo, since Elo is
        order-dependent and we want the CI to reflect uncertainty in the
        underlying win/loss/tie outcomes, not artifacts of a fixed ordering.
        """
        if not comparisons:
            return {}

        model_names = {c.model_a for c in comparisons} | {c.model_b for c in comparisons}
        samples: dict[str, list[float]] = defaultdict(list)

        for _ in range(self.bootstrap_iterations):
            resample = [self._rng.choice(comparisons) for _ in range(len(comparisons))]
            self._rng.shuffle(resample)
            ratings = self.compute_elo(resample)
            for name in model_names:
                samples[name].append(ratings.get(name, self.initial_rating))

        point_estimate = self.compute_elo(comparisons)

        alpha = 1 - self.confidence_level
        lower_pct = alpha / 2 * 100
        upper_pct = (1 - alpha / 2) * 100

        results: dict[str, EloRating] = {}
        for name in model_names:
            values = sorted(samples[name])
            lower = _percentile(values, lower_pct)
            upper = _percentile(values, upper_pct)
            num_comparisons = sum(1 for c in comparisons if c.model_a == name or c.model_b == name)
            results[name] = EloRating(
                model_name=name,
                rating=point_estimate[name],
                ci_lower=lower,
                ci_upper=upper,
                num_comparisons=num_comparisons,
            )
        logger.info("Elo ratings (with %d%% CI): %s", int(self.confidence_level * 100), results)
        return results

    @staticmethod
    def win_rates(comparisons: list[PairwiseComparison]) -> dict[str, float]:
        """Fraction of comparisons each model wins (ties count as 0.5 for each side)."""
        wins: dict[str, float] = defaultdict(float)
        totals: dict[str, int] = defaultdict(int)
        for c in comparisons:
            score_a = _verdict_score(c.verdict)
            wins[c.model_a] += score_a
            wins[c.model_b] += 1 - score_a
            totals[c.model_a] += 1
            totals[c.model_b] += 1
        return {name: wins[name] / totals[name] for name in totals}

    @staticmethod
    def per_dimension_comparison(comparisons: list[PairwiseComparison]) -> dict[str, dict[str, float]]:
        """Average dimension scores per model, across all comparisons they appeared in."""
        sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for c in comparisons:
            for dim, score in c.dimension_scores_a.items():
                sums[c.model_a][dim] += score
                counts[c.model_a][dim] += 1
            for dim, score in c.dimension_scores_b.items():
                sums[c.model_b][dim] += score
                counts[c.model_b][dim] += 1
        return {
            model: {dim: sums[model][dim] / counts[model][dim] for dim in sums[model]}
            for model in sums
        }

    @staticmethod
    def is_distinguishable(rating_a: EloRating, rating_b: EloRating) -> bool:
        """Two models are statistically distinguishable if their Elo confidence intervals don't overlap."""
        return rating_a.ci_upper < rating_b.ci_lower or rating_b.ci_upper < rating_a.ci_lower


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolated percentile of a pre-sorted list (no numpy dependency needed here)."""
    if not sorted_values:
        return float("nan")
    k = (len(sorted_values) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)
