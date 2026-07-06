"""Pass@k Evaluator for code generation models.

Implements the unbiased Pass@k estimator from Chen et al. 2021
("Evaluating Large Language Models Trained on Code"), which avoids the bias
of the naive "run k samples, check if any pass" approach: with only k
samples, an unlucky draw can massively undercount a model's true pass rate.
Instead we generate n >> k samples once, count how many `c` pass, and
compute the expected pass rate over all C(n, k) ways of choosing k of them:

    pass@k = E[1 - C(n - c, k) / C(n, k)]

Each problem's generated candidates are executed in an isolated subprocess
with a hard timeout, since generated code is untrusted and can contain
infinite loops, side effects, or outright malicious code.
"""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable

from llm_eval_audit.core.logging_config import get_logger
from llm_eval_audit.core.types import BenchmarkVersion, Difficulty, Example, PassAtKResult

logger = get_logger(__name__)

GenerateFn = Callable[[str, int, float], list[str]]
"""Callable(prompt, n_samples, temperature) -> list of n generated completions."""


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """Chen et al. 2021 unbiased Pass@k estimator.

    n: total samples generated: c: number that passed all tests: k: the
    "pass@k" we're estimating. Computed via the complement (probability that
    ALL k sampled-without-replacement solutions fail) for numerical
    stability, rather than summing the combinatorial terms directly.
    """
    if n < k:
        raise ValueError(f"n ({n}) must be >= k ({k})")
    if c == 0:
        return 0.0
    if n - c < k:
        # Fewer failing samples than k means it's impossible to pick k failures.
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


@dataclass
class ExecutionOutcome:
    """Result of running one generated candidate against a problem's tests."""

    passed: bool
    error_type: str  # "none" | "syntax_error" | "runtime_error" | "test_failure" | "timeout"
    detail: str


def execute_candidate(
    prompt: str,
    completion: str,
    test_cases: list[str],
    entry_point: str | None,
    timeout_seconds: int = 10,
) -> ExecutionOutcome:
    """Run one generated candidate in an isolated subprocess with a timeout.

    Assembles `prompt + completion` (the candidate solution) followed by the
    problem's test code, then executes it with a fresh Python interpreter
    subprocess so a runaway or malicious completion can't affect the parent
    process. `entry_point`, if given, is invoked via `check(entry_point)`
    per the HumanEval test convention.
    """
    full_program = prompt + completion + "\n" + "\n".join(test_cases)
    if entry_point:
        full_program += f"\ncheck({entry_point})\n"

    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", full_program],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return ExecutionOutcome(passed=False, error_type="timeout", detail=f"Exceeded {timeout_seconds}s")

    if result.returncode == 0:
        return ExecutionOutcome(passed=True, error_type="none", detail="")

    stderr = result.stderr.strip()
    if "SyntaxError" in stderr or "IndentationError" in stderr:
        return ExecutionOutcome(passed=False, error_type="syntax_error", detail=stderr[-500:])
    if "AssertionError" in stderr:
        return ExecutionOutcome(passed=False, error_type="test_failure", detail=stderr[-500:])
    return ExecutionOutcome(passed=False, error_type="runtime_error", detail=stderr[-500:])


class PassAtKEvaluator:
    """Evaluates a code-generation model's Pass@k on a benchmark of coding problems."""

    def __init__(
        self,
        generate_fn: GenerateFn,
        n_samples: int = 200,
        k_values: list[int] | None = None,
        timeout_seconds: int = 10,
        temperature: float = 0.8,
    ):
        self.generate_fn = generate_fn
        self.n_samples = n_samples
        self.k_values = k_values or [1, 5, 10, 100]
        if max(self.k_values) > n_samples:
            raise ValueError(
                f"n_samples ({n_samples}) must be >= max(k_values) ({max(self.k_values)}) "
                "for the unbiased estimator to be defined"
            )
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature

    def evaluate_problem(self, problem: Example) -> tuple[PassAtKResult, list[ExecutionOutcome], list[str]]:
        """Generate `n_samples` completions for one problem and compute Pass@k.

        Returns the aggregated `PassAtKResult`, the raw per-sample
        `ExecutionOutcome` list, and the completions themselves (in the same
        order), since the failure taxonomy classifier (Step 8) wants to see
        *why* and *what* individual samples failed, not just the aggregate
        pass rate — and re-generating completions to recover that would be
        both wasteful and, under sampling, non-deterministic.
        """
        completions = self.generate_fn(problem.question, self.n_samples, self.temperature)
        outcomes = [
            execute_candidate(
                prompt=problem.question,
                completion=completion,
                test_cases=problem.test_cases or [],
                entry_point=problem.entry_point,
                timeout_seconds=self.timeout_seconds,
            )
            for completion in completions
        ]
        n_correct = sum(1 for o in outcomes if o.passed)

        estimates = {
            k: unbiased_pass_at_k(len(outcomes), n_correct, k)
            for k in self.k_values
            if k <= len(outcomes)
        }
        result = PassAtKResult(
            problem_id=problem.id,
            difficulty=problem.difficulty,
            n_samples=len(outcomes),
            n_correct=n_correct,
            estimates=estimates,
        )
        logger.info(
            "Problem '%s': %d/%d passed | pass@1=%.3f",
            problem.id, n_correct, len(outcomes), estimates.get(1, float("nan")),
        )
        return result, outcomes, completions

    def evaluate_benchmark(
        self, bench: BenchmarkVersion
    ) -> tuple[list[PassAtKResult], dict[int, float], dict[str, dict[int, float]]]:
        """Evaluate every active problem in a benchmark version.

        Returns:
            - per-problem PassAtKResult list
            - aggregate pass@k averaged across all problems
            - pass@k broken down by difficulty tier
        """
        problem_results: list[PassAtKResult] = []
        for problem in bench.active_examples():
            result, _, _ = self.evaluate_problem(problem)
            problem_results.append(result)

        aggregate = self.average_estimates(problem_results)
        by_difficulty: dict[str, dict[int, float]] = {}
        for tier in Difficulty:
            tier_results = [r for r in problem_results if r.difficulty == tier]
            if tier_results:
                by_difficulty[tier.value] = self.average_estimates(tier_results)

        logger.info("Benchmark Pass@k aggregate: %s", aggregate)
        return problem_results, aggregate, by_difficulty

    def average_estimates(self, results: list[PassAtKResult]) -> dict[int, float]:
        aggregate: dict[int, float] = {}
        for k in self.k_values:
            values = [r.estimates[k] for r in results if k in r.estimates]
            if values:
                aggregate[k] = sum(values) / len(values)
        return aggregate
