"""Evaluation Pipeline orchestrator.

Ties every component built in Steps 2, 3, and 6-9 together into one
runnable flow:

    1. Load benchmark version.
    2. Run n-gram contamination checks; decide whether to fall back to a
       clean subset of the benchmark.
    3. Evaluate: Pass@k for examples with test cases (code generation),
       LLM-as-judge (candidate vs. baseline, with the swap trick) for
       open-ended examples.
    4. Classify every failure via the taxonomy classifier.
    5. Compute Elo ratings (candidate vs. registered baseline) with a
       bootstrap confidence interval.
    6. Check regression gates from `eval_config.yaml`.
    7. Return a `PipelineResult`: PASS (promote) or FAIL (blocked, with reasons).

All model access is injected as plain callables/clients rather than this
module constructing HuggingFace models or OpenAI clients itself — that
keeps the orchestrator testable with fakes and agnostic to which model
backend is actually used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from llm_eval_audit.analysis.failure_taxonomy import FailureTaxonomyClassifier
from llm_eval_audit.analysis.model_comparator import ModelComparator
from llm_eval_audit.benchmark.manager import BenchmarkManager
from llm_eval_audit.contamination.ngram_detector import NgramContaminationDetector
from llm_eval_audit.core.config import load_config
from llm_eval_audit.core.logging_config import get_logger
from llm_eval_audit.core.types import (
    BenchmarkVersion,
    Example,
    FailureRecord,
    GateResult,
    PairwiseComparison,
    PipelineResult,
    Severity,
    Verdict,
)
from llm_eval_audit.evaluation.llm_judge import ChatClient, LLMJudge
from llm_eval_audit.evaluation.pass_at_k import GenerateFn as CodeGenerateFn
from llm_eval_audit.evaluation.pass_at_k import PassAtKEvaluator

logger = get_logger(__name__)

# Single-turn "produce an open-ended answer" function used for judge-based examples.
AnswerFn = Callable[[str], str]


@dataclass
class PipelineDependencies:
    """Injectable model/backends the orchestrator needs; nothing here is instantiated by the pipeline itself."""

    candidate_code_generate_fn: CodeGenerateFn
    baseline_code_generate_fn: CodeGenerateFn
    candidate_answer_fn: AnswerFn
    baseline_answer_fn: AnswerFn
    judge_client: ChatClient
    corpus_hashes: set[str]


def _is_code_example(example: Example) -> bool:
    return bool(example.test_cases)


class EvaluationPipeline:
    """Orchestrates a full evaluation run for a candidate model against a registered baseline."""

    def __init__(
        self,
        deps: PipelineDependencies,
        config_path: str = "configs/eval_config.yaml",
        candidate_name: str = "candidate",
        baseline_name: str = "baseline",
    ):
        self.deps = deps
        self.config = load_config(config_path)
        self.candidate_name = candidate_name
        self.baseline_name = baseline_name

        self.benchmark_manager = BenchmarkManager(base_path=self.config["benchmark"]["path"])

        contam_cfg = self.config["contamination"]
        self.contamination_detector = NgramContaminationDetector(
            n=contam_cfg["ngram_n"],
            flag_threshold=contam_cfg["overlap_flag_threshold"],
            high_risk_threshold=contam_cfg["overlap_high_risk_threshold"],
            check_question_side=contam_cfg["check_question_side"],
            check_answer_side=contam_cfg["check_answer_side"],
        )

        pak_cfg = self.config["pass_at_k"]
        self.pass_at_k_evaluator = PassAtKEvaluator(
            generate_fn=deps.candidate_code_generate_fn,
            n_samples=pak_cfg["n_samples"],
            k_values=pak_cfg["k_values"],
            timeout_seconds=pak_cfg["timeout_seconds"],
            temperature=pak_cfg["temperature"],
        )

        judge_cfg = self.config["llm_judge"]
        self.judge = LLMJudge(
            client=deps.judge_client,
            model=self.config["models"]["judge"],
            temperature=judge_cfg["temperature"],
            max_tokens=judge_cfg["max_tokens"],
            dimensions=judge_cfg["dimensions"],
        )
        self.taxonomy_classifier = FailureTaxonomyClassifier(
            client=deps.judge_client, model=self.config["models"]["judge"]
        )

        elo_cfg = self.config["elo"]
        self.comparator = ModelComparator(
            initial_rating=elo_cfg["initial_rating"],
            k_factor=elo_cfg["k_factor"],
            bootstrap_iterations=elo_cfg["bootstrap_iterations"],
            confidence_level=elo_cfg["confidence_level"],
        )

    def run(self, benchmark_version: Optional[str] = None) -> PipelineResult:
        version = benchmark_version or self.config["benchmark"]["version"]
        bench = self.benchmark_manager.load_version(version)

        contamination_report = self.contamination_detector.scan(bench, self.deps.corpus_hashes)

        contam_cfg = self.config["contamination"]
        use_clean_subset = (
            contam_cfg["use_clean_subset_if_contaminated"]
            and len(contamination_report.high_risk_example_ids) > 0
        )
        eval_examples = self._select_examples(bench, contamination_report, use_clean_subset)

        code_examples = [e for e in eval_examples if _is_code_example(e)]
        open_examples = [e for e in eval_examples if not _is_code_example(e)]

        pass_at_k_results, pass_at_k_aggregate, failures = self._run_pass_at_k(code_examples)
        comparisons, judge_failures = self._run_judge_comparisons(open_examples)
        failures.extend(judge_failures)

        failure_records = self.taxonomy_classifier.classify_batch(failures)

        elo_ratings = self.comparator.bootstrap_elo_ci(comparisons) if comparisons else {}

        gate_results = self._check_gates(
            pass_at_k_aggregate=pass_at_k_aggregate,
            elo_ratings=elo_ratings,
            failure_records=failure_records,
            contamination_report=contamination_report,
        )
        passed = all(g.passed for g in gate_results)
        fail_reasons = [
            f"{g.gate_name}: actual={g.actual_value:.3f} vs threshold {g.comparison} {g.threshold:.3f}"
            for g in gate_results
            if not g.passed
        ]

        result = PipelineResult(
            model_name=self.candidate_name,
            baseline_name=self.baseline_name,
            benchmark_version=version,
            contamination_report=contamination_report,
            used_clean_subset=use_clean_subset,
            pass_at_k_results=pass_at_k_results,
            pass_at_k_aggregate=pass_at_k_aggregate,
            pairwise_comparisons=comparisons,
            elo_ratings=elo_ratings,
            failure_records=failure_records,
            gate_results=gate_results,
            passed=passed,
            fail_reasons=fail_reasons,
        )
        logger.info("Pipeline run complete for '%s': %s", self.candidate_name, "PASS" if passed else "FAIL")
        return result

    # ------------------------------------------------------------------
    def _select_examples(
        self, bench: BenchmarkVersion, contamination_report, use_clean_subset: bool
    ) -> list[Example]:
        active = bench.active_examples()
        if not use_clean_subset:
            return active
        clean_ids = set(contamination_report.clean_example_ids)
        logger.warning(
            "%d example(s) flagged high-risk for contamination; falling back to %d clean examples",
            len(contamination_report.high_risk_example_ids), len(clean_ids),
        )
        return [e for e in active if e.id in clean_ids]

    def _run_pass_at_k(
        self, code_examples: list[Example]
    ) -> tuple[list, dict[int, float], list[tuple[Example, str, Optional[str]]]]:
        if not code_examples:
            return [], {}, []

        results = []
        failures: list[tuple[Example, str, Optional[str]]] = []
        for problem in code_examples:
            result, outcomes, completions = self.pass_at_k_evaluator.evaluate_problem(problem)
            results.append(result)
            for outcome, completion in zip(outcomes, completions):
                if not outcome.passed:
                    failures.append((problem, completion, f"{outcome.error_type}: {outcome.detail}"))
                    break  # one representative failure per problem keeps judge calls bounded

        aggregate = self.pass_at_k_evaluator.average_estimates(results)
        return results, aggregate, failures

    def _run_judge_comparisons(
        self, open_examples: list[Example]
    ) -> tuple[list[PairwiseComparison], list[tuple[Example, str, Optional[str]]]]:
        comparisons = []
        failures: list[tuple[Example, str, Optional[str]]] = []
        for example in open_examples:
            candidate_answer = self.deps.candidate_answer_fn(example.question)
            baseline_answer = self.deps.baseline_answer_fn(example.question)
            score = self.judge.compare_with_swap(example, candidate_answer, baseline_answer)

            comparisons.append(
                PairwiseComparison(
                    example_id=example.id,
                    model_a=self.candidate_name,
                    model_b=self.baseline_name,
                    verdict=score.verdict,
                    dimension_scores_a=score.dimension_scores_a,
                    dimension_scores_b=score.dimension_scores_b,
                )
            )
            if score.verdict == Verdict.B:
                failures.append((example, candidate_answer, score.reasoning))
        return comparisons, failures

    def _check_gates(
        self,
        pass_at_k_aggregate: dict[int, float],
        elo_ratings: dict,
        failure_records: list[FailureRecord],
        contamination_report,
    ) -> list[GateResult]:
        gates_cfg = self.config["regression_gates"]
        results: list[GateResult] = []

        if pass_at_k_aggregate:
            pass_at_1 = pass_at_k_aggregate.get(1, 0.0)
            results.append(
                GateResult(
                    gate_name="min_pass_at_1",
                    passed=pass_at_1 >= gates_cfg["min_pass_at_1"],
                    actual_value=pass_at_1,
                    threshold=gates_cfg["min_pass_at_1"],
                    comparison=">=",
                )
            )

        if self.candidate_name in elo_ratings and self.baseline_name in elo_ratings:
            drop = elo_ratings[self.baseline_name].rating - elo_ratings[self.candidate_name].rating
            results.append(
                GateResult(
                    gate_name="max_elo_drop_vs_baseline",
                    passed=drop <= gates_cfg["max_elo_drop_vs_baseline"],
                    actual_value=drop,
                    threshold=gates_cfg["max_elo_drop_vs_baseline"],
                    comparison="<=",
                )
            )

        if failure_records:
            high_severity_rate = sum(1 for f in failure_records if f.severity == Severity.HIGH) / len(failure_records)
            results.append(
                GateResult(
                    gate_name="max_high_severity_failure_rate",
                    passed=high_severity_rate <= gates_cfg["max_high_severity_failure_rate"],
                    actual_value=high_severity_rate,
                    threshold=gates_cfg["max_high_severity_failure_rate"],
                    comparison="<=",
                )
            )

        if contamination_report.total_examples > 0:
            high_risk_rate = len(contamination_report.high_risk_example_ids) / contamination_report.total_examples
            results.append(
                GateResult(
                    gate_name="max_contamination_high_risk_rate",
                    passed=high_risk_rate <= gates_cfg["max_contamination_high_risk_rate"],
                    actual_value=high_risk_rate,
                    threshold=gates_cfg["max_contamination_high_risk_rate"],
                    comparison="<=",
                )
            )

        return results
