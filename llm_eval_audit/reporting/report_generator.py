"""Report Generator: renders a `PipelineResult` into a markdown evaluation report.

Kept as pure string-building over an already-computed `PipelineResult` — it
does not call any model or re-run any evaluation, so a report can always be
regenerated from a stored result without re-incurring API/GPU cost.
"""

from __future__ import annotations

from pathlib import Path

from llm_eval_audit.analysis.failure_taxonomy import FailureTaxonomyClassifier
from llm_eval_audit.analysis.model_comparator import ModelComparator
from llm_eval_audit.core.types import PipelineResult

_NEXT_STEPS_IF_FAILED = (
    "- Investigate each failed gate listed above before considering promotion.\n"
    "- Re-run contamination checks after removing/replacing any high-risk benchmark examples.\n"
    "- Prioritize fixes for the highest-frequency x highest-severity failure categories first "
    "(see Priority Matrix)."
)
_NEXT_STEPS_IF_PASSED = (
    "- Promote this model checkpoint as the new baseline for future comparisons.\n"
    "- Archive this report alongside the model artifact for auditability.\n"
    "- Continue monitoring the flagged/high-risk contamination examples in future benchmark versions."
)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _render_contamination_section(result: PipelineResult) -> str:
    report = result.contamination_report
    lines = [
        "## Contamination Analysis",
        "",
        f"- N-gram size: {report.n}",
        f"- Flag threshold: {report.threshold}",
        f"- Total examples scanned: {report.total_examples}",
        f"- Clean: {len(report.clean_example_ids)}",
        f"- Flagged: {len(report.flagged_example_ids)}",
        f"- High-risk: {len(report.high_risk_example_ids)}",
        f"- Clean subset used for evaluation: {'Yes' if result.used_clean_subset else 'No'}",
        "",
    ]
    if report.high_risk_example_ids:
        lines.append("**High-risk example IDs:** " + ", ".join(report.high_risk_example_ids))
        lines.append("")
    return "\n".join(lines)


def _render_evaluation_section(result: PipelineResult) -> str:
    lines = ["## Evaluation Results", ""]
    if result.pass_at_k_aggregate:
        lines.append("### Pass@k (code generation)")
        lines.append("")
        lines.append("| k | Pass@k |")
        lines.append("|---|--------|")
        for k in sorted(result.pass_at_k_aggregate):
            lines.append(f"| {k} | {_fmt_pct(result.pass_at_k_aggregate[k])} |")
        lines.append("")
    if result.pairwise_comparisons:
        lines.append("### LLM-as-Judge (open-ended, pairwise vs. baseline)")
        lines.append("")
        win_rates = ModelComparator.win_rates(result.pairwise_comparisons)
        dim_scores = ModelComparator.per_dimension_comparison(result.pairwise_comparisons)
        lines.append(f"- Candidate ({result.model_name}) win rate: {_fmt_pct(win_rates.get(result.model_name, 0.0))}")
        lines.append(f"- Baseline ({result.baseline_name}) win rate: {_fmt_pct(win_rates.get(result.baseline_name, 0.0))}")
        lines.append("")
        lines.append("| Dimension | " + result.model_name + " | " + result.baseline_name + " |")
        lines.append("|---|---|---|")
        all_dims = set(dim_scores.get(result.model_name, {})) | set(dim_scores.get(result.baseline_name, {}))
        for dim in sorted(all_dims):
            a_score = dim_scores.get(result.model_name, {}).get(dim, float("nan"))
            b_score = dim_scores.get(result.baseline_name, {}).get(dim, float("nan"))
            lines.append(f"| {dim} | {a_score:.2f} | {b_score:.2f} |")
        lines.append("")
    if not result.pass_at_k_aggregate and not result.pairwise_comparisons:
        lines.append("_No evaluation results (empty benchmark subset after contamination filtering)._")
        lines.append("")
    return "\n".join(lines)


def _render_elo_section(result: PipelineResult) -> str:
    if not result.elo_ratings:
        return ""
    lines = ["## Model Comparison (Elo)", "", "| Model | Elo | 95% CI | # Comparisons |", "|---|---|---|---|"]
    for name, rating in result.elo_ratings.items():
        lines.append(
            f"| {name} | {rating.rating:.1f} | [{rating.ci_lower:.1f}, {rating.ci_upper:.1f}] | {rating.num_comparisons} |"
        )
    lines.append("")

    if result.model_name in result.elo_ratings and result.baseline_name in result.elo_ratings:
        distinguishable = ModelComparator.is_distinguishable(
            result.elo_ratings[result.model_name], result.elo_ratings[result.baseline_name]
        )
        lines.append(
            f"**Statistically distinguishable from baseline:** {'Yes' if distinguishable else 'No (confidence intervals overlap)'}"
        )
        lines.append("")
    return "\n".join(lines)


def _render_failure_section(result: PipelineResult) -> str:
    if not result.failure_records:
        return "## Failure Analysis\n\n_No failures recorded._\n"

    records = result.failure_records
    distribution = FailureTaxonomyClassifier.failure_distribution(records)
    priority_matrix = FailureTaxonomyClassifier.priority_matrix(records)
    top_patterns = FailureTaxonomyClassifier.top_patterns(records)
    fixes = FailureTaxonomyClassifier.recommended_fixes(records)

    lines = ["## Failure Analysis", "", "### Distribution", ""]
    lines.append("| Category | Count |")
    lines.append("|---|---|")
    for category, count in distribution.items():
        lines.append(f"| {category} | {count} |")
    lines.append("")

    lines.append("### Priority Matrix (frequency x severity)")
    lines.append("")
    lines.append("| Category | High | Medium | Low |")
    lines.append("|---|---|---|---|")
    for category, sev_counts in priority_matrix.items():
        lines.append(f"| {category} | {sev_counts['high']} | {sev_counts['medium']} | {sev_counts['low']} |")
    lines.append("")

    lines.append("### Top Failure Patterns")
    lines.append("")
    for category, count in top_patterns:
        lines.append(f"- **{category}**: {count} occurrence(s)")
    lines.append("")

    lines.append("### Recommended Fix Directions")
    lines.append("")
    for category, fix in fixes.items():
        lines.append(f"- **{category}**: {fix}")
    lines.append("")

    return "\n".join(lines)


def _render_gates_section(result: PipelineResult) -> str:
    lines = ["## Regression Gates", ""]
    if not result.gate_results:
        lines.append("_No gates were evaluated (insufficient data)._")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Gate | Status | Actual | Threshold |")
    lines.append("|---|---|---|---|")
    for gate in result.gate_results:
        status = "PASS" if gate.passed else "FAIL"
        lines.append(f"| {gate.gate_name} | {status} | {gate.actual_value:.3f} | {gate.comparison} {gate.threshold:.3f} |")
    lines.append("")
    return "\n".join(lines)


def generate_report(result: PipelineResult, output_path: str | Path | None = None) -> str:
    """Render a `PipelineResult` into a full markdown report and optionally write it to disk."""
    overall_status = "PASS" if result.passed else "FAIL"
    lines = [
        f"# Evaluation Report: {result.model_name} vs. {result.baseline_name}",
        "",
        f"**Overall Status:** {overall_status}",
        "",
        "## Run Metadata",
        "",
        f"- Candidate model: {result.model_name}",
        f"- Baseline model: {result.baseline_name}",
        f"- Benchmark version: {result.benchmark_version}",
        f"- Generated at: {result.generated_at}",
        "",
        _render_contamination_section(result),
        _render_evaluation_section(result),
        _render_elo_section(result),
        _render_failure_section(result),
        _render_gates_section(result),
        "## Recommended Next Steps",
        "",
        _NEXT_STEPS_IF_PASSED if result.passed else _NEXT_STEPS_IF_FAILED,
        "",
    ]
    if not result.passed and result.fail_reasons:
        lines.insert(4, "**Fail reasons:**\n" + "\n".join(f"- {r}" for r in result.fail_reasons) + "\n")

    markdown = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")

    return markdown
