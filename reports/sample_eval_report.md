# Evaluation Report: Qwen/Qwen2.5-Coder-1.5B vs. Qwen/Qwen2.5-Coder-0.5B

**Overall Status:** PASS

## Run Metadata

- Candidate model: Qwen/Qwen2.5-Coder-1.5B
- Baseline model: Qwen/Qwen2.5-Coder-0.5B
- Benchmark version: v1
- Generated at: 2026-07-06T09:57:48.944006+00:00

## Contamination Analysis

- N-gram size: 13
- Flag threshold: 0.2
- Total examples scanned: 19
- Clean: 18
- Flagged: 1
- High-risk: 0
- Clean subset used for evaluation: No

## Evaluation Results

### LLM-as-Judge (open-ended, pairwise vs. baseline)

- Candidate (Qwen/Qwen2.5-Coder-1.5B) win rate: 65.0%
- Baseline (Qwen/Qwen2.5-Coder-0.5B) win rate: 35.0%

| Dimension | Qwen/Qwen2.5-Coder-1.5B | Qwen/Qwen2.5-Coder-0.5B |
|---|---|---|
| code_quality | 4.10 | 3.00 |
| completeness | 4.00 | 3.40 |
| correctness | 4.20 | 3.10 |
| explanation_clarity | 4.30 | 3.20 |

## Model Comparison (Elo)

| Model | Elo | 95% CI | # Comparisons |
|---|---|---|---|
| Qwen/Qwen2.5-Coder-0.5B | 1451.1 | [1380.0, 1515.9] | 20 |
| Qwen/Qwen2.5-Coder-1.5B | 1548.9 | [1484.1, 1620.0] | 20 |

**Statistically distinguishable from baseline:** No (confidence intervals overlap)

## Failure Analysis

### Distribution

| Category | Count |
|---|---|
| format_error | 1 |
| reasoning_error | 1 |

### Priority Matrix (frequency x severity)

| Category | High | Medium | Low |
|---|---|---|---|
| format_error | 0 | 1 | 0 |
| reasoning_error | 1 | 0 | 0 |

### Top Failure Patterns

- **format_error**: 1 occurrence(s)
- **reasoning_error**: 1 occurrence(s)

### Recommended Fix Directions

- **reasoning_error**: Add chain-of-thought / self-consistency prompting; fine-tune on harder algorithmic examples with step-by-step solutions.
- **format_error**: Tighten prompt instructions on required output format; add few-shot examples showing exact expected shape.

## Regression Gates

| Gate | Status | Actual | Threshold |
|---|---|---|---|
| max_elo_drop_vs_baseline | PASS | -97.747 | <= 25.000 |
| max_high_severity_failure_rate | PASS | 0.250 | <= 0.600 |
| max_contamination_high_risk_rate | PASS | 0.000 | <= 0.050 |

## Recommended Next Steps

- Promote this model checkpoint as the new baseline for future comparisons.
- Archive this report alongside the model artifact for auditability.
- Continue monitoring the flagged/high-risk contamination examples in future benchmark versions.
