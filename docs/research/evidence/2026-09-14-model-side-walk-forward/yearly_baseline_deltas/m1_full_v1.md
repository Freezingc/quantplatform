# Per-year baseline RankIC on the candidate's day set: M1

Source report `/data/zhangyudong/qrv2_research_artifacts/m1_full_v1/report.json` (sha256 `6da61acaa08555ae`).
Both acceptance checks passed bit-for-bit, so the per-year baseline column is a
partition of the pooled baseline mean the signed report already publishes.

| Year | Candidate mean RankIC | Days | Baseline on the same days | Delta | Baseline before coverage matching | Days | Days lost to matching | Baseline mean there |
|---|---|---|---|---|---|---|---|---|
| 2022 | 0.091083 | 241 | 0.080637 | +0.010446 | 0.080637 | 241 | 0 | - |
| 2023 | 0.073011 | 219 | 0.069806 | +0.003205 | 0.069806 | 219 | 0 | - |
| 2024 | 0.105018 | 242 | 0.090604 | +0.014414 | 0.090604 | 242 | 0 | - |
| 2025 | 0.101157 | 243 | 0.098322 | +0.002835 | 0.098322 | 243 | 0 | - |

Pooled: candidate 0.093054 over 945 days, baseline 0.085227 over 945 days, delta +0.007827. Before coverage matching the same baseline artifact averages 0.085227 over 945 days.

These columns are all computed inside this candidate's workspace, on this candidate's label frame. The baseline run's OWN report is a separate evaluation on a different (dense-only) dataset and a different valid-day count; its numbers are not reproduced here and must not be mixed into this table.

Acceptance: A the replay reproduces the signed report bit for bit; B the replayed baseline branch reproduces the evaluator's pooled baseline mean; C the day-weighted yearly deltas rebuild the published pooled delta to 1e-12; D the valid-day ratio rebuilds the report's G6 to 1e-12.

## Days the candidate cannot be evaluated on

The label frame offers a scoreable cross-section on 969 OOS days (that count is G6's denominator). The candidate is evaluated on 945 of them and is unevaluable on 24; the baseline scores carried in this same workspace are unevaluable on 24 of them.

| Year | Label dates | Candidate valid | Candidate unevaluable | Baseline unevaluable (same frame) | Median coverage on those days | Median scored rows | Median valid-label rows |
|---|---|---|---|---|---|---|---|
| 2022 | 242 | 241 | 1 | 1 | 0.8951 | 3003.0000 | 3355.0000 |
| 2023 | 242 | 219 | 23 | 23 | 0.8989 | 3255.0000 | 3623.0000 |
| 2024 | 242 | 242 | 0 | 0 | - | - | - |
| 2025 | 243 | 243 | 0 | 0 | - | - | - |

A day is unevaluable when fewer than 30 rows carry both a valid label and a finite score, or when the production score-health thresholds reject it -- `ScoreHealthEvidence.passes` requires `0.90 <= coverage <= 1.0`, at least 10 unique scores and a score standard deviation above 1e-6. The median coverage column says which of the two it was, and the exact dates are in the JSON output.

Method: the frames are rebuilt from the signed selection, prediction and dataset
artifacts with the evaluator's own checks; the baseline branch is a replay of
`model_evaluator.evaluate_frozen_oos`'s baseline path plus `summarize_rank_ic_by_year`.

## Sensitivity: the same measurement with the score-health floor off

| Year | Candidate mean RankIC | Days | Baseline | Days | Delta |
|---|---|---|---|---|---|
| 2022 | 0.089977 | 242 | 0.081186 | 242 | +0.008791 |
| 2023 | 0.084595 | 242 | 0.078064 | 242 | +0.006530 |
| 2024 | 0.105206 | 242 | 0.090604 | 242 | +0.014602 |
| 2025 | 0.101197 | 243 | 0.098322 | 243 | +0.002876 |

Pooled with the floor off: candidate 0.095250 over 969 days, baseline 0.087056 over 969 days, delta +0.008194 -- against the published +0.007827 over 945 days.

This is a sensitivity check, not an alternative official number: the gates in the signed report are computed with the floor on, and nothing here rewrites them.
