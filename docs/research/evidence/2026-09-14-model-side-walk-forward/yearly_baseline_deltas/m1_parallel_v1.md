# Per-year baseline RankIC on the candidate's day set: M1

Source report `/data/zhangyudong/qrv2_research_artifacts/m1_parallel_v1/report.json` (sha256 `427bb58403bebf64`).
Both acceptance checks passed bit-for-bit, so the per-year baseline column is a
partition of the pooled baseline mean the signed report already publishes.

| Year | Candidate mean RankIC | Days | Baseline on the same days | Delta | Baseline before coverage matching | Days | Days lost to matching | Baseline mean there |
|---|---|---|---|---|---|---|---|---|
| 2022 | 0.086588 | 241 | 0.080637 | +0.005951 | 0.080637 | 241 | 0 | - |
| 2023 | 0.069627 | 219 | 0.069806 | -0.000179 | 0.069806 | 219 | 0 | - |
| 2024 | 0.104870 | 242 | 0.090604 | +0.014265 | 0.090604 | 242 | 0 | - |
| 2025 | 0.101103 | 243 | 0.098322 | +0.002782 | 0.098322 | 243 | 0 | - |

Pooled: candidate 0.091071 over 945 days, baseline 0.085227 over 945 days, delta +0.005845. Before coverage matching the same baseline artifact averages 0.085227 over 945 days.

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

Both frames in that table are the MATCHED pair the evaluator actually scores: `evaluate_frozen_oos` calls `_match_prediction_coverage` first, so the coverage shown is the intersection's, not either side's own. The next table separates the two sides, which is what identifies which one drops the day.

### Which side of the pair decides the day set

| Year | Days | Median candidate coverage | Median baseline coverage | Median pair coverage | Min pair | Pair below the 0.90 floor | Of those, days the candidate side alone would have passed | Passing by <0.005 |
|---|---|---|---|---|---|---|---|---|
| 2022 | 242 | 1.0000 | 0.9254 | 0.9254 | 0.8951 | 1 | 1 | 0 |
| 2023 | 242 | 1.0000 | 0.9093 | 0.9093 | 0.8903 | 23 | 23 | 42 |
| 2024 | 242 | 1.0000 | 0.9924 | 0.9924 | 0.9711 | 0 | 0 | 0 |
| 2025 | 243 | 1.0000 | 0.9859 | 0.9859 | 0.9809 | 0 | 0 | 0 |

Of the 24 OOS days the matched pair could not pass the floor, **24** were days the candidate side alone would have passed -- i.e. days dropped by the baseline artifact's row coverage rather than by anything the candidate did.

Method: the frames are rebuilt from the signed selection, prediction and dataset
artifacts with the evaluator's own checks; the baseline branch is a replay of
`model_evaluator.evaluate_frozen_oos`'s baseline path plus `summarize_rank_ic_by_year`.
