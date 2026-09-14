# Real-Data M0R Candidate Walk-Forward Report

Constrained-search candidate evaluation over F1-F4 (test years 2022-2025) against the frozen best_simple baseline evidence chain.

## Overall OOS statistics

| Metric | Value |
|---|---|
| Valid day count | 945 |
| Mean RankIC | 0.0944 |
| ICIR | 0.7682 |
| HAC t-statistic | 12.905 |
| Baseline mean RankIC | 0.0852 |
| Delta RankIC | 0.0091 |

## Gate verdicts (G1-G6)

| Gate | Definition | Observed | Threshold | Status |
|---|---|---|---|---|
| G1 | OOS mean RankIC >= 0.030 | 0.0944 | 0.030 | PASS |
| G2 | HAC t-statistic >= 2.000 | 12.9054 | 2.000 | PASS |
| G3 | positive-year share >= 0.70 (2022-2025) | 1.0000 | 0.700 | PASS |
| G4 | delta RankIC vs best_simple baseline >= 0.005 | 0.0091 | 0.005 | PASS |
| G5 | max single-year positive contribution share <= 0.50 | 0.2791 | 0.500 | PASS |
| G6 | valid score-day ratio >= 0.95 | 0.9752 | 0.950 | PASS |

## Per-fold constrained search

| Fold | Model | Best config | Searched | Ens. val RankIC | Worst seed |
|---|---|---|---|---|---|
| F1 | m0r-f1 | feature_fraction=0.85, lambda_l2=0.0, learning_rate=0.05, max_depth=4, min_child_samples=200, num_leaves=15 | 1 | 0.0679 | 20260802 |
| F2 | m0r-f2 | feature_fraction=0.85, lambda_l2=0.0, learning_rate=0.05, max_depth=4, min_child_samples=200, num_leaves=15 | 1 | 0.0987 | 20260803 |
| F3 | m0r-f3 | feature_fraction=0.85, lambda_l2=0.0, learning_rate=0.05, max_depth=4, min_child_samples=200, num_leaves=15 | 1 | 0.0894 | 20260803 |
| F4 | m0r-f4 | feature_fraction=0.85, lambda_l2=0.0, learning_rate=0.05, max_depth=4, min_child_samples=200, num_leaves=15 | 1 | 0.1032 | 20260803 |

## Yearly OOS mean RankIC

| Year | Mean RankIC |
|---|---|
| 2022 | 0.0931 |
| 2023 | 0.0730 |
| 2024 | 0.1045 |
| 2025 | 0.1048 |

## Validation-stage comparison against B0-B3

Candidate ensemble validation RankIC versus each baseline family's validation mean RankIC (TEST-stage deltas are only defined against the selected best_simple baseline).

| Fold | Candidate | B0 | B1 | B2 | B3 | Selected |
|---|---|---|---|---|---|---|
| F1 | 0.0679 | 0.0544 | -0.0720 | 0.0627 | 0.0450 | B2 |
| F2 | 0.0987 | 0.0606 | -0.0830 | 0.0852 | 0.0850 | B2 |
| F3 | 0.0894 | 0.0630 | -0.0927 | 0.0987 | 0.0734 | B2 |
| F4 | 0.1032 | 0.0709 | -0.0785 | 0.0867 | 0.1019 | B3 |

## Data versions (baseline run snapshot)

| Table | Partitions |
|---|---|
| factor_daily | 144 partitions (2014-01..2025-12) |
| label_daily | 151 partitions (2014-01..2026-07) |
| universe_daily | 151 partitions (2014-01..2026-07) |

## Candidate verdict

- Full-mode run: yes
- All G1-G6 gates satisfied: yes
- F1-F4 all present: yes
- Yearly RankIC complete for 2022-2025: yes

**M0R candidate: WALK_FORWARD_PASSED**
