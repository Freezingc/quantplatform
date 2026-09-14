# Real-Data Candidate Comparison (plan Task 5.1)

Every number below is copied from a signed walk-forward report; nothing is recomputed and nothing is estimated. Pairwise increments are printed only inside the comparability group defined in the next section.

## Inputs

| Label | Family | Mode | Report sha256 | Path |
|---|---|---|---|---|
| M0R | M0R | full | `d16b5708c4cdc961` | `/data/zhangyudong/qrv2_research_artifacts/m0r_v2/report.json` |
| M1-full | M1 | full | `6da61acaa08555ae` | `/data/zhangyudong/qrv2_research_artifacts/m1_full_v1/report.json` |
| M1-carrier | M1 | full | `427bb58403bebf64` | `/data/zhangyudong/qrv2_research_artifacts/m1_parallel_v1/report.json` |
| M0-refined | M0 | full | `41a90d62fd710ef4` | `/data/zhangyudong/qrv2_research_artifacts/m0_v3/report.json` |
| M0-base | M0 | full | `cbf25c5d5ecc153a` | `/data/zhangyudong/qrv2_research_artifacts/m0_v2/report.json` |
| M0R-replicate | M0R | full | `d16b5708c4cdc961` | `/data/zhangyudong/qrv2_research_artifacts/m0r_v3/report.json` |

## Comparability

Reference group (6 of 6): M0R, M1-full, M1-carrier, M0-refined, M0-base, M0R-replicate. Shared basis: valid_day_count=945, baseline_mean_rank_ic=0.0852268662737014, baseline_selections=6fe33eda88430e5ee90e8d975c16f36c8d0de1a44a70eda594b12b82ff5678b9,3ea7d0c5c7a44d23ab43b3e1ade5079b50d771ac3d6313f4564e2d8e45678d74,3a9ac836c383cfd0e6778af132e6f5d3f84e1b9dd4966db2e59374f1c29ed565,6b26867126c28a5d99a62bc539d04f00a653039d564e2757e669f0e13c7c6219, datasets=226a8dfc497c,eb9bba1b5a0b,9766e69f3425,6f9478d7645c.

Same valid-day count, same baseline mean RankIC, same per-fold frozen baseline selections and same per-fold dataset ids is exactly the condition under which two G4 deltas share a baseline term and a day set, so their difference is a legal subtraction.

## Gate matrix (G1-G6)

| Label | G1 | G2 | G3 | G4 | G5 | G6 | Verdict |
|---|---|---|---|---|---|---|---|
| M0R | 0.094376 PASS | 12.905449 PASS | 1.000000 PASS | 0.009149 PASS | 0.279082 PASS | 0.975232 PASS | WALK_FORWARD_PASSED |
| M1-full | 0.093054 PASS | 12.472142 PASS | 1.000000 PASS | 0.007827 PASS | 0.283626 PASS | 0.975232 PASS | WALK_FORWARD_PASSED |
| M1-carrier | 0.091071 PASS | 11.851545 PASS | 1.000000 PASS | 0.005845 PASS | 0.289545 PASS | 0.975232 PASS | WALK_FORWARD_PASSED |
| M0-refined | 0.089429 PASS | 12.383336 PASS | 1.000000 PASS | 0.004202 FAIL | 0.281950 PASS | 0.975232 PASS | WALK_FORWARD_FAILED |
| M0-base | 0.088867 PASS | 12.297744 PASS | 1.000000 PASS | 0.003640 FAIL | 0.283864 PASS | 0.975232 PASS | WALK_FORWARD_FAILED |
| M0R-replicate | 0.094376 PASS | 12.905449 PASS | 1.000000 PASS | 0.009149 PASS | 0.279082 PASS | 0.975232 PASS | WALK_FORWARD_PASSED |

Gate definitions: G1 OOS mean RankIC >= 0.030; G2 HAC t-statistic >= 2.000; G3 positive-year share >= 0.70 (2022-2025); G4 delta RankIC vs fold-matched best_simple >= 0.005; G5 max single-year positive contribution share <= 0.50; G6 valid score-day ratio >= 0.95

## Standings

| Label | Grid (max configs/fold) | Validation-selection free | Mean RankIC | ICIR | HAC t | G4 delta vs baseline | Verdict |
|---|---|---|---|---|---|---|---|
| M0R | 1 | yes | 0.094376 | 0.768173 | 12.905 | +0.009149 | WALK_FORWARD_PASSED |
| M0R-replicate | 1 | yes | 0.094376 | 0.768173 | 12.905 | +0.009149 | WALK_FORWARD_PASSED |
| M1-full | 27 | no | 0.093054 | 0.749371 | 12.472 | +0.007827 | WALK_FORWARD_PASSED |
| M1-carrier | 1 | yes | 0.091071 | 0.712569 | 11.852 | +0.005845 | WALK_FORWARD_PASSED |
| M0-refined | 24 | no | 0.089429 | 0.748720 | 12.383 | +0.004202 | WALK_FORWARD_FAILED |
| M0-base | 24 | no | 0.088867 | 0.743053 | 12.298 | +0.003640 | WALK_FORWARD_FAILED |

A candidate whose grid is larger than one config chose its winner on the validation split, so its OOS delta carries validation-selection optimism; a one-config candidate carries none. Read a gap between the two as 'the wider grid is still behind by that much, with bias in its favour'.

## Pairwise increments (delta RankIC, row minus column)

| row \ col | M0R | M1-full | M1-carrier | M0-refined | M0-base | M0R-replicate |
|---|---|---|---|---|---|---|
| M0R | 0 | +0.001322 | +0.003304 | +0.004947 | +0.005509 | +0.000000 |
| M1-full | -0.001322 | 0 | +0.001982 | +0.003625 | +0.004187 | -0.001322 |
| M1-carrier | -0.003304 | -0.001982 | 0 | +0.001642 | +0.002205 | -0.003304 |
| M0-refined | -0.004947 | -0.003625 | -0.001642 | 0 | +0.000563 | -0.004947 |
| M0-base | -0.005509 | -0.004187 | -0.002205 | -0.000563 | 0 | -0.005509 |
| M0R-replicate | +0.000000 | +0.001322 | +0.003304 | +0.004947 | +0.005509 | 0 |

Because every candidate in the group shares one baseline term, `delta(a) - delta(b)` must equal `mean(a) - mean(b)`; each cell is printed only after that identity is checked to 1e-12. A cell reading `IDENTITY_VIOLATION` means the two reports disagree about their own arithmetic and neither should be quoted.

## Leading candidate

**M0R** (M0R, G4 +0.009149, grid 1 config(s) per fold, validation-selection free: yes).

### Yearly OOS distribution

| Year | Mean RankIC |
|---|---|
| 2022 | 0.093128 |
| 2023 | 0.072998 |
| 2024 | 0.104523 |
| 2025 | 0.104774 |

### Per-fold winner and worst seed (validation stage)

| Fold | Model | Winning config | Searched | Ens. val RankIC | Worst seed | Its val RankIC | All seeds |
|---|---|---|---|---|---|---|---|
| F1 | m0r-f1 | feature_fraction=0.85, lambda_l2=0.0, learning_rate=0.05, max_depth=4, min_child_samples=200, num_leaves=15 | 1 | 0.067948 | 20260802 | 0.067647 | 20260801:0.067894, 20260802:0.067647, 20260803:0.067887 |
| F2 | m0r-f2 | feature_fraction=0.85, lambda_l2=0.0, learning_rate=0.05, max_depth=4, min_child_samples=200, num_leaves=15 | 1 | 0.098708 | 20260803 | 0.098350 | 20260801:0.098795, 20260802:0.098368, 20260803:0.098350 |
| F3 | m0r-f3 | feature_fraction=0.85, lambda_l2=0.0, learning_rate=0.05, max_depth=4, min_child_samples=200, num_leaves=15 | 1 | 0.089440 | 20260803 | 0.089156 | 20260801:0.089369, 20260802:0.089241, 20260803:0.089156 |
| F4 | m0r-f4 | feature_fraction=0.85, lambda_l2=0.0, learning_rate=0.05, max_depth=4, min_child_samples=200, num_leaves=15 | 1 | 0.103177 | 20260803 | 0.103037 | 20260801:0.103074, 20260802:0.103063, 20260803:0.103037 |

Worst single seed across folds: **20260802 on F1** at 0.067647 validation RankIC. These are validation-stage numbers: the signed reports do not carry a per-seed OOS breakdown, so an OOS worst seed is **not derivable** from them and is not printed here.

## Baseline run (the term every G4 subtracts)

| Metric | Value |
|---|---|
| Valid day count | 969 |
| Mean RankIC | 0.087056 |
| ICIR | 0.750745 |
| HAC t-statistic | 12.750 |
| Mean RankIC 2022 | 0.081186 |
| Mean RankIC 2023 | 0.078064 |
| Mean RankIC 2024 | 0.090604 |
| Mean RankIC 2025 | 0.098322 |
| Fold-matched best_simple | F1=B2, F2=B2, F3=B2, F4=B3 |

**Do not subtract these yearly values from a candidate's yearly values.** The baseline run scores its own day set (it sees only the dense baseline-eligible factors), while each candidate's report carries the baseline *recomputed on the candidate's day set* as `baseline_mean_rank_ic`. The pooled figures differ for exactly that reason, and the signed reports contain no per-year baseline on the candidate day set, so per-year deltas are not derivable from them.

## Limits of this document

* RankIC only: no transaction cost, impact cost, capacity or turnover constraint is applied anywhere in these numbers.
* The six gates are a joint threshold, not a multiple-testing correction; a wider grid raises the chance of clearing G4 on validation-selected luck.
* A candidate whose grid exceeds one config is a per-fold ensemble of up to four different winning architectures, not one model specification.
* The locked 2026 segment is not part of any of these runs.
