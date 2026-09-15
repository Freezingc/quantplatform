# Validation row-set asymmetry: candidate vs frozen baseline

## Reproduction

- F1: recorded `0.059238829903105104`, recomputed with the repository's own `validation_rank_ic` `0.059238829903105104`, bit-identical = **True**.

## Who scores how much of the panel

| carrier | fold | panel rows | scored rows | deficit | deficit share | whole panel? |
|---|---|---|---|---|---|---|
| M0R | F1 | 696906 | 696906 | 0 | 0.0000% | yes |
| M0R | F2 | 783477 | 783477 | 0 | 0.0000% | yes |
| M0R | F3 | 859748 | 859748 | 0 | 0.0000% | yes |
| M0R | F4 | 908649 | 908649 | 0 | 0.0000% | yes |
| M0R-replicate | F1 | 696906 | 696906 | 0 | 0.0000% | yes |
| M0R-replicate | F2 | 783477 | 783477 | 0 | 0.0000% | yes |
| M0R-replicate | F3 | 859748 | 859748 | 0 | 0.0000% | yes |
| M0R-replicate | F4 | 908649 | 908649 | 0 | 0.0000% | yes |
| M1-full | F1 | 696906 | 695819 | 1087 | 0.1560% | **no** |
| M1-full | F2 | 783477 | 782500 | 977 | 0.1247% | **no** |
| M1-full | F3 | 859748 | 859188 | 560 | 0.0651% | **no** |
| M1-full | F4 | 908649 | 907933 | 716 | 0.0788% | **no** |
| M1-carrier | F1 | 696906 | 695819 | 1087 | 0.1560% | **no** |
| M1-carrier | F2 | 783477 | 782500 | 977 | 0.1247% | **no** |
| M1-carrier | F3 | 859748 | 859188 | 560 | 0.0651% | **no** |
| M1-carrier | F4 | 908649 | 907933 | 716 | 0.0788% | **no** |
| M0-refined | F1 | 696906 | 696906 | 0 | 0.0000% | yes |
| M0-refined | F2 | 783477 | 783477 | 0 | 0.0000% | yes |
| M0-refined | F3 | 859748 | 859748 | 0 | 0.0000% | yes |
| M0-refined | F4 | 908649 | 908649 | 0 | 0.0000% | yes |
| M0-base | F1 | 696906 | 696906 | 0 | 0.0000% | yes |
| M0-base | F2 | 783477 | 783477 | 0 | 0.0000% | yes |
| M0-base | F3 | 859748 | 859748 | 0 | 0.0000% | yes |
| M0-base | F4 | 908649 | 908649 | 0 | 0.0000% | yes |
| M2 | F1 | 696906 | 691097 | 5809 | 0.8335% | **no** |

## Which rows M2 does not score (F1)

5809 of 696906 panel rows (**0.8335%**), touching 237 of 237 days.
Per-day dropped: min 7, median 24.0, max 52; per-day share min 0.2450%, median 0.8230%, max 1.7196%; days with zero dropped: 0.
Concentration in the window's first 20 days: {'rows_dropped_there': 440, 'share_of_all_dropped': 0.07574453434326046, 'share_of_panel_rows_there': 0.08087891336851742}.
Distinct securities affected: 615 of 3884; affected on every one of their days: 38.
label_valid among the dropped rows: {'true': 4722, 'false_or_nan': 1087}.

Feature NaN rate, dropped vs kept (first five factors):

- f00: dropped 0.0363 vs kept 0.0228
- f01: dropped 0.2944 vs kept 0.0315
- f02: dropped 0.5073 vs kept 0.5143
- f03: dropped 0.3987 vs kept 0.5228
- f04: dropped 0.0069 vs kept 0.0015

## How much can a row-set difference of that size move the metric?

- **m2_scale** (5809 rows, 0.8405% of scored rows, 50 random draws): mean 0.059264, sd **0.000114**, range 0.058989–0.059456; shift from the unsubsampled value +0.000025. The M2-minus-baseline gap is -0.003472 = **-30.38 sd** of this scale.
- **m1_scale** (1087 rows, 0.1573% of scored rows, 50 random draws): mean 0.059240, sd **0.000058**, range 0.059121–0.059367; shift from the unsubsampled value +0.000001. The M2-minus-baseline gap is -0.003472 = **-60.11 sd** of this scale.

_Caveat: a random row loss of M2's size is the null hypothesis 'the dropped rows carry no particular information'; the real dropped rows are not random (see dropped_rows), so this bounds the scale of the effect, not its sign or its size._

## Does the OOS acceptance path share this?

no. model_evaluator._match_prediction_coverage inner-joins the candidate's and the baseline's business keys and scores both on that intersection, so the row-set asymmetry measured here exists on the validation/selection path only

G1-G6 read the OOS evaluation, so they are not exposed to this asymmetry. What the asymmetry does affect is which config each fold's search selects, and any comparison drawn from validation_rank_ic -- including the standing table in §AS.2

comparison_interim.md's comparability basis checks valid_day_count, baseline mean RankIC, per-fold baseline selections and per-fold dataset ids. It does not check the prediction row set. On the OOS side _match_prediction_coverage makes that unnecessary; on the validation side nothing does

## Not done here

No baseline validation predictions exist on disk (only baseline_oos/F*.parquet, which is the 2022-2025 test side), so B2's validation RankIC could not be recomputed on M2's row set -- the sensitivity block bounds the scale of such a difference instead of measuring it. Nothing was written into any artifact directory; no database was touched; no weight was loaded; no process was signalled.
