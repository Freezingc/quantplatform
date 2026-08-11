# Compact reader smoke evidence

Date: 2026-08-03  
Database: `dfs://QuantResearchV2Test`

## Reader path

`CompactDatasetReader` was exercised against the real V2 compact tables for the
2025-01 partition. The reader resolved `latest_manifest`, selected the matching
`batch_id`, loaded `universe_daily_latest`, `label_daily_latest`, and filtered
`factor_observation_clean_latest` by requested `(factor_id, factor_version)`.

With a temporary in-memory frozen contract containing `yeslab_10` and an
explicitly relaxed smoke-only finite coverage threshold of `0.30`, the reader
produced 70,607 PIT rows with columns:

```text
signal_date, ts_code, yeslab_10, raw_return_5d, label_valid, exit_date
```

The production default threshold remains `0.70`. A strict run against the
available data rejected `yeslab_1` at a finite coverage of `0.388375`, as
intended.

## Factor availability finding

The current compact factor batch contains 48 factor IDs (`yeslab_1` through the
available IDs up to `yeslab_53`), while the frozen factor sets currently stored
in V2 reference factors such as `yeslab_65`, `yeslab_95`, `yeslab_96`,
`yeslab_98`, `yeslab_114`, `yeslab_155`, `yeslab_252`, `yeslab_271`,
`yeslab_357`, `yeslab_359`, `yeslab_363`, `yeslab_371`, `yeslab_372`,
`yeslab_376`, `yeslab_471`, `yeslab_482`, and `yeslab_484`.

The reader therefore fails closed for those existing frozen sets instead of
silently substituting a different factor list. The next data task is to
reconcile the frozen factor sets with a freshly materialized current factor
batch (or explicitly freeze a new set after factor-quality review); no training
result should be generated until that reconciliation passes.

## F03 factor coverage after source reconciliation

The F03 frozen set was materialized from YesLab for 2025-01 and its clean
source version was compact-published. The strict admission check still rejects
the set because five members do not meet the default 70% finite-coverage gate
on the 70,607 mature label rows:

| factor | finite coverage |
|---|---:|
| `yeslab_371` | 26.67% |
| `yeslab_411` | 32.06% |
| `yeslab_379` | 32.27% |
| `yeslab_406` | 32.46% |
| `yeslab_372` | 45.46% |

The remaining F03 factors are above 98.8% for this month. The low-coverage
members remain in the frozen contract and are not silently removed; a new
factor set must be created only after the normal training-window quality and
RankIC/ICIR review.

## Published source and compact evidence

The reconciled F03 clean source version is
`505efa794bb9a4ac35b586022baea292b6084534dccb89f7f6f4815016143cf8`. It is
the current pointer for `factor_observation_clean/2025-01`; the corresponding
compact batch is `1552505386348930877` with 1,497,237 rows. Compact
verification passed for all current objects (tradability, universe, label, and
factor observation), and the latest manifest points to this source version.

The initial deterministic attempt was interrupted after partial streaming
inserts and was subsequently rolled back from the current pointer. Its
quarantine audit batch remains available for inspection, but it is not used by
the clean reader. A later retry used a fresh nonce and published the complete
deduplicated source version above. Failed repair attempts produced only
unreferenced data versions; they are invisible to readers and were not deleted
because whole-table cleanup queries exceeded the DolphinDB memory budget. No
training or promotion is based on those versions.

## January RankIC diagnostic (not OOS evidence)

As a diagnostic only, the 2025-01 factor rows produced 342 daily factor/label
observations (18 valid trading days per factor). The highest raw January mean
RankICs were `yeslab_271` (0.0778), `yeslab_471` (0.0697), `yeslab_252`
(0.0642), `yeslab_484` (0.0271), and `yeslab_114` (0.0222). Several members
were near zero or negative. This one-month snapshot is not a substitute for
the required five-year training, one-year validation, and one-year locked test
windows; those runs remain blocked until monthly historical materialization
and the strict admission review are complete.

## Historical preflight stop condition

A read-only preflight for the same F03 membership on `2020-01` was run before
any target write. It stopped because `yeslab_464` had no clean observations in
that month. A direct source check found 95,280 January rows for this factor,
all null, so the clean-value contract had nothing publishable. The preflight
therefore did not create a source version, compact batch, or visibility
pointer. This is a data-availability blocker rather than a publisher failure:
the frozen 2020-era set cannot be assumed to be materializable for every month.
Historical work must first inventory
per-factor month coverage, apply the training-window RankIC/ICIR and finite
coverage gates, and then freeze a replacement set whose identities are
available across the entire requested walk-forward scope.

## Final verification

The earlier F03 checkpoint completed with 244 unit tests passing; Ruff reported
no violations and mypy reported no issues in 118 source files. The test database
compact verification at that checkpoint was green: 73 tradability pointers/manifests,
73 universe pointers/manifests, 72 label pointers/manifests, and one current
factor-observation pointer/manifest with 1,497,237 rows. The existing source
factor table still contains all five lineage columns, while the compact target
contains none of them.

## Continuation: frozen candidate set and historical backfill

The source audit was completed without changing any existing V2 table schema.
The selected set is frozen as factor set
`b8fd9f5f82ffe546cbc1738539ab46b6e055f44782de50008a535e83cdfd0e6c` under
`FINAL_train2020_2024_validation2025_dual_gate_v1`. Its six members and frozen
directions are:

| factor ID | feature name | direction |
|---|---|---:|
| `yeslab_377` | `GP_008` | -1 |
| `yeslab_380` | `GP_011` | -1 |
| `yeslab_387` | `GP_018` | -1 |
| `yeslab_388` | `GP_019` | -1 |
| `yeslab_391` | `GP_022` | -1 |
| `yeslab_433` | `GP_064` | -1 |

The dual gate used combined 2020–2024 training observations and 2025
validation observations. All six passed the training thresholds (absolute mean
RankIC at least 0.02, absolute ICIR at least 0.50, at least 60 valid days, and
directional share at least 0.55) and retained the same direction in validation.
`yeslab_382` was deliberately excluded because its validation absolute ICIR was
0.491595, just below the 0.50 gate. The resulting set and registry definition
hashes were written to the test V2 database only.

The factor-level gate evidence (raw RankIC sign is negative; the frozen
direction is therefore -1) was:

| factor | 2020–2024 train mean RankIC | train ICIR | 2025 validation mean RankIC | validation ICIR |
|---|---:|---:|---:|---:|
| `yeslab_377` | -0.081089 | -0.537393 | -0.082688 | -0.500939 |
| `yeslab_380` | -0.081672 | -0.592950 | -0.084273 | -0.545697 |
| `yeslab_387` | -0.081176 | -0.576865 | -0.083982 | -0.532922 |
| `yeslab_388` | -0.082160 | -0.573620 | -0.083028 | -0.500316 |
| `yeslab_391` | -0.079709 | -0.576067 | -0.082809 | -0.549882 |
| `yeslab_433` | -0.068070 | -0.535468 | -0.077396 | -0.614102 |

The 2020-01 through 2020-12 candidate source versions and compact manifests are
now all visible. They contain 5,528,660 compact rows in total; each month has a
single current source pointer and a `PUBLISHED` manifest. A strict
`CompactDatasetReader` plus `FactorAdmissionContract` build over the full 2020
window produced 666,421 PIT rows and 243 valid cross-sections. The directional
cross-sectional-rank ensemble baseline (daily ranks, frozen directions, equal
factor weight) achieved mean RankIC `0.053124`, RankIC standard deviation
`0.148293`, and ICIR `0.358236`. This is a diagnostic baseline, not a final
promotion result; the final-window comparison is recorded below.

The strict 2025 validation admission ratios were `GP_008=1.000000`,
`GP_011=1.000000`, `GP_018=0.999757`, `GP_019=0.997719`, `GP_022=0.985956`,
and `GP_064=0.983092`; no factor was close to the 0.70 rejection threshold.

After all 2020-01 through 2025-12 candidate manifests were published, the
purged FINAL validation comparison (2025, last six signal dates removed) was:

| model | valid days | mean RankIC | RankIC std | ICIR |
|---|---:|---:|---:|---:|
| directional rank equal-weight | 237 | 0.090791 | 0.157301 | 0.577179 |
| date-equal Ridge on directional ranks | 237 | 0.086343 | 0.150767 | 0.572690 |

The Ridge fit used only 2020–2024 purged training rows (3,893,789 valid labels
across 1,182 dates); its coefficients were frozen before reading 2025. The
simple rank ensemble is retained as the stronger validation baseline. These
are validation diagnostics because 2025 was part of the approved dual-gate
factor-set freeze; they are not the locked 2026 OOS result.

The preflight/materialization path was also tightened: duplicate groups are
aggregated with vectorized pandas operations, conflicts remain blocking
quarantine rows, while all-null groups are recorded as `NO_FINITE_VALUE` and
are allowed to continue to the explicit 70% finite-coverage admission gate.
The new rank ensemble implementation is covered by unit tests and does not
alter any database table.

The 2021-01 through 2025-12 candidate backfill completed as a resumable,
single-writer job. At that historical checkpoint the candidate had 72 current
source pointers and 72 current compact manifests covering 2020-01 through
2025-12, with 41,619,921 current compact factor rows. The locked 2026 test was
still intentionally inaccessible at that checkpoint; the subsequent authorized
continuation is recorded below and did not change the frozen factor selection.

The compact label manifest covered exactly 2020-01 through 2025-12 (72 monthly
partitions) before the authorized continuation added seven 2026 YTD partitions.
Consequently, the feasible complete fold in this database remains the approved
FINAL window (2020–2024 train, 2025 validation, 2026 locked test); the available
2026 data is only a partial-year test. Ordinary F01–F06 folds would require
labels for 2014–2019, which are not present; they are not silently approximated
with a shorter training window.

At the pre-continuation checkpoint an explicit database check found zero rows
dated 2026 or later in both `label_daily` and `label_daily_latest`; the
authorized continuation subsequently materialized the mature January–July
source window. The strict admission result and the still-unavailable full-year
RankIC/ICIR are recorded in the continuation section below.

An intermediate checkpoint after the factor-selection changes had 253 tests
passing, Ruff clean, and mypy clean across 120 source files.

The four model families now accept a one-to-sixty factor input range; this is
intentional because the frozen admission contract, rather than a historical
15-factor assumption in the model constructors, is the source of truth for
feature count. The historical post-backfill checkpoint had 257 tests passed,
with Ruff and mypy clean; the locked-access/preflight continuation now has 260
tests passed, with Ruff and mypy still clean.

During the long historical rebuild, a full-table orphan-row count exhausted
DolphinDB memory at 2023-07. The publisher now adds the declared logical-month
date predicate to that check; the regression test and the complete suite (257
tests) pass, and the resumed job completed all remaining months without another
memory error. The historical checkpoint reported 73/73/72/72 current
tradability/universe/label/factor pointers. After the authorized continuation,
full compact verification reports 79/79/79/79 current pointers and manifests,
with current row totals 7,802,520 / 7,802,520 / 5,481,974 / 45,624,276;
source lineage columns remain intact and compact tables contain no row-level
lineage columns.

## 2026 locked-test materialization and strict input-quality result

The user-authorized locked workflow was opened with capability
`user-approved-final-evaluation`, and all writes were isolated to
`dfs://QuantResearchV2Test`. No existing table schema was changed. The source
coverage available at this run was 2026-01-05 through 2026-07-31, so this is a
year-to-date test window rather than a complete 2026 calendar-year test.

Seven 2026 monthly source partitions are now published and compacted:

| object | partitions | rows |
|---|---:|---:|
| `market_daily_clean` | 7 | 759,251 |
| `tradability_daily` | 7 | 759,251 |
| `universe_daily` | 7 | 759,251 |
| `label_daily` | 7 | 542,136 |
| `factor_observation_clean` (six frozen factors) | 7 | 4,004,355 |

The 2026 label partitions contain 139 signal dates; 133 have at least one
valid five-day return (511,814 valid rows). The final six signal dates are
retained with explicit invalid/maturity status rather than being dropped from
the source table.

The security SCD snapshot was built as of 2026-07-31. Twenty-three codes present
in the market source but absent from the authoritative security snapshot were
handled through the existing market-observation-derived profile fallback. One
source name-history overlap for `600676.SH` was resolved by clipping the older
interval at the newer announced effective date. Both decisions are recorded in
the locked audit details and do not modify YesLab.

Before reading locked labels, the run wrote an `AUTHORIZE` event to
`locked_test_audit`. After the strict read was attempted, it wrote a `CONSUME`
event with outcome `REJECTED_INPUT_QUALITY`; the failed read is therefore not
treated as a successful model evaluation and cannot be used to retune factors.
The event IDs are `a7d63f7a-a305-48d4-ab05-7d56c71d7563` (authorize) and
`7039f59f-0d6a-4a50-9b3f-5ad53207ff44` (consumed rejection). The frozen
contract hash was `33df679240050b8e632c20a01fc4a6a6bc3ee5f96c7f7b7cb79acc62f915e1d9`;
the pre-registration and capability hashes were
`d544c9a8953ad0deb3c5cae02dfd76a91bfbbc9876e7540becab1d8a67eab816` and
`d44868a6a1b247f597bc6af14fbfcb5e07059fe6228abad873d89d6e91e60f9f`.

The frozen `FINAL` factor contract rejected the YTD frame before any score or
RankIC was calculated. The finite-coverage gate is 0.70:

| factor | finite ratio on 2026 YTD FINAL universe |
|---|---:|
| `GP_008` | 1.000000 |
| `GP_011` | 1.000000 |
| `GP_018` | 0.994168 |
| `GP_019` | 0.947322 |
| `GP_022` | 0.660249 **FAIL** |
| `GP_064` | 0.658257 **FAIL** |

The coverage break is concentrated in the upstream `factor_data9` source:
`GP_022` and `GP_064` are near expected coverage through April, then have only
roughly one quarter of the eligible universe populated from May onward. The
strict admission boundary therefore correctly refuses to produce an official
2026 YTD RankIC/ICIR. No fallback factor set, threshold relaxation, or
post-lock model selection was performed.

The root-cause check distinguished a source gap from a V2 materialization bug:
the May 29 `factor_data9` partitions still contain 5,542 GP_064 keys, but only
1,442 values are non-null. The non-null keys are concentrated in a subset of
exchange/code prefixes (for example, 300/002/301/601 have zero finite values
on that date), which is consistent with a partial upstream factor-computation
job rather than a deterministic universe filter. As an independent check, a
write-free reconstruction of `GP_022` from the raw market fields using
`ts_max_60(pct_chg) * amount` matched every one of the 508,684 finite source
rows exactly; the reconstruction produced finite candidates for 753,975 of
the 759,251 deduplicated 2026 raw-market rows. This validates the diagnosis but
is not used to replace
the frozen source values, because GP_064's original calculation engine is not
available in the current workspace.

The earlier January–March duplicate groups were resolved by the existing
finite-value-aware deduplicator, retaining a finite representative when a
duplicate group contained one. By April–July the remaining problem is not
duplicate resolution: the affected source keys are present once and their
factor values are `NaN`.

For troubleshooting only (not a promotion or OOS result), the exact frozen
contract was evaluated on the quality-qualified January–April subset, where all
six factors pass the 0.70 finite gate. That diagnostic produced 302,810 rows
over 77 valid days, mean RankIC `0.040184`, RankIC standard deviation `0.151032`,
ICIR `0.266067`, HAC t-statistic `1.292758`, and positive-day share `0.584416`.
It is intentionally not persisted as the official `FINAL/TEST` evaluation:
the requested one-year test remains incomplete until upstream factor coverage
is repaired and a fresh, pre-registered locked run is authorized.

The production compact reader now performs the same frozen-factor finite-
coverage preflight before querying `label_daily_latest`. A future input-quality
failure will therefore stop before consuming the locked-label capability; this
ordering is covered by a regression test.

## 2026 factor-source repair and compact correction

The previous locked-test rejection above was based on the unrepaired 2026
factor snapshot. It is retained as an audit record and is superseded for data
quality purposes by the repair below; it is not silently rewritten or reused as
an OOS result.

The source calculator implementations for all six frozen factors are now
present in `数据库/factor_calculators`. In particular, the source hashes for
`GP_022` and `GP_064` are:

```text
GP_022: e5d63b603d748f03c48cb65b5d2451f06b7c936e6749eebc5b71bee4a194cecf
GP_064: 63296bc0fa092735e984da30ef2047010d1662863d0001348df7f431b9b08dce
```

The raw YesLab `stock_factor_daily` check for 2026-05-08 found only 3,164
records (1,582 unique codes after exact-key deduplication), while the Tushare
`daily` response contained 5,491 records. The 3,909 missing `(trade_date,
ts_code)` keys were added only to the V2-derived market input; the existing
YesLab table was not modified. The Tushare response hash is
`b9d615bf1486503c0cf6716667b9fad61088428b35eaa83d3c909bb7a2ef98e8`, the
added-key batch hash is
`0f3772405a1241c80e0da22cb83e00a292c0b1f70f4bd60d936de3282f61b85d`, and the
combined repair hash is
`184ea1d737167d44f8db543e20ac4fb5e0fdd9453748cf98e4a2e32c26482343`.

Candidate reconstruction was performed only for non-finite source values and
never overwrote a finite source value. For `GP_022`, 224,727 source keys became
finite candidates; for `GP_064`, 223,385 became finite candidates. The latter
had 10 finite source/candidate mismatches; the source values were retained.
No guessed formula or untracked replacement was used.

The repaired six-factor batches were published under new immutable source
versions for 2026-01 through 2026-07 and then compacted. The current compact
factor row counts are:

| partition | rows | source data-version prefix |
|---|---:|---|
| `2026-01` | 644,986 | `28fda6d4f2b1` |
| `2026-02` | 454,418 | `23af22766586` |
| `2026-03` | 715,751 | `ceb7a775df67` |
| `2026-04` | 684,063 | `f765c397e438` |
| `2026-05` | 527,373 | `ad6a853362cf` |
| `2026-06` | 679,609 | `44ae05113a45` |
| `2026-07` | 746,267 | `84f434893a78` |

That is 4,452,467 current 2026 factor rows and 46,072,388 current factor
rows across 2020-01 through 2026-07. Factor-only and full compact verification
both pass: 79/79 pointers and manifests for every object, no row-level lineage
columns in compact targets, and all five lineage columns intact in the source
tables.

A write-free factor-admission preflight on the repaired 2026-01 through
2026-07 universe (542,136 eligible rows) reports:

| factor | finite ratio |
|---|---:|
| `GP_008` | 1.000000 |
| `GP_011` | 1.000000 |
| `GP_018` | 0.994197 |
| `GP_019` | 0.947569 |
| `GP_022` | 0.978011 |
| `GP_064` | 0.974381 |

All six pass the 0.70 gate in this partial-year preflight. This check did not
read or consume locked labels. Since the available 2026 market source ends on
2026-07-31, a complete calendar-year OOS RankIC/ICIR still cannot be reported;
a fresh explicit locked-test authorization is required before any official
test-label read.

## Verification environment

Project verification was rerun in the requested `quantyd` Conda environment
(Python 3.11.13; PyTorch 2.6.0, scikit-learn 1.7.2, LightGBM 4.6.0). The full
suite passed with 271 tests; Ruff reported no violations and mypy reported no
issues in 122 source files.

## Missingness versus predictive-gate evidence

The final six-factor freeze was not caused solely by missing values. On the
2025 validation universe, the six frozen factors had finite ratios of
`1.000000`, `1.000000`, `0.999757`, `0.997719`, `0.985956`, and `0.983092`.
The known near-miss factor `yeslab_382` was excluded for a predictive reason:
its validation absolute ICIR was `0.491595`, below the `0.50` gate. The complete
reasons for every factor not in the final set are not reconstructed from the
current registry alone; they require the original candidate summary, so the
report does not label all non-members as missing-value failures.

The earlier 2026 rejection did contain a genuine missingness failure. Before
repair, `GP_022` and `GP_064` had YTD finite ratios of `0.660249` and
`0.658257`, respectively. After adding only the 3,909 missing 2026-05-08
market keys and recomputing the two formulas from the supplied calculator
source, the write-free 2026-01 through 2026-07 preflight ratios became
`0.978011` and `0.974381`. This is evidence that those two failures were a
source-data gap, not an overly strict economic threshold.

Reconstruction is permitted only when the exact factor implementation and all
required point-in-time inputs are available. The repair therefore retains every
finite YesLab value, fills only non-finite source values, checks finite
source/candidate overlaps, quarantines conflicts, and publishes a new immutable
`data_version`. A Tushare row by itself is not sufficient for every factor:
adjustment conventions, lookback history, suspended/IPO/delist status, and
non-price inputs must match the original calculator. Structural missingness
from a required lookback or unavailable security history is not imputed.

## Six-factor tabular model diagnostic

Using the repaired six-factor contract, train-only robust preprocessing, the
2020-01 through 2024-12 TRAIN window, and the 2025 VALIDATION window (last six
signal dates purged in each role), the first tabular comparison produced:

| model | valid validation days | mean RankIC | RankIC std | ICIR |
|---|---:|---:|---:|---:|
| Ridge (alpha=1) | 237 | 0.084718 | 0.165185 | 0.512865 |
| LightGBM LambdaRank (100 rounds) | 237 | 0.094559 | 0.160515 | 0.589098 |

The LightGBM implementation initially used full within-date ranks as labels,
which exceeded its default LambdaRank relevance mapping for large A-share
cross-sections. It now uses deterministic five-quantile relevance labels
`0..4`, with a regression test covering cross-sections larger than 31 stocks.
These are validation diagnostics only; no 2026 label or locked-test capability
was read, and no model has been promoted.

## 2026-08-04 training continuation

The static PIT-to-training boundary is now implemented by
`research_v2/src/research_v2/training/static_batches.py`. It turns an already
assembled frame into one keyed `TrainingBatch` per signal date, pads each
cross-section to a common width, preserves an explicit factor-missing mask,
excludes invalid/non-finite labels, and rejects 2026 or later signal dates
before constructing tensors. Preprocessing remains a separate train-only step;
the helper does not fit medians, winsor limits, or scales.

The FINAL factor set has six members, while the approved D3.5 rule requires at
least 15 admitted factors before complex models are eligible. Therefore no
Residual MLP, Set Attention, GRU, or TCN checkpoint was created in this
continuation. The existing 1–60 model constructor range is a tensor-shape
contract only and does not override the factor-count gate. Ridge and
LightGBM remain validation diagnostics, with LightGBM the stronger result
shown above; they are not locked-test or production evidence.

In the requested `quantyd` environment (Python 3.11.13, PyTorch 2.6.0,
scikit-learn 1.7.2, LightGBM 4.6.0), the research suite now passes 271 tests;
Ruff and mypy are clean across 122 source files. Verification uses the direct
environment interpreter `/data/zhangyudong/.conda/envs/quantyd/bin/python` to
avoid shell-wrapper resolution differences observed in the current session.
