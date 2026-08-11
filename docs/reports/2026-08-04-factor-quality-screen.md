# Factor quality screen

Date: 2026-08-04  
Data scope: 2020-01–2024-12 TRAIN and 2025 VALIDATION  
Locked data: not read  
Database writes: append-only quality snapshots in `dfs://QuantResearchV2Test`

At this checkpoint the existing snapshot table contains 2,586 append-only rows
across 19 scopes. No existing source, label, universe, or factor table schema
was modified.

The YesLab metadata inventory contains 487 factor definitions. This report is
the strict cross-window screen of the 26 GP candidates for which a consistent
PIT evaluation artifact already exists; it is not a claim that all remaining
461 definitions have failed. The untested definitions remain `UNSCREENED`
until they pass the same staged process.

The metadata flag `is_verified` is `False` for all 487 definitions, so it is
not being used as a quality signal; quality is determined from reproducible
point-in-time observations and labels below.

## Screening protocol

The screen is deliberately separated from model training. A candidate must be
useful in both the training and validation windows before it can enter the
research shortlist:

- absolute mean daily RankIC at least `0.02`;
- absolute ICIR at least `0.50`;
- at least 60 valid RankIC days;
- directional positive-day share at least `0.55` in both windows;
- the frozen direction must remain unchanged from TRAIN to VALIDATION;
- overall finite coverage and month-level clean coverage must be checked
  separately from predictive statistics.

The source audit produced 26 strict cross-window candidates. Twenty-four
passed the predictive validation gate. The two explicit predictive failures
were `yeslab_418/GP_049` (2025 validation ICIR `0.407420`) and
`yeslab_444/GP_075` (2025 validation ICIR `-0.448749`).

An exploratory similarity file was then used to inspect redundancy. That file
was built on the 2025 VALIDATION window, so it is not valid for formal
representative selection. The five removals below are therefore recorded only
as a validation-based diagnostic, not as a final train-only decision:

| removed factor | retained representative | reason |
|---|---|---|
| `yeslab_393/GP_024` | `yeslab_376/GP_007` | high-similarity cluster |
| `yeslab_383/GP_014` | `yeslab_381/GP_012` | high-similarity cluster |
| `yeslab_385/GP_016` | `yeslab_405/GP_036` | high-similarity cluster |
| `yeslab_431/GP_062` | `yeslab_432/GP_063` | high-similarity cluster |
| `yeslab_458/GP_089` | `yeslab_432/GP_063` | high-similarity cluster |

The resulting provisional 19-factor shortlist was checked for clean monthly availability.
Three factors had at least one month with no clean observation and were not
admitted to the expanded shortlist:

| factor | source factor | overall finite ratio | minimum monthly finite ratio |
|---|---|---:|---:|
| `yeslab_398` | `GP_029` | 0.959455 | 0.000000 |
| `yeslab_404` | `GP_035` | 0.895337 | 0.000000 |
| `yeslab_464` | `GP_095` | 0.977296 | 0.000000 |

This leaves 16 provisional research candidates. The zero-month issue is a
data-availability failure, not evidence that the factor has no predictive
signal. The 16-factor list must be re-derived after train-only similarity
clustering and the complete structural/PIT/FDR audit.

## 16 research candidates

The table reports raw RankIC. A negative raw value is valid when the frozen
direction is `-1`; the admission test uses signed RankIC.

| factor | source | dir | TRAIN mean IC | TRAIN ICIR | VALID mean IC | VALID ICIR | finite | min monthly |
|---|---|:---:|---:|---:|---:|---:|---:|---:|
| `yeslab_376` | GP_007 | -1 | -0.082297 | -0.704694 | -0.094218 | -0.772261 | 0.996784 | 0.991952 |
| `yeslab_377` | GP_008 | -1 | -0.079987 | -0.532756 | -0.086075 | -0.520858 | 0.999649 | 0.977013 |
| `yeslab_380` | GP_011 | -1 | -0.080690 | -0.588312 | -0.088117 | -0.571801 | 0.999649 | 0.977013 |
| `yeslab_381` | GP_012 | -1 | -0.074693 | -0.607553 | -0.096269 | -0.906742 | 0.998126 | 0.995869 |
| `yeslab_387` | GP_018 | -1 | -0.080153 | -0.572264 | -0.087810 | -0.557987 | 0.999408 | 0.976433 |
| `yeslab_388` | GP_019 | -1 | -0.081130 | -0.569067 | -0.086880 | -0.523704 | 0.994148 | 0.788125 |
| `yeslab_391` | GP_022 | -1 | -0.078762 | -0.570828 | -0.087495 | -0.586322 | 0.983567 | 0.965358 |
| `yeslab_395` | GP_026 | -1 | -0.074639 | -0.543989 | -0.088258 | -0.680962 | 0.984076 | 0.952766 |
| `yeslab_396` | GP_027 | -1 | -0.073869 | -0.585969 | -0.083888 | -0.617998 | 0.998126 | 0.995869 |
| `yeslab_400` | GP_031 | -1 | -0.076552 | -0.543120 | -0.085009 | -0.675405 | 0.984076 | 0.952766 |
| `yeslab_405` | GP_036 | -1 | -0.066503 | -0.614139 | -0.087727 | -0.994570 | 0.998126 | 0.995869 |
| `yeslab_425` | GP_056 | -1 | -0.071064 | -0.556954 | -0.097025 | -0.848573 | 0.996784 | 0.991952 |
| `yeslab_432` | GP_063 | -1 | -0.071031 | -0.587406 | -0.088740 | -0.816972 | 0.998989 | 0.996916 |
| `yeslab_433` | GP_064 | -1 | -0.067607 | -0.531452 | -0.080874 | -0.644538 | 0.980265 | 0.960852 |
| `yeslab_442` | GP_073 | +1 | 0.071885 | 0.588799 | 0.080531 | 0.623635 | 0.995811 | 0.991758 |
| `yeslab_443` | GP_074 | -1 | -0.066206 | -0.582075 | -0.076120 | -0.672329 | 0.990208 | 0.977432 |

## Decision

The 16 factors above are marked `PROVISIONAL_RESEARCH_CANDIDATE`, not `FROZEN`.
The six
currently registered FINAL factors are a subset:

`yeslab_377`, `yeslab_380`, `yeslab_387`, `yeslab_388`, `yeslab_391`,
`yeslab_433`.

Before changing the FINAL contract, the candidates still need train-only
similarity clustering, formula-definition hashes, duplicate and conflict
quarantine, availability dates, source-version identity, FDR/annual stability,
and reproducible compact reads. This report does not authorize model training
or 2026 locked-test consumption.

## Quality snapshot persistence

The existing V2 `factor_quality_snapshot` table now contains one append-only
row for each of the 487 metadata definitions under scope
`SCREEN_TRAIN2020_2024_VALIDATION2025_STRICT26_V1`. The rows record the source
metadata, evidence metrics where available, decision status, reason codes,
rules version, data version, and record hash. The current status counts are:

| decision status | rows |
|---|---:|
| `FROZEN` | 6 |
| `PREDICTIVE_STABLE_PENDING_DQ` | 18 |
| `PREDICTIVE_FAIL` | 2 |
| `UNSCREENED` | 461 |

The 461 `UNSCREENED` rows are intentional: they prevent an absent evaluation
from being mistaken for a quality failure or a pass. The next screen will use
the existing annual RankIC summaries as a provisional prefilter, then re-run
the structural/PIT and train-only similarity checks before any factor-set
change.

A second snapshot scope, `SCREEN_TRAIN2020_2024_ANNUAL_SUMMARY_PREFILTER_V1`,
was also recorded from the existing five-year annual RankIC summaries. It
contains 14 provisional annual-stable factors, 459 annual predictive failures,
and 14 definitions without all five annual summary scopes. The 14 provisional
IDs are:

```text
yeslab_98, yeslab_155, yeslab_357, yeslab_359, yeslab_372, yeslab_376,
yeslab_379, yeslab_393, yeslab_406, yeslab_411, yeslab_442, yeslab_464,
yeslab_471, yeslab_484
```

This second scope is only a prefilter because the historical summary does not
include the 2025 validation year and does not by itself prove formula-level
PIT/DQ, FDR, or train-only similarity compliance.

The full daily TRAIN screen was then recomputed from 571,678 daily RankIC rows
covering 476 factor IDs. It applies 5-lag HAC p-values with Benjamini–Hochberg
q-values across the complete factor family, plus minimum valid days, signed
mean RankIC, signed ICIR, directional-day share, and positive-year count. The
corrected scope is `SCREEN_TRAIN2020_2024_DAILY_HAC_FDR_V2`:

| decision status | rows |
|---|---:|
| `TRAIN_PREDICTIVE_PASS_PENDING_DQ` | 70 |
| `TRAIN_PREDICTIVE_FAIL` | 404 |
| `EXCLUDED_SYNTHETIC` | 2 |
| `UNSCREENED` | 11 |

The two excluded IDs are `yeslab_0` and `yeslab_124`, which are synthetic
factor IDs already excluded by the admission contract. The earlier V1 daily
snapshot is retained as an append-only diagnostic, but V2 is the corrected
scope. The 70 passing factors are still only predictive candidates: structural
coverage, PIT availability, duplicate/conflict quarantine, and train-only
similarity selection remain required before formal admission.

## Structural coverage checkpoint

The 70 daily predictive passes were compared with the same-date synthetic
reference cross-section (`yeslab_0`) over 2020-01–2024-12. The gate requires at
least 80% of reference dates, median sample coverage of at least 80%, 5th
percentile coverage of at least 70%, no month below 70%, and no duplicate daily
factor/date key. The append-only scope is
`SCREEN_TRAIN2020_2024_STRUCTURAL_COVERAGE_V2`:

| decision status | rows |
|---|---:|
| `STRUCTURAL_DQ_PASS_PENDING_RAW_PIT` | 47 |
| `STRUCTURAL_DQ_FAIL` | 23 |

The 23 failures are not predictive failures. They are blocked by one or more
of: missing months/low monthly day coverage, low median sample coverage, low
5th-percentile sample coverage, or an insufficient five-year day ratio. The
47 passes have complete daily coverage in this proxy, but the proxy does not
replace source-level PIT, revision resolution, or formula validation.

## Raw-source sample audit

For the 47 structural passes, a bounded read-only audit of YesLab raw source
rows for 2024-01 was recorded under
`SCREEN_TRAIN2024_01_RAW_SOURCE_AUDIT_V1`. All 47 had source rows and a finite
value ratio above the 70% gate; the mean finite ratio was 0.8967 (minimum
0.8308). The median duplicate-row ratio was 0.50 because the source contains
revision rows that the existing deterministic deduplication step must resolve.
This is why every row remains
`RAW_SAMPLE_AUDIT_PASS_PENDING_FULL_RANGE_PIT`, rather than being admitted.
The scope is an audit checkpoint only: it does not assert that the full
2020–2024 source range has passed PIT/deduplication, nor does it read 2025 or
2026 labels.

The initial sample scope was intentionally conservative: it did not assert
full-range PIT/deduplication. The full-range source and PIT-key checks below
supersede that pending status while retaining this sample scope as an
append-only checkpoint.

## Full-range source and PIT key checks

The chunked server-side source audit was completed for all 47 source-qualified
candidates. It covers 2020-01-02–2024-12-31 and transfers only daily aggregates
(`factor_name × trade_date`), plus distinct security/date counts; the raw source
rows were not loaded as one client-side panel. Scope:
`SCREEN_TRAIN2020_2024_RAW_SOURCE_AGG_V1`.

| metric | result |
|---|---:|
| factors with full date coverage | 47 / 47 |
| minimum finite-value ratio | 0.737690 |
| mean finite-value ratio | 0.792646 |
| factors with missing months | 0 |
| duplicate-row ratio | 0 or 0.50 (factor-dependent) |

The distinct-value conflict audit is recorded under
`SCREEN_TRAIN2020_2024_RAW_CONFLICT_AGG_V1`: all 47 factors have zero keys with
multiple distinct values. Identical revision rows remain expected input to the
deterministic deduplicator; a zero conflict count does not mean the rows were
silently accepted without deduplication.

After filtering to finite, deduplicated security/date keys, the PIT coverage
scope `SCREEN_TRAIN2020_2024_PIT_KEY_COVERAGE_V1` shows 47/47 passes. All have
1212 valid source days, a minimum 5th-percentile coverage ratio of 1.253870
against the RankIC reference cross-section, and no missing month. This is a
source-key PIT proxy for technical same-day-close factors; compact universe and
label intersection checks remain separate.

The consolidated source gate
`SCREEN_TRAIN2020_2024_SOURCE_DQ_COMBINED_V1` therefore contains 47
`SOURCE_DQ_PASS_PENDING_SIMILARITY` rows. These rows are not yet formal model
inputs: train-only similarity and the compact-universe reconciliation are still
required.

## TRAIN-only similarity diagnostic

To avoid the earlier validation-window leakage, a corrected diagnostic uses only
20 deterministic TRAIN dates (four dates per year from 2020–2024). The first
implementation scope `SCREEN_TRAIN2020_2024_SIMILARITY_SAMPLE_V1` is retained
as an invalid append-only diagnostic because its Boolean common-member matrix
count made every pair appear insufficient. The corrected scope is
`SCREEN_TRAIN2020_2024_SIMILARITY_SAMPLE_V2`:

| decision status | rows |
|---|---:|
| `SIMILARITY_SAMPLE_REDUNDANT_PENDING_FULL` | 26 |
| `SIMILARITY_SAMPLE_UNIQUE_PENDING_FULL` | 21 |

The highest sampled absolute Spearman similarity is approximately 0.999995;
the median of each factor's nearest sampled similarity is approximately 0.9124.
These are evidence-scoped diagnostics only. They do not select representatives
or alter the six-factor frozen contract. A full TRAIN-range similarity pass (or
a documented deterministic approximation with sufficient dates) must be run
before marking any factor `REDUNDANT` or changing the factor set.

The stronger streaming monthly diagnostic
`SCREEN_TRAIN2020_2024_SIMILARITY_MONTHLY_V1` uses the first available TRAIN
date of each month (60 dates), releases each daily panel before moving to the
next date, and produces the same `26` redundant / `21` unique split. The
nearest-pair median is approximately `0.90797` across factors, with all pairs
having 60 valid monthly observations. The agreement between the 20-date and
60-date diagnostics is useful stability evidence, but it remains a sample of
the 1212-day window rather than a formal full-day cross-sectional gate.

The 21 factors below are the monthly-sample `UNIQUE` diagnostics; they are a
research shortlist only and are not published as a new active factor version:

```text
yeslab_155, yeslab_163, yeslab_363, yeslab_385, yeslab_395, yeslab_396,
yeslab_401, yeslab_405, yeslab_41, yeslab_425, yeslab_437, yeslab_439,
yeslab_442, yeslab_443, yeslab_471, yeslab_473, yeslab_484, yeslab_94,
yeslab_95, yeslab_96, yeslab_98
```

As a complementary full-window dependence check, the scope
`SCREEN_TRAIN2020_2024_RANKIC_SERIES_SIM_V1` computes absolute Pearson
correlation of each factor's daily RankIC series across all 1212 TRAIN dates:

| decision status | rows |
|---|---:|
| `RANKIC_SERIES_SIM_HIGH_PENDING_CROSS_SECTION` | 28 |
| `RANKIC_SERIES_SIM_UNIQUE_PENDING_CROSS_SECTION` | 19 |

This series-level result is stronger in time coverage but is not equivalent to
cross-sectional value similarity. It is therefore retained as supporting
evidence only and cannot replace the full cross-sectional similarity gate.

## 2025 independent validation

The 21 monthly-sample unique candidates were then evaluated against the
2025-01-01–2025-12-31 validation labels using the same strict source
materialization path. The validation artifact contains 5,103 daily RankIC rows
and 243 valid trading days per candidate. The new snapshot scope
`SCREEN_TRAIN2020_2024_VALIDATION2025_UNIQUE21_V1` fixes direction from the
TRAIN window and applies the double gate: absolute mean RankIC, absolute ICIR,
directional share, positive signed mean, and 5-lag HAC Benjamini–Hochberg
q-value ≤ 0.10.

| decision status | rows |
|---|---:|
| `VALIDATION_2025_PASS_PENDING_FINAL_SIMILARITY` | 18 |
| `VALIDATION_2025_FAIL` | 3 |

The three validation failures are:

| factor | failed condition |
|---|---|
| `yeslab_41` | validation ICIR `0.434583` < `0.50` |
| `yeslab_439` | validation ICIR `-0.431176` < `0.50` |
| `yeslab_473` | validation ICIR `-0.487262` < `0.50` |

The 18 passing IDs are:

```text
yeslab_155, yeslab_163, yeslab_363, yeslab_385, yeslab_395, yeslab_396,
yeslab_401, yeslab_405, yeslab_425, yeslab_437, yeslab_442, yeslab_443,
yeslab_471, yeslab_484, yeslab_94, yeslab_95, yeslab_96, yeslab_98
```

These 18 were the current research admission candidates at the validation
stage, not a new active factor version. The full-day similarity result and the
remaining publication review are recorded below.

## Full-day cross-sectional similarity and final quality scope

The remaining full-day check was then executed for these 18 candidates across
all 1,212 TRAIN trading dates. The implementation streams one month at a time,
deduplicates finite technical keys once per source group, and accumulates daily
Spearman values without retaining the five-year panel. Scope:
`SCREEN_TRAIN2020_2024_SIMILARITY_FULL_DAY18_V2`.

All 18 are unique at the `0.90` absolute-Spearman threshold. The nearest-neighbor
similarity ranges from `0.294474` to `0.886670`, and every nearest-neighbor
comparison has 1,212 valid days. No candidate was removed by the formal full-day
cross-sectional gate.

The consolidated scope
`SCREEN_TRAIN2020_2024_FACTOR_QUALITY_FINAL_CANDIDATES_V1` now contains 18
`FACTOR_QUALITY_PASS_PENDING_PUBLICATION` rows. This is the final quality
candidate list for review; it does not alter `factor_registry`, the active
factor pointer, or the six-factor frozen contract. Model training remains
unstarted and the 2026 locked test remains unread.
