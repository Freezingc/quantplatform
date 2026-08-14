# Factor Quality Latest Registry And Canonical Alpha484 Design

## Goal

Create a compact, current-state factor-quality registry for model feature
screening, and resolve the remaining research evidence path for Alpha484
without changing an existing table, overwriting a historical factor value, or
adding an unapproved candidate to a model.

The registry replaces none of the existing V2 audit tables. In particular,
`factor_quality_snapshot` remains an append-only diagnostic record. The new
registry is the one small table a model-selection workflow reads.

## Context And Chosen Approach

Three quality-storage alternatives were considered:

1. Query `factor_quality_snapshot` directly. This was rejected because it has
   2,612 rows across 20 append-only scopes, JSON metrics, and lineage fields
   that are not necessary for training-time selection.
2. Create `factor_quality_latest` with one current row per factor identity.
   This is the selected approach. It keeps explicit numerical gates in typed
   columns and uses a full-table refresh, so storage remains on the order of
   the catalog size.
3. Store only an approved factor set. This was rejected because it loses
   rejected and deferred research outcomes, making failed experiments easy to
   repeat and making candidate diagnosis opaque.

For Alpha484, three alternatives were considered:

1. Require point-for-point equality with historical YesLab values. This stays
   blocked because the historical cross-section includes BJ and has distinct
   adjusted-price provenance.
2. Drop Alpha484 despite its predictive quality. This wastes a factor with a
   five-of-five annual training gate, a passing 2025 validation gate, and no
   formal-nine redundancy breach.
3. Define a separate Tushare SH/SZ candidate. This is the selected approach.
   It makes the canonical market scope and adjusted-price source explicit,
   retains historical `yeslab_484` unchanged, and requires the same strict
   quality gates before it can reach an explicit admission review.

## Factor Quality Latest Registry

### Storage Contract

Migration `012_factor_quality_latest.dos` creates only the new
`factor_quality_latest` table in `dfs://QuantResearchV2Test`. It is date
partitioned on `effective_date`, matching the existing V2 database layout, and
sorted by `factor_id`, `effective_date`.

The table has exactly one current row for every current factor identity. A
refresh validates the entire replacement frame first, clears the small table,
and inserts the validated frame. The refresh is an operational maintenance
step, not a publisher or a historical-version service. It is executed only
when no model read is running, and its post-write checks require a nonempty
catalog-complete table before the command returns.

There is only one version field: `effective_date`, the date that the current
quality conclusion takes effect. The table deliberately contains none of
`run_id`, `data_version`, `created_at`, `source_watermark_id`, or
`record_hash`.

| Group | Columns |
|---|---|
| Identity | `factor_id`, `source_factor_id`, `effective_date` |
| Decision | `decision`, `model_eligible`, `monitoring_required`, `reason_code` |
| Scope | `market_scope`, `label_id`, `train_start`, `train_end`, `validation_start`, `validation_end`, `frozen_direction` |
| Training | `train_valid_days`, `train_signed_rank_ic`, `train_icir`, `train_directional_share`, `annual_gate_years`, `train_max_similarity` |
| Validation | `validation_valid_days`, `validation_signed_rank_ic`, `validation_icir`, `validation_directional_share`, `validation_max_similarity` |
| Evidence | `evidence_path`, `evidence_commit` |

`factor_id` is the exact candidate identity read by downstream code.
`source_factor_id` links a derived candidate to its historical parent without
claiming that their values are interchangeable.

### Decision Semantics

- `APPROVED`: currently eligible for a model feature query.
- `APPROVED_MONITOR`: currently eligible, with `monitoring_required=true`.
- `PENDING_EXPLICIT_ADMISSION`: quality evidence may be complete but the
  factor is not a model input.
- `DEFER`: needs a distinct repair or evidence step and is not a model input.
- `REJECT`: failed a fixed quality gate and is not a model input.
- `UNSCREENED`: has no complete strict predictive-quality evidence and is not
  a model input.
- `EXCLUDED`: synthetic or out-of-scope and is not a model input.

The initial refresh reads the 487 catalog identities from YesLab metadata,
the formal-nine scoreboard, the Alpha95 audit, and the Alpha484 canonical
review. It must yield:

- all 487 catalog identities represented exactly once;
- the formal nine classified as `APPROVED` or `APPROVED_MONITOR` and model
  eligible;
- all Alpha95 variants classified `REJECT` and model ineligible; and
- Alpha484's new candidate represented separately and model ineligible until
  an explicit admission decision.

The old snapshot table and local/Git artifacts retain detailed audit history;
this latest table retains only values needed for filtering and ranking.

### Reader Contract

Provide a typed reader that rejects mixed scope records and returns only rows
where all of the following hold:

```text
model_eligible = true
market_scope = SHSZ
label_id = forward_open_5d:v1
```

The reader must never turn a quality-table row into a frozen factor set. The
existing admission contract remains the control point for any actual model
training.

## Canonical Alpha484 Candidate

### Identity And Formula

The candidate identity is `yeslab_484_tushare_shsz_v1`; its
`source_factor_id` is `yeslab_484`. It uses only Tushare SH/SZ daily prices and
adjustment factors, excludes BJ, and implements the supplied
`HerdingCSADStd21Calculator` logic:

1. calculate trailing adjusted-close returns;
2. over each trailing 120 trading-day window, retain stocks with at least 20
   finite nonzero return observations;
3. select the nine highest-correlated peers for each central stock;
4. include the central stock in its ten-stock group and calculate daily CSAD;
5. calculate the sample standard deviation (`ddof=1`) of the trailing 21 daily
   CSAD observations.

The candidate uses only observations available on or before its signal date.
It never replays historical YesLab values as a correctness oracle and it does
not overwrite `yeslab_484`.

### Evidence And Gates

The already recorded Tushare SH/SZ evidence is revalidated as a new canonical
candidate:

- training: 2020-01-02 through 2024-12-31;
- validation: 2025-01-02 through 2025-12-31;
- label: raw five-trading-day adjusted-open forward return;
- frozen direction: `-1` from training only;
- predictive gates: signed mean RankIC > 0.02, signed ICIR >= 0.50, and
  directional-day share >= 0.55;
- annual gate: at least four of five training years pass the predictive gates;
- similarity: median absolute daily Spearman < 0.90 for all nine formal-factor
  pairs in training and validation; and
- coverage: the candidate must satisfy the existing structural threshold of at
  least 80% median same-date coverage, at least 70% fifth-percentile coverage,
  and no monthly coverage below 70% relative to the canonical valid-label
  universe.

The legacy YesLab/Tushare equality mismatch is recorded as provenance, not as
a failure of this new candidate. Tushare availability and adjusted-price
consistency are checked inside the canonical scope itself. A coverage or
formula-semantics failure rejects the candidate.

Passing these gates produces only
`READY_FOR_EXPLICIT_ADMISSION_REVIEW`; it does not change the formal nine,
write factor observations, create a frozen factor set, train a model, or read
the 2026 locked test.

## Safety And Failure Handling

- Existing YesLab and V2 table schemas are never modified.
- All research evaluation is SH/SZ-only and must reject an end date on or after
  2026-01-01.
- The quality registry refresh has an explicit `--publish` switch; its default
  is dry-run and no database write.
- Invalid source artifacts, duplicate factor IDs, nonfinite required metrics,
  incomplete catalog coverage, or a failed formal-nine count stop the refresh
  before any table replacement.
- An Alpha484 coverage, predictive, annual, similarity, or formula-semantic
  failure yields `REJECT` or `DEFER`, never a silent fallback to the historical
  source factor.

## Verification

Before a database write:

1. unit tests validate table-frame construction, one-row-per-factor identity,
   no lineage columns, model-eligibility filtering, and dry-run behavior;
2. migration tests assert only the new table and its compact columns;
3. Alpha484 tests check trailing-only computation, central-stock inclusion,
   deterministic peer ordering, SH/SZ exclusion, locked-test rejection, and
   candidate identity isolation;
4. the complete `quantyd` test suite and Ruff run cleanly.

After an explicit publish:

1. query the new table and verify 487 catalog rows plus the isolated Alpha484
   candidate with no duplicate `factor_id` values;
2. verify the formal-nine eligible count is nine and Alpha95/Alpha484 remain
   model ineligible; and
3. read back the exact typed metrics and compare them with their source
   artifacts.
