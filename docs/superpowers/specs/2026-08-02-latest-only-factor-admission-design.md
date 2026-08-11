# Latest-only compact storage and strict factor admission

- Status: approved for implementation
- Date: 2026-08-02
- Scope: add compact latest-only research tables without modifying existing V2 tables; enforce frozen factor-set admission at training boundaries.

## Goal

Reduce repeated row-level lineage in high-volume research facts while preserving the correctness guarantees that matter for quantitative research: point-in-time availability, deterministic factor selection, train-only preprocessing, and reproducible out-of-sample evaluation.

## Non-goals

- Do not alter or drop columns from existing `dfs://QuantResearchV2Test` tables.
- Do not recompute historical labels, factor RankIC/ICIR, or already published OOS results solely because storage changes.
- Do not allow latest-only tables to become an ungoverned overwrite path.
- Do not let an arbitrary factor-name list bypass factor-set admission.

## Architecture

Existing V2 tables remain immutable/read-only archive sources. New `*_latest` tables contain only the current published business rows for daily high-volume objects. A small `latest_manifest` table records one batch per object and logical partition, including publication time, row count, and business-content hash. The compact tables are built from the current V2 pointers and are verified against those pointers before consumers switch.

The factor admission boundary remains strict and low-volume. A frozen factor set is the only legal model feature source. Admission checks require factor identity/version/order/direction agreement, explicit exclusion of synthetic test factors, a configurable minimum finite-coverage ratio, and exact feature-frame columns. Training data and preprocessing state must match the admitted factor contract exactly; RankIC/ICIR gates remain downstream evaluation gates.

## New database objects

Migration `008_latest_compact_tables.dos` creates:

- `latest_manifest`: `object_name`, `logical_partition`, `batch_id`, `published_at`, `row_count`, `business_hash`, `source_object`, `source_data_version`, `status`.
- `tradability_daily_latest`: business columns from `tradability_daily`.
- `universe_daily_latest`: business columns from `universe_daily`.
- `label_daily_latest`: business columns from `label_daily`.
- `factor_observation_clean_latest`: business columns from `factor_observation_clean`.

The compact tables retain their business dates and keys plus a numeric `batch_id`. They omit repeated `run_id`, `data_version`, `created_at`, `source_watermark_id`, and `record_hash`. Consumers filter by the `batch_id` recorded as `PUBLISHED` in `latest_manifest`; old compact batches can be garbage-collected after verification. Low-volume governance and evaluation tables retain complete lineage.

## Data flow and safety

1. Resolve the current `published_version` for each source object and logical partition.
2. Copy only rows from that exact source `data_version` into a staged compact frame.
3. Validate row count, business-key uniqueness, date partition, finite values, and a deterministic hash over ordered business columns.
4. Insert the compact batch and advance `latest_manifest` only after validation succeeds.
5. Consumers read compact tables only after the manifest status is `PUBLISHED`.

The first implementation publishes compact batches append-only under a numeric `batch_id` and uses the manifest as the visibility boundary. Existing V2 pointers remain the rollback source. A future physical partition-replacement optimization is intentionally separate from this migration.

## Factor admission contract

`FactorAdmissionContract` contains `factor_set_id`, ordered members, frozen directions, selection scope, rules hash, and preprocessing hash. `admit_feature_frame` rejects unknown/duplicate/missing factors, wrong order or version, insufficient finite coverage, and any factor set that is not `FROZEN`. It also rejects factor IDs on the explicit synthetic-test exclusion list. The PIT dataset builder checks the frozen `(factor_id, factor_version)` sequence before joining values.

The contract is independent of row storage: latest-only facts may omit lineage columns, while factor-set evidence remains reproducible in `factor_set`, `factor_set_member`, `factor_rank_ic_summary`, and model manifests.

## Verification

- Unit tests cover manifest identity, compact materialization, factor admission failures, and exact feature-order enforcement.
- Migration discovery and schema tests cover all new tables.
- A database backfill derives the manifest business-content hash from the exact current source frame, and verification independently checks pointer coverage, current-batch row counts, compact schema, and preservation of the source lineage schema.
- Existing regression suite, Ruff, and mypy must remain clean.
