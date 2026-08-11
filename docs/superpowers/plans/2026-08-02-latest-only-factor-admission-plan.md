# Latest-only compact storage and strict factor admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with verification checkpoints.

**Goal:** Add compact latest-only research tables and a hard factor-admission contract without changing existing V2 tables or recomputing historical research results.

**Architecture:** Existing V2 tables remain the archive and source of current published partitions. New compact tables store business columns plus a numeric `batch_id`; `latest_manifest` maps each object/partition to the published batch. A pure Python factor-admission contract validates frozen factor-set identity, order, direction, exclusions, and feature-frame schema before training.

**Tech Stack:** Python 3.11, pandas, PyTorch contracts, DolphinDB `.dos` migrations, pytest, Ruff, mypy.

---

### Task 1: Define compact batch and manifest contracts

**Files:**
- Create: `research_v2/src/research_v2/compact/__init__.py`
- Create: `research_v2/src/research_v2/compact/contracts.py`
- Test: `research_v2/tests/unit/compact/test_contracts.py`

- [x] **Step 1–4:** Implemented and tested `LatestManifestEntry`, `CompactManifestStatus`, and deterministic positive `compute_batch_id`; invalid identity, hash, and row-count values are rejected.

### Task 2: Implement latest-only business-frame materialization

**Files:**
- Create: `research_v2/src/research_v2/compact/materializer.py`
- Test: `research_v2/tests/unit/compact/test_compact_materializer.py`

- [x] **Step 1–5:** Implemented `build_compact_materialization` with four allowlisted object schemas, exact-version filtering, date/key/finite checks, deterministic business hash, and non-mutating projection tests.

### Task 3: Add latest compact database migration

**Files:**
- Create: `research_v2/migrations/008_latest_compact_tables.dos`
- Test: `research_v2/tests/unit/db/test_migrations.py`

- [x] **Step 1–4:** Added and applied idempotent migration 008. The five new tables have `batch_id`/business columns only; existing V2 tables were not altered.

### Task 4: Implement strict factor-admission contract

**Files:**
- Create: `research_v2/src/research_v2/factors/admission.py`
- Create: `research_v2/tests/unit/factors/test_admission.py`

- [x] **Step 1–5:** Implemented `FactorAdmissionContract`/`admit_feature_frame` with FROZEN status, exact IDs/versions/order/directions, synthetic exclusions, finite-coverage threshold, duplicate-key rejection, and rules/preprocessor hashes.

### Task 5: Connect admission to dataset/model training boundary

**Files:**
- Modify: `research_v2/src/research_v2/datasets/builder.py`
- Create: `research_v2/src/research_v2/datasets/reader.py`
- Test: `research_v2/tests/unit/datasets/test_dataset_builder.py`
- Test: `research_v2/tests/unit/datasets/test_reader.py`

- [x] **Step 1–5:** Added an optional `admission_contract` and finite-ratio threshold to the PIT dataset builder, plus a production-facing `CompactDatasetReader` that reads every requested latest month, fails closed on missing manifests, filters factor IDs/versions, and invokes the builder. It checks exact frozen member identities before joins and exact feature columns before returning a training frame; existing FINAL/TEST locking remains unchanged. The low-level builder remains backward-compatible when no contract is supplied.

### Task 6: Backfill current V2 pointers into compact tables

**Files:**
- Create: `research_v2/src/research_v2/compact/publisher.py`
- Create: `research_v2/src/research_v2/compact/verification.py`
- Test: `research_v2/tests/unit/compact/test_publisher.py`
- Test: `research_v2/tests/unit/compact/test_verification.py`

- [x] **Step 1–4:** Implemented a session-injected current-pointer backfill/publisher with typed DolphinDB inserts, manifest-last visibility, idempotent retries, daily chunking for the large factor-observation partition, and verification helpers. Source V2 tables are read-only in this path.
- [x] Streaming source publication now explicitly undefines each temporary
  DolphinDB staging variable after insert, keeping long factor backfills from
  retaining one staging frame per chunk.

### Task 7: Apply migration, backfill, and verify database equality

**Files:**
- Modify only database state through `MigrationRunner` and the new backfill module.
- Create: `research_v2/tests/integration/compact/test_latest_backfill.py` if an integration fixture is available.

- [x] **Step 1–5:** Applied migration 008 and backfilled all current pointers in `dfs://QuantResearchV2Test`; 219 published manifests cover 73/73/72/1 source partitions. Current-batch row counts match manifests, compact schemas contain no row-level lineage fields, and source schemas retain all five lineage fields.

### Task 8: Full verification and handoff

- [x] Run `cd research_v2 && pytest -q` (latest: 257 passed).
- [x] Run `cd research_v2 && ruff check src tests`.
- [x] Run `cd research_v2 && python -m mypy src`.
- [x] Query database pointers and manifest counts; record exact evidence in the handoff.
- [x] Existing V2 tables remain untouched. The compact reader and admission-aware builder are available; callers must explicitly opt into compact reads and pass the frozen admission contract.

### Task 9: Source factor backfill preflight

**Files:**
- Modify: `research_v2/src/research_v2/factors/source_backfill.py`
- Modify: `research_v2/src/research_v2/factors/__init__.py`
- Test: `research_v2/tests/unit/factors/test_source_backfill.py`

- [x] Add a write-free month preflight that consumes one materialization stream,
  checks clean/quarantine publication identity, and reports per-factor row
  counts before any V2 pointer is touched.
- [x] Run the preflight against F03/`2020-01`; it failed closed because
  `yeslab_464` had no clean observations (source absent or fully quarantined).
- [x] Record the stop condition and the exact current F03 source/compact
  versions in the smoke evidence report. Historical backfill remains pending a
  cross-month availability audit and a newly frozen factor set.

### Task 10: Freeze a cross-month candidate set and rebuild its compact history

**Files:**
- Create: `research_v2/src/research_v2/factors/source_evaluation.py`
- Modify: `research_v2/src/research_v2/factors/deduplication.py`
- Modify: `research_v2/src/research_v2/factors/source_backfill.py`
- Create: `research_v2/src/research_v2/baselines/rank_ensemble.py`
- Test: `research_v2/tests/unit/factors/test_source_evaluation.py`
- Test: `research_v2/tests/unit/baselines/test_rank_ensemble.py`

- [x] Audit source coverage and PIT RankIC in bounded monthly chunks without
  mutating YesLab or V2.
- [x] Freeze the six-factor set
  `b8fd9f5f82ffe546cbc1738539ab46b6e055f44782de50008a535e83cdfd0e6c` after
  the combined 2020–2024 training and 2025 validation dual gate.
- [x] Vectorize duplicate-group reduction and distinguish blocking conflicts
  from all-null coverage gaps; retain the explicit 70% admission gate.
- [x] Backfill and compact-publish all 72 candidate partitions from 2020-01
  through 2025-12 (41,619,921 current compact rows) with a resumable
  single-writer job.
- [x] Add a scale-free directional rank baseline. The full 2020 diagnostic is
  mean RankIC 0.053124 and ICIR 0.358236; final validation and locked-test
  handling remain separate gates.
- [x] Remove the obsolete 15-factor constructor floor from the four model
  families. Their valid range is now 1–60, while the frozen admission contract
  remains the authoritative factor-count and identity gate.
- [x] Bound the publisher's same-version orphan check by the declared logical
  month to avoid full-table DolphinDB scans during long factor backfills.
- [x] Complete the purged FINAL validation comparison: rank equal-weight
  mean RankIC 0.090791 / ICIR 0.577179 versus date-equal Ridge mean RankIC
  0.086343 / ICIR 0.572690; leave 2026 TEST locked.

### 2026 locked-test continuation checkpoint

The authorized 2026 workflow now uses an explicit
`LockedTestAccess(2026, "user-approved-final-evaluation")` capability at both
the dataset split and label boundaries. Seven 2026 monthly partitions of
market, tradability, universe, labels, and the six frozen factors were
materialized in `dfs://QuantResearchV2Test` without changing any existing
schema. The source market ends at 2026-07-31, so the available test scope is
YTD rather than a complete calendar-year test.

The strict admission contract failed closed: `GP_022` finite coverage was
0.660249 and `GP_064` was 0.658257 versus the 0.70 minimum. The failure is
concentrated in May–July `factor_data9` observations. The locked audit records
the authorization and a consumed `REJECTED_INPUT_QUALITY` event; no official
2026 RankIC/ICIR row was published and no factor/model retuning was performed.
The January–April quality-qualified diagnostic (not a promotion result) was
mean RankIC 0.040184, ICIR 0.266067 over 77 days. A full 2026 OOS evaluation is
pending upstream factor coverage repair and a new explicit locked-test
authorization.

The compact reader was then tightened to preflight factor finite coverage before
querying locked labels. The regression suite is now 260 tests, with Ruff and
mypy clean; the strict failure remains visible rather than being bypassed.
