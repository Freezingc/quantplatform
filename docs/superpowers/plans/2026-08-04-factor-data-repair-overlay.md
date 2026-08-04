# Factor Data Repair Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with review checkpoints.

**Goal:** Add a read-only, point-in-time-safe repair overlay for the 47 source-qualified factors, persist repair audits/quarantine in new V2 tables, and re-run data-quality checks without changing source schemas or training models.

**Architecture:** Pure DataFrame repair contracts perform duplicate/conflict/PIT decisions before any database write. A bounded source runner reads one source-table/month group at a time, applies only registered repair providers, and publishes new `factor_observation_clean` data versions plus new repair audit/quarantine objects through the existing V2 publisher. TRAIN is repaired and rechecked first; VALIDATION is processed only with the same repair rules and no locked-test access.

**Tech Stack:** Python 3.11 (`quantyd`), pandas/NumPy, DolphinDB V2 sessions and migrations, existing `StagedBatch`/`V2BatchPublisher`, pytest, Ruff, mypy.

---

## File map

- Modify: `research_v2/src/research_v2/factors/repair.py` — extend the existing no-overwrite helpers with source-key audit and overlay decision contracts.
- Create: `research_v2/src/research_v2/factors/repair_overlay.py` — provider protocol, bounded overlay assembly, and repair decision/result types.
- Modify: `research_v2/src/research_v2/factors/source_backfill.py` — expose a factor/month stream that accepts an optional repair provider without changing normal materialization behavior.
- Create: `research_v2/src/research_v2/migrations/009_factor_repair_tables.dos` — new audit and quarantine tables only.
- Modify: `research_v2/src/research_v2/db/publish.py` only if the audit/quarantine batch needs a narrowly scoped typed-column conversion; existing publish semantics remain unchanged.
- Create: `research_v2/scripts/run_factor_repair_overlay.py` — dry-run-by-default TRAIN/VALIDATION runner with an explicit `--publish` gate.
- Create: `research_v2/scripts/compare_repaired_factor_quality.py` — before/after coverage and RankIC comparison report.
- Modify: `research_v2/tests/unit/factors/test_repair.py` — red/green tests for source audit and no-overwrite repair behavior.
- Create: `research_v2/tests/unit/factors/test_repair_overlay.py` — provider, PIT, unresolved, and batch assembly tests.
- Modify: `research_v2/tests/unit/db/test_migrations.py` — migration 009 table declarations.
- Create: `research_v2/tests/unit/scripts/test_factor_repair_runner.py` — CLI scope, dry-run, and locked-test guard tests.
- Create: `docs/reports/2026-08-04-factor-data-repair.md` — generated runbook/report after the database run, not hand-edited before evidence exists.

## Task 1: Add source-key and candidate audit contracts

**Files:**
- Test: `research_v2/tests/unit/factors/test_repair.py`
- Modify: `research_v2/src/research_v2/factors/repair.py`

- [ ] **Step 1: Write failing tests for exact duplicates, conflicts, finite overwrite protection, and unresolved candidates.**

Add tests with the existing small factor frames. The expected API is:

```python
audit = audit_factor_source(source)
assert audit.exact_duplicate_groups == 1
assert audit.conflict_groups == 1

decision = apply_repair_overlay(source, candidate, factor_name="GP_022")
assert decision.status == RepairStatus.BLOCKED
assert decision.quarantine["reason"].tolist() == ["CONFLICT_QUARANTINE"]
```

Also assert that an existing finite source value remains unchanged when a
candidate supplies a different value, and that a candidate dated after the
signal date is quarantined with `PIT_DATE_VIOLATION`.

- [ ] **Step 2: Run the focused tests and verify they fail for missing APIs.**

Run:

```bash
/data/zhangyudong/.conda/envs/quantyd/bin/python -m pytest research_v2/tests/unit/factors/test_repair.py -q
```

Expected result: failures naming the not-yet-defined audit/overlay functions,
not import or fixture errors.

- [ ] **Step 3: Implement the minimal audit contract.**

Add immutable types and functions without changing the existing public helpers:

```python
class RepairStatus(StrEnum):
    PASS = "REPAIR_AUDIT_PASS"
    PASS_WITH_QUARANTINE = "REPAIR_AUDIT_PASS_WITH_UNRESOLVED_QUARANTINE"
    BLOCKED = "REPAIR_BLOCKED"
    PENDING_RECHECK = "REPAIRED_PENDING_QUALITY_RECHECK"

@dataclass(frozen=True)
class SourceKeyAudit:
    source_rows: int
    exact_duplicate_groups: int
    exact_duplicate_rows: int
    conflict_groups: int
    nonfinite_rows: int
    finite_rows: int

def audit_factor_source(source: pd.DataFrame) -> SourceKeyAudit: ...

def apply_repair_overlay(
    source: pd.DataFrame,
    candidate: pd.DataFrame | None,
    *,
    factor_name: str,
    signal_date: date | None = None,
    abs_tolerance: float = 1e-8,
    rel_tolerance: float = 1e-6,
) -> RepairOverlayResult: ...
```

`apply_repair_overlay` must normalize the existing long-form columns, collapse
only identical duplicate groups, quarantine all distinct finite conflicts,
fill only non-finite values from unique finite candidates, and return clean and
quarantine frames plus deterministic counts. It must never select a conflicting
value or overwrite a finite source observation.

- [ ] **Step 4: Run focused tests and then the existing repair test module.**

Run:

```bash
/data/zhangyudong/.conda/envs/quantyd/bin/python -m pytest research_v2/tests/unit/factors/test_repair.py -q
```

Expected result: all repair tests pass, including the pre-existing GP_022,
GP_064, market-row, and finite-fill tests.

- [ ] **Step 5: Refactor only after green.**

Keep normalization and conflict classification private, preserve the existing
`FactorRepairAudit` compatibility fields, and run the focused tests again.

## Task 2: Add bounded repair providers and overlay assembly

**Files:**
- Test: `research_v2/tests/unit/factors/test_repair_overlay.py`
- Create: `research_v2/src/research_v2/factors/repair_overlay.py`
- Modify: `research_v2/src/research_v2/factors/source_backfill.py`

- [ ] **Step 1: Write failing provider and PIT tests.**

Cover four cases: a provider returns a finite matching candidate; a provider
returns duplicate candidate keys; no provider exists; and a candidate violates
same-day technical availability. Use a session double and in-memory frames,
never a live database.

```python
def test_overlay_provider_can_fill_only_nonfinite_values() -> None:
    result = build_repair_overlay(source, provider, definition=definition)
    assert result.status is RepairStatus.PENDING_RECHECK
    assert result.filled_rows == 1
    assert result.clean.loc[0, "value"] == 2.0
```

- [ ] **Step 2: Run the new module and verify the expected missing-interface failures.**

Run:

```bash
/data/zhangyudong/.conda/envs/quantyd/bin/python -m pytest research_v2/tests/unit/factors/test_repair_overlay.py -q
```

- [ ] **Step 3: Implement provider and result boundaries.**

Define a provider protocol that receives one factor/month source frame and
returns either `None` or a candidate frame with `(trade_date, ts_code,
candidate_value)`. Implement:

- `NullRepairProvider`, which records `NO_EVIDENCE_BACKFILL` and never fabricates rows;
- `MarketCandidateProvider`, which delegates to existing
  `fill_missing_market_rows` and `reconcile_market` and requires a secondary
  watermark;
- `KnownFormulaCandidateProvider`, which delegates GP_022/GP_064 to the
  existing exact calculator functions and rejects any overlap mismatch;
- `build_repair_overlay`, which calls `apply_repair_overlay` and annotates the
  repair method, source watermark, candidate hash, and status.

No provider may use validation/test metrics. The provider mapping must be
explicit by factor ID; unknown factors use `NullRepairProvider`.

- [ ] **Step 4: Add a source-backfill hook that preserves the normal path.**

Extend the factor/month iterator with an optional keyword-only
`repair_provider: RepairProvider | None = None`. When it is absent, output
must be byte-for-byte equivalent at the business columns to the current
materialization. When present, run the provider before
`build_technical_factor_batches`, and return repair audit metadata alongside
the clean/quarantine batches.

- [ ] **Step 5: Run factor, backfill, and overlay tests.**

Run:

```bash
/data/zhangyudong/.conda/envs/quantyd/bin/python -m pytest \
  research_v2/tests/unit/factors/test_repair.py \
  research_v2/tests/unit/factors/test_repair_overlay.py \
  research_v2/tests/unit/factors/test_source_backfill.py -q
```

## Task 3: Create new repair audit and quarantine tables

**Files:**
- Test: `research_v2/tests/unit/db/test_migrations.py`
- Create: `research_v2/migrations/009_factor_repair_tables.dos`

- [ ] **Step 1: Add a failing migration declaration test.**

Assert migration 009 exists and contains `factor_repair_audit` and
`factor_repair_quarantine`, with the columns used by the Python batch builders:
`logical_partition`, `factor_id`, `factor_version`, `repair_method`,
`decision_status`, `filled_rows`, `conflict_groups`, `run_id`, `data_version`,
`source_watermark_id`, and `record_hash`.

- [ ] **Step 2: Run the migration test and confirm it fails because 009 is absent.**

Run:

```bash
/data/zhangyudong/.conda/envs/quantyd/bin/python -m pytest \
  research_v2/tests/unit/db/test_migrations.py::test_repository_repair_migration_declares_repair_tables -q
```

- [ ] **Step 3: Add migration 009 without touching migrations 001–008.**

Create `factorRepairAudit` partitioned by `as_of_date` and sorted by
`factor_id`/`logical_partition`; create `factorRepairQuarantine` partitioned by
`observation_date` and sorted by factor/key/reason. Both tables use the existing
lineage convention (`run_id`, `data_version`, `created_at`,
`source_watermark_id`, `record_hash`). The quarantine table stores source and
candidate values as nullable `DOUBLE` columns and reason as `SYMBOL`.

- [ ] **Step 4: Run all migration unit tests.**

Run:

```bash
/data/zhangyudong/.conda/envs/quantyd/bin/python -m pytest research_v2/tests/unit/db/test_migrations.py -q
```

## Task 4: Build repair audit/quarantine batches and atomic persistence

**Files:**
- Test: `research_v2/tests/unit/factors/test_repair_overlay.py`
- Create: `research_v2/src/research_v2/factors/repair_persistence.py`
- Modify: `research_v2/src/research_v2/db/publish.py` only if typed staging requires it

- [ ] **Step 1: Write failing batch identity and atomicity tests.**

Test that audit and quarantine batches share one run/data version, that a
duplicate production key is rejected before upload, and that a failed blocker
does not advance the visibility pointer.

- [ ] **Step 2: Implement pure batch builders.**

Expose:

```python
def build_repair_audit_batch(records: Iterable[RepairAuditRecord]) -> StagedBatch: ...
def build_repair_quarantine_batch(rows: Iterable[RepairQuarantineRow]) -> StagedBatch | None: ...
def repair_quality_checks(audit: RepairAuditRecord) -> list[CheckResult]: ...
```

Build deterministic `data_version` and `source_watermark_id` values from the
sorted factor IDs, source definition hashes, date range, provider hashes, and
rules version. Use `V2BatchPublisher.publish_chunk_stream` so no complete
five-year frame is retained in client memory. Audit and quarantine writes must
occur before the repaired factor pointer is advanced; a failed blocker writes
no visible repaired version.

- [ ] **Step 3: Run persistence and publisher tests.**

Run:

```bash
/data/zhangyudong/.conda/envs/quantyd/bin/python -m pytest \
  research_v2/tests/unit/factors/test_repair_overlay.py \
  research_v2/tests/unit/db/test_publish.py -q
```

## Task 5: Add the dry-run-first repair runner

**Files:**
- Test: `research_v2/tests/unit/scripts/test_factor_repair_runner.py`
- Create: `research_v2/scripts/run_factor_repair_overlay.py`

- [ ] **Step 1: Write failing CLI guard tests.**

Test that the runner:

- resolves factor IDs only from `SCREEN_TRAIN2020_2024_SOURCE_DQ_COMBINED_V1` with `SOURCE_DQ_PASS_PENDING_SIMILARITY`;
- defaults to `--dry-run` behavior when `--publish` is absent;
- accepts only `--phase train` or `--phase validation`;
- refuses any end date in or after 2026;
- refuses a requested factor outside the resolved 47-factor set unless an
  explicit `--factor-id` is used for a single audit fixture;
- does not issue a source write through `YesLabReadOnlySession`.

- [ ] **Step 2: Run the CLI tests and verify missing-runner failures.**

Run:

```bash
/data/zhangyudong/.conda/envs/quantyd/bin/python -m pytest \
  research_v2/tests/unit/scripts/test_factor_repair_runner.py -q
```

- [ ] **Step 3: Implement bounded source reads and explicit provider mapping.**

The runner must:

1. connect with `YesLabReadOnlySession` and `V2WriteSession` using environment
   variables already used by the repository;
2. query the candidate IDs from the source-quality scope;
3. query metadata and group factors by source table;
4. iterate each month and source-table group with a bounded `_GROUP_SIZE`;
5. apply the explicit provider map (`yeslab_391`/`yeslab_433` for known formula
   repairs, secondary market reconciliation only when a configured Tushare
   token and cache are available, `NullRepairProvider` otherwise);
6. build audit/quarantine/clean batches and print counts in dry-run mode;
7. require `--publish` before applying migration-backed writes;
8. never query `locked_test_audit`, 2026 rows, or model tables.

The default TRAIN command is:

```bash
/data/zhangyudong/.conda/envs/quantyd/bin/python \
  research_v2/scripts/run_factor_repair_overlay.py \
  --phase train --start 2020-01-01 --end 2024-12-31
```

Publishing is a separate explicit command after dry-run review:

```bash
/data/zhangyudong/.conda/envs/quantyd/bin/python \
  research_v2/scripts/run_factor_repair_overlay.py \
  --phase train --start 2020-01-01 --end 2024-12-31 --publish
```

- [ ] **Step 4: Run the CLI unit tests and repository static checks.**

Run:

```bash
/data/zhangyudong/.conda/envs/quantyd/bin/python -m pytest \
  research_v2/tests/unit/scripts/test_factor_repair_runner.py -q
/data/zhangyudong/.conda/envs/quantyd/bin/ruff check research_v2/src research_v2/scripts research_v2/tests
/data/zhangyudong/.conda/envs/quantyd/bin/python -m mypy research_v2/src/research_v2
```

## Task 6: Add before/after quality comparison and execute TRAIN repair

**Files:**
- Create: `research_v2/scripts/compare_repaired_factor_quality.py`
- Create: `docs/reports/2026-08-04-factor-data-repair.md`
- Test: `research_v2/tests/unit/scripts/test_factor_repair_runner.py`

- [ ] **Step 1: Add tests for deterministic comparison output.**

Given two small coverage/RankIC summary frames, assert the report contains
`before`, `after`, `filled_rows`, `remaining_nonfinite_rows`, and quality delta
columns sorted by factor ID.

- [ ] **Step 2: Implement comparison output.**

Read only the repair audit and existing quality snapshots; write a CSV/Markdown
summary without replacing any existing snapshot. The report must identify
factors that were repaired, unchanged, blocked, or unresolved and include the
repair `data_version`.

- [ ] **Step 3: Run the TRAIN dry-run and review blocker counts.**

Run the dry-run command from Task 5 and save its stdout/artifact manifest. Do
not publish a batch when any source table has unresolved conflict keys or when
candidate overlap checks fail.

- [ ] **Step 4: Apply migration 009 in the V2 test database only.**

Run the existing migration CLI in dry-run mode first, verify migration 009 is
the only planned new migration, then apply it to
`dfs://QuantResearchV2Test`. Do not target `dfs://QuantResearchV2` in this
phase.

- [ ] **Step 5: Publish the TRAIN repair overlay only after the dry-run gate passes.**

Use `--publish` with the exact 2020–2024 TRAIN range. Record the resulting
repair audit status counts and pointer/version identities in the report.

## Task 7: Recheck repaired TRAIN and 2025 VALIDATION without model training

**Files:**
- Modify: `research_v2/scripts/compare_repaired_factor_quality.py`
- Modify: `docs/reports/2026-08-04-factor-data-repair.md`

- [ ] **Step 1: Re-run source/PIT coverage and daily RankIC against the repaired TRAIN version.**

Use the existing evaluation functions and the repaired data version only. Keep
the original snapshot scope unchanged and write a new repair-recheck scope.

- [ ] **Step 2: Compare repaired TRAIN metrics with the original metrics.**

Report finite coverage, duplicate/conflict counts, valid RankIC days, mean
RankIC, ICIR, HAC/FDR, and direction separately for every repaired factor.

- [ ] **Step 3: Run the 2025 VALIDATION dry-run with TRAIN-fixed repair rules.**

Use the same provider mapping and repair rules, but do not tune providers from
2025 metrics. Do not read 2026 data. Publish only after the user-visible TRAIN
comparison has no unresolved blocker.

- [ ] **Step 4: Verify that model training and active factor pointers remain unchanged.**

Query only the active factor/model pointer metadata and assert no new model run
or active factor version was created. Include these checks in the report.

## Final verification checklist

- [ ] Run the full test suite:

```bash
/data/zhangyudong/.conda/envs/quantyd/bin/python -m pytest
```

- [ ] Run Ruff and mypy over the complete research package.
- [ ] Run migration discovery and confirm migration checksums are stable.
- [ ] Confirm `YesLabReadOnlySession` saw no mutating scripts.
- [ ] Confirm no existing V2 table definition changed.
- [ ] Confirm all 47 requested factor/month combinations have an audit status.
- [ ] Confirm every unresolved row is in repair quarantine, not silently dropped.
- [ ] Confirm no model training command ran and no 2026 locked-test query ran.
- [ ] Update the repair report only with fresh command/database evidence.

