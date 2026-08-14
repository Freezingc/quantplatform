# Factor Quality Latest Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compact `factor_quality_latest` registry and typed reader so feature screening can select current SH/SZ quality-approved factors without reading append-only snapshots.

**Architecture:** Migration 012 adds the only new table. Pure Python builders merge the 487-factor catalog, formal-nine scoreboard, Alpha95 rejection artifact, and Alpha484 canonical review into 491 typed rows. A dry-run-first command validates the entire replacement frame before replacing only the small current-state table; the reader filters eligible rows but never constructs a factor set.

**Tech Stack:** Python 3.11, pandas, DolphinDB, pytest, Ruff.

---

### Task 1: Create the Compact Table Migration

**Files:**
- Create: `research_v2/migrations/012_factor_quality_latest.dos`
- Modify: `research_v2/tests/unit/db/test_migrations.py`

- [ ] **Step 1: Write the failing migration contract test**

```python
def test_repository_quality_latest_migration_declares_compact_columns() -> None:
    directory = Path(__file__).resolve().parents[3] / "migrations"
    migration = next(item for item in discover_migrations(directory) if item.version == 12)
    assert "`factor_quality_latest" in migration.script
    for column in (
        "factor_id", "source_factor_id", "effective_date", "decision",
        "model_eligible", "train_signed_rank_ic", "validation_icir",
        "evidence_path", "evidence_commit",
    ):
        assert f"`{column}" in migration.script
    for forbidden in (
        "run_id", "data_version", "created_at", "source_watermark_id", "record_hash",
    ):
        assert f"`{forbidden}" not in migration.script
```

- [ ] **Step 2: Verify RED**

Run:

```bash
source /data/share/miniconda3/etc/profile.d/conda.sh
conda activate quantyd
PYTHONPATH=research_v2/src python -m pytest -q research_v2/tests/unit/db/test_migrations.py -k quality_latest
```

Expected: FAIL with `StopIteration` because migration 012 is absent.

- [ ] **Step 3: Add migration 012 exactly as follows**

```dos
// targetDatabase is injected by the V2 migration runner.
db = database(targetDatabase)

if (!existsTable(targetDatabase, `factor_quality_latest)) {
    factorQualityLatest = table(1:0,
        `factor_id`source_factor_id`effective_date`decision`model_eligible`monitoring_required`reason_code`market_scope`label_id`train_start`train_end`validation_start`validation_end`frozen_direction`train_valid_days`train_signed_rank_ic`train_icir`train_directional_share`annual_gate_years`train_max_similarity`validation_valid_days`validation_signed_rank_ic`validation_icir`validation_directional_share`validation_max_similarity`evidence_path`evidence_commit,
        [SYMBOL, SYMBOL, DATE, SYMBOL, BOOL, BOOL, STRING, SYMBOL, SYMBOL, DATE, DATE, DATE, DATE, INT, INT, DOUBLE, DOUBLE, DOUBLE, INT, DOUBLE, INT, DOUBLE, DOUBLE, DOUBLE, DOUBLE, STRING, STRING])
    db.createPartitionedTable(table=factorQualityLatest, tableName=`factor_quality_latest,
        partitionColumns=`effective_date, sortColumns=`factor_id`effective_date)
}
```

- [ ] **Step 4: Verify GREEN and commit**

Run:

```bash
PYTHONPATH=research_v2/src python -m pytest -q research_v2/tests/unit/db/test_migrations.py -k quality_latest
git add research_v2/migrations/012_factor_quality_latest.dos research_v2/tests/unit/db/test_migrations.py
git commit -m "add factor quality latest schema"
```

Expected: the migration test passes before committing.

### Task 2: Build and Validate Current Quality Rows

**Files:**
- Create: `research_v2/src/research_v2/quality/latest.py`
- Create: `research_v2/tests/unit/quality/test_latest.py`

- [ ] **Step 1: Write failing row-builder tests**

```python
def test_build_quality_latest_frame_has_catalog_and_isolated_candidates() -> None:
    catalog = pd.DataFrame({"factor_id": [1, 2]})
    frame = latest.build_quality_latest_frame(
        catalog=catalog,
        scoreboard={"rows": [formal_row("yeslab_1")]},
        alpha95=alpha95_payload(), alpha484=alpha484_payload(),
        effective_date=date(2026, 8, 14), evidence_commit="a" * 40,
    )
    assert set(frame["factor_id"]) == {
        "yeslab_1", "yeslab_2",
        "yeslab_163_relative_amount_volatility_10_v1",
        "yeslab_163_relative_amount_volatility_20_v1",
        "yeslab_163_relative_amount_volatility_40_v1",
        "yeslab_484_tushare_shsz_v1",
    }
    assert frame["factor_id"].is_unique
    assert frame.loc[frame["factor_id"] == "yeslab_1", "model_eligible"].item()
    assert not frame.loc[frame["factor_id"].str.contains("relative_amount"), "model_eligible"].any()


def test_validate_quality_latest_frame_rejects_lineage_fields() -> None:
    with pytest.raises(ValueError, match="forbidden columns"):
        latest.validate_quality_latest_frame(
            pd.DataFrame([valid_row()]).assign(run_id="forbidden"),
            expected_catalog_count=1,
        )
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=research_v2/src python -m pytest -q research_v2/tests/unit/quality/test_latest.py
```

Expected: import error because `research_v2.quality.latest` is absent.

- [ ] **Step 3: Implement typed construction**

Define `QUALITY_LATEST_COLUMNS`, `FORBIDDEN_LINEAGE_COLUMNS`,
`build_quality_latest_frame`, `validate_quality_latest_frame`, and
`select_model_eligible`. The public builder takes `catalog`, `scoreboard`,
`alpha95`, `alpha484`, `effective_date`, and `evidence_commit`; the validator
takes `frame` and `expected_catalog_count`; the reader takes only `frame`.

Begin every catalog row as `UNSCREENED`, model-ineligible, and with
`source_factor_id=factor_id`. Override only formal-scoreboard rows: status
`PASS` becomes `APPROVED`; `PASS_WITH_MONITORING` becomes
`APPROVED_MONITOR`; both set `model_eligible=True`. Add all three Alpha95 IDs
as `REJECT` using the actual typed quality metrics. Add the Alpha484 candidate
as `PENDING_EXPLICIT_ADMISSION` unless the canonical review reports a rejection.

`validate_quality_latest_frame` requires exact column order, unique IDs,
expected catalog count plus four isolated candidates, nine eligible rows,
SHSZ / `forward_open_5d:v1` on every eligible row, and finite metrics for
every `APPROVED`, `APPROVED_MONITOR`, `REJECT`, or pending candidate row.
`select_model_eligible` validates the scope and returns factor-ID-sorted rows;
it must not create a `FactorSet`.

- [ ] **Step 4: Verify GREEN and commit**

Run:

```bash
PYTHONPATH=research_v2/src python -m pytest -q research_v2/tests/unit/quality/test_latest.py
ruff check research_v2/src/research_v2/quality/latest.py research_v2/tests/unit/quality/test_latest.py
git add research_v2/src/research_v2/quality/latest.py research_v2/tests/unit/quality/test_latest.py
git commit -m "build factor quality latest rows"
```

Expected: unit tests pass and Ruff is clean.

### Task 3: Add a Dry-Run-First Refresh and Reader CLI

**Files:**
- Create: `research_v2/scripts/refresh_factor_quality_latest.py`
- Create: `research_v2/tests/unit/scripts/test_refresh_factor_quality_latest.py`

- [ ] **Step 1: Write failing dry-run and publish tests**

```python
def test_refresh_dry_run_does_not_upload_or_execute() -> None:
    session = RecordingSession()
    assert refresh.refresh_latest(session, pd.DataFrame([valid_row()]), publish=False) == "dry-run"
    assert session.uploads == []
    assert session.executed == []


def test_refresh_publish_replaces_only_compact_target() -> None:
    session = RecordingSession()
    refresh.refresh_latest(session, pd.DataFrame([valid_row()]), publish=True)
    script = "\n".join(session.executed)
    assert "factor_quality_latest" in script
    assert "factor_quality_snapshot" not in script
    assert "delete from loadTable" in script
    assert "tableInsert" in script
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=research_v2/scripts:research_v2/src python -m pytest -q research_v2/tests/unit/scripts/test_refresh_factor_quality_latest.py
```

Expected: import error because the refresh script is absent.

- [ ] **Step 3: Implement refresh and reader behavior**

Implement:

```python
def refresh_latest(session: V2WriteSession, frame: pd.DataFrame, *, publish: bool) -> str:
    validated = validate_quality_latest_frame(frame, expected_catalog_count=487)
    if not publish:
        return "dry-run"
    # Upload `validated`, delete only factor_quality_latest, insert typed rows,
    # undefine staging variables, then query count, uniqueness, and eligibility.
```

The command reads catalog metadata with `YesLabReadOnlySession`, reads only
local scoreboard / Alpha95 / Alpha484 JSON artifacts, and accepts these flags:

```text
--effective-date YYYY-MM-DD
--evidence-commit <40-hex-sha>
--alpha484-review <path>
--publish
--model-eligible
```

Without `--publish`, it prints a JSON dry-run summary and makes no upload or
execute call. `--model-eligible` queries and prints only typed eligible rows;
it never mutates the database. On a publish path, post-write checks must reject
anything other than 491 rows, 491 unique IDs, and nine eligible rows.

- [ ] **Step 4: Verify GREEN and commit**

Run:

```bash
PYTHONPATH=research_v2/scripts:research_v2/src python -m pytest -q research_v2/tests/unit/scripts/test_refresh_factor_quality_latest.py
ruff check research_v2/scripts/refresh_factor_quality_latest.py research_v2/tests/unit/scripts/test_refresh_factor_quality_latest.py
git add research_v2/scripts/refresh_factor_quality_latest.py research_v2/tests/unit/scripts/test_refresh_factor_quality_latest.py
git commit -m "refresh factor quality latest registry"
```

Expected: tests pass and Ruff is clean.

### Task 4: Apply, Populate, and Read Back the New Table

**Files:**
- Create: `research_v2/artifacts/diagnostics/factor_quality_latest_refresh_v1.json`
- Modify: `research_v2/artifacts/diagnostics/tushare_shsz_factor_audit_coverage_v1.json`

- [ ] **Step 1: Verify migration dry-run and refresh dry-run**

Run this exact migration dry-run and assert that version 12 is planned or is a
matching-checksum skip. Then run the refresh without `--publish` using the
completed Alpha484 review:

```python
runner = MigrationRunner(session, migration_directory=Path("research_v2/migrations"))
result = runner.apply(dry_run=True)
assert any(item.version == 12 for item in (*result.planned, *result.skipped))
assert result.executed == ()
```

Run:

```bash
source /data/share/miniconda3/etc/profile.d/conda.sh
conda activate quantyd
set -a
source .env
set +a
PYTHONPATH=research_v2/scripts:research_v2/src python research_v2/scripts/refresh_factor_quality_latest.py --effective-date 2026-08-14 --evidence-commit "$(git rev-parse HEAD)" --alpha484-review research_v2/artifacts/diagnostics/tushare_alpha484_canonical_v1/admission_review.json
```

Expected: a 491-row dry-run with exactly nine eligible formal factors and no
database write.

- [ ] **Step 2: Apply only migration 012 and publish validated rows**

Apply migrations with `MigrationRunner` only after full tests are green, then
repeat the exact refresh command with `--publish`. The only new business table
that may be mutated is `factor_quality_latest`; migration bookkeeping remains
the existing migration runner behavior.

- [ ] **Step 3: Verify database result and record audit evidence**

Read the target and assert:

```text
count(*) = 491
count(distinct factor_id) = 491
sum(model_eligible) = 9
Alpha95 reject count = 3 with model_eligible = false
Alpha484 candidate count = 1 with model_eligible = false
```

Write decision counts, typed-row counts, and safety flags to
`factor_quality_latest_refresh_v1.json`. Add this evidence path to the quality
ledger without changing `formal_admission_factor_ids`, training status, or the
locked-test state.

- [ ] **Step 4: Run final verification and commit**

Run:

```bash
python -m json.tool research_v2/artifacts/diagnostics/tushare_shsz_factor_audit_coverage_v1.json >/dev/null
git diff --check
PYTHONPATH=research_v2/scripts:research_v2/src python -m pytest -q
ruff check research_v2/src/research_v2/quality/latest.py research_v2/scripts/refresh_factor_quality_latest.py research_v2/tests/unit/quality/test_latest.py research_v2/tests/unit/scripts/test_refresh_factor_quality_latest.py
git add -f research_v2/artifacts/diagnostics/factor_quality_latest_refresh_v1.json
git add research_v2/migrations/012_factor_quality_latest.dos research_v2/src/research_v2/quality/latest.py research_v2/scripts/refresh_factor_quality_latest.py research_v2/tests/unit/db/test_migrations.py research_v2/tests/unit/quality/test_latest.py research_v2/tests/unit/scripts/test_refresh_factor_quality_latest.py research_v2/artifacts/diagnostics/tushare_shsz_factor_audit_coverage_v1.json
git commit -m "publish factor quality latest registry"
```

Expected: all tests pass, the formal nine remains unchanged, no model has been
trained, and no locked 2026 data has been read.
