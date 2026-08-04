from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from research_v2.db.migrations import MigrationError, MigrationRunner, discover_migrations


class RecordingSession:
    target_database = "dfs://QuantResearchV2Test"

    def __init__(self) -> None:
        self.scripts: list[str] = []
        self.applied = pd.DataFrame(columns=["schema_version", "checksum"])

    def execute(self, script: str) -> object:
        self.scripts.append(script)
        return None

    def query(self, script: str) -> object:
        return self.applied


def _write_migration(path: Path, filename: str, body: str) -> None:
    (path / filename).write_text(body, encoding="utf-8")


def test_discover_migrations_orders_by_numeric_version_and_hashes_content(tmp_path: Path) -> None:
    _write_migration(tmp_path, "010_experiment_tables.dos", "ten")
    _write_migration(tmp_path, "001_control_tables.dos", "one")

    migrations = discover_migrations(tmp_path)

    assert [migration.version for migration in migrations] == [1, 10]
    assert [migration.name for migration in migrations] == [
        "control_tables",
        "experiment_tables",
    ]
    assert migrations[0].checksum != migrations[1].checksum


def test_discover_migrations_rejects_an_invalid_filename(tmp_path: Path) -> None:
    _write_migration(tmp_path, "control_tables.dos", "invalid")

    with pytest.raises(MigrationError, match="NNN_name.dos"):
        discover_migrations(tmp_path)


def test_discover_migrations_rejects_duplicate_versions(tmp_path: Path) -> None:
    _write_migration(tmp_path, "001_control_tables.dos", "first")
    _write_migration(tmp_path, "001_other_tables.dos", "second")

    with pytest.raises(MigrationError, match="duplicate"):
        discover_migrations(tmp_path)


def test_migration_runner_dry_run_never_executes_a_script(tmp_path: Path) -> None:
    _write_migration(tmp_path, "001_control_tables.dos", "answer = 42")
    session = RecordingSession()

    result = MigrationRunner(session, migration_directory=tmp_path).apply(dry_run=True)

    assert [migration.version for migration in result.planned] == [1]
    assert result.executed == ()
    assert session.scripts == []


def test_migration_runner_injects_only_the_approved_target_database(tmp_path: Path) -> None:
    _write_migration(tmp_path, "001_control_tables.dos", "answer = 42")
    session = RecordingSession()

    result = MigrationRunner(session, migration_directory=tmp_path).apply()

    assert [migration.version for migration in result.executed] == [1]
    assert session.scripts[0] == "targetDatabase = 'dfs://QuantResearchV2Test'\nanswer = 42"


def test_migration_runner_skips_an_applied_migration_with_the_same_checksum(
    tmp_path: Path,
) -> None:
    _write_migration(tmp_path, "001_control_tables.dos", "answer = 42")
    session = RecordingSession()
    migration = discover_migrations(tmp_path)[0]
    session.applied = pd.DataFrame(
        [{"schema_version": migration.version, "checksum": migration.checksum}]
    )

    result = MigrationRunner(session, migration_directory=tmp_path).apply()

    assert result.executed == ()
    assert [migration.version for migration in result.skipped] == [1]
    assert session.scripts == []


def test_migration_runner_rejects_checksum_drift_for_an_applied_version(tmp_path: Path) -> None:
    _write_migration(tmp_path, "001_control_tables.dos", "answer = 42")
    session = RecordingSession()
    session.applied = pd.DataFrame([{"schema_version": 1, "checksum": "changed"}])

    with pytest.raises(MigrationError, match="checksum"):
        MigrationRunner(session, migration_directory=tmp_path).apply()

    assert session.scripts == []


def test_migration_runner_records_an_audit_row_only_after_ddl_execution(tmp_path: Path) -> None:
    _write_migration(tmp_path, "001_control_tables.dos", "answer = 42")
    session = RecordingSession()

    MigrationRunner(session, migration_directory=tmp_path).apply()

    assert len(session.scripts) == 2
    assert session.scripts[0].endswith("answer = 42")
    assert "tableInsert" in session.scripts[1]
    assert "schema_version" in session.scripts[1]
    assert "control_tables" in session.scripts[1]


def test_repository_pit_migration_declares_all_required_tables() -> None:
    migration_directory = Path(__file__).resolve().parents[3] / "migrations"
    migrations = discover_migrations(migration_directory)

    pit_migration = next(migration for migration in migrations if migration.version == 2)

    for table_name in (
        "security_master_scd",
        "market_daily_clean",
        "tradability_daily",
        "universe_definition",
        "universe_daily",
    ):
        assert f"`{table_name}" in pit_migration.script


def test_repository_factor_and_label_migration_declares_all_required_tables() -> None:
    migration_directory = Path(__file__).resolve().parents[3] / "migrations"
    migrations = discover_migrations(migration_directory)

    factor_migration = next(migration for migration in migrations if migration.version == 3)

    for table_name in (
        "factor_registry",
        "factor_observation_clean",
        "factor_quarantine",
        "factor_quality_snapshot",
        "factor_similarity",
        "factor_set",
        "factor_set_member",
        "label_daily",
    ):
        assert f"`{table_name}" in factor_migration.script


def test_repository_experiment_migration_declares_all_required_tables() -> None:
    migration_directory = Path(__file__).resolve().parents[3] / "migrations"
    migrations = discover_migrations(migration_directory)

    experiment_migration = next(migration for migration in migrations if migration.version == 4)

    for table_name in (
        "split_definition",
        "dataset_manifest",
        "experiment_run",
        "metric_daily",
        "metric_summary",
        "prediction_daily",
        "model_registry",
        "active_model_pointer",
        "locked_test_audit",
    ):
        assert f"`{table_name}" in experiment_migration.script


def test_repository_quality_migration_declares_market_quarantine_table() -> None:
    migration_directory = Path(__file__).resolve().parents[3] / "migrations"
    migrations = discover_migrations(migration_directory)

    quality_migration = next(migration for migration in migrations if migration.version == 5)

    assert "`market_quarantine" in quality_migration.script


def test_repository_latest_compact_migration_declares_compact_tables_without_row_lineage() -> None:
    migration_directory = Path(__file__).resolve().parents[3] / "migrations"
    migrations = discover_migrations(migration_directory)

    compact_migration = next(migration for migration in migrations if migration.version == 8)

    for table_name in (
        "latest_manifest",
        "tradability_daily_latest",
        "universe_daily_latest",
        "label_daily_latest",
        "factor_observation_clean_latest",
    ):
        assert f"`{table_name}" in compact_migration.script
    assert "`batch_id" in compact_migration.script
    assert "`record_hash" not in compact_migration.script
    assert "`source_watermark_id" not in compact_migration.script


def test_repository_repair_migration_declares_repair_tables() -> None:
    migration_directory = Path(__file__).resolve().parents[3] / "migrations"
    migrations = discover_migrations(migration_directory)

    repair_migration = next(migration for migration in migrations if migration.version == 9)

    for table_name in ("factor_repair_audit", "factor_repair_quarantine"):
        assert f"`{table_name}" in repair_migration.script
    for column_name in (
        "logical_partition",
        "factor_id",
        "factor_version",
        "repair_method",
        "decision_status",
        "filled_rows",
        "conflict_groups",
        "run_id",
        "data_version",
        "source_watermark_id",
        "record_hash",
    ):
        assert f"`{column_name}" in repair_migration.script

    for variable_name in ("factorRepairAudit", "factorRepairQuarantine"):
        declaration = re.search(
            rf"{variable_name} = table\(1:0,\s+([^,]+),\s+\[([^\]]+)\]\)",
            repair_migration.script,
            flags=re.DOTALL,
        )
        assert declaration is not None
        assert declaration.group(1).count("`") == len(
            [item for item in declaration.group(2).split(",") if item.strip()]
        )
