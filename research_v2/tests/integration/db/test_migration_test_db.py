"""Migration runner integration evidence against the in-memory Test DB (A5.4/A5.5)."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from research_v2.db.migrations import MigrationError, MigrationRunner, discover_migrations

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("QRV2_INTEGRATION") != "1",
        reason="set QRV2_INTEGRATION=1 to run database integration tests",
    ),
]

_MIGRATION_DIRECTORY = Path(__file__).resolve().parents[3] / "migrations"
_TABLE_NAME = re.compile(r"tableName=`(?P<name>[A-Za-z_][A-Za-z0-9_]*)")


def _expected_tables() -> set[str]:
    expected: set[str] = set()
    for migration in discover_migrations(_MIGRATION_DIRECTORY):
        expected.update(_TABLE_NAME.findall(migration.script))
    return expected


def test_test_db_applies_all_migrations_in_order(fake_v2_test_database) -> None:
    runner = MigrationRunner(fake_v2_test_database, migration_directory=_MIGRATION_DIRECTORY)

    result = runner.apply()

    planned = discover_migrations(_MIGRATION_DIRECTORY)
    assert [migration.version for migration in result.executed] == [
        migration.version for migration in planned
    ]
    assert len(planned) >= 13
    applied = fake_v2_test_database.tables["schema_migration"]
    assert applied["schema_version"].tolist() == [
        migration.version for migration in planned
    ]
    assert applied["checksum"].tolist() == [migration.checksum for migration in planned]
    created = set(fake_v2_test_database.tables) - {"schema_migration"}
    assert _expected_tables() - {"schema_migration"} <= created


def test_second_apply_is_idempotent_on_test_db(fake_v2_test_database) -> None:
    runner = MigrationRunner(fake_v2_test_database, migration_directory=_MIGRATION_DIRECTORY)
    first = runner.apply()

    second = runner.apply()

    assert len(first.executed) >= 13
    assert second.executed == ()
    assert [migration.version for migration in second.skipped] == [
        migration.version for migration in second.planned
    ]
    assert len(fake_v2_test_database.tables["schema_migration"]) == len(second.planned)


def test_checksum_drift_is_rejected_on_test_db(
    fake_v2_test_database, tmp_path
) -> None:
    runner = MigrationRunner(fake_v2_test_database, migration_directory=_MIGRATION_DIRECTORY)
    runner.apply()

    drifted = tmp_path / "migrations"
    drifted.mkdir()
    for path in sorted(_MIGRATION_DIRECTORY.glob("*.dos")):
        (drifted / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    first_migration = sorted(drifted.glob("*.dos"))[0]
    first_migration.write_text(
        first_migration.read_text(encoding="utf-8") + "\n// drifted\n", encoding="utf-8"
    )

    drifted_runner = MigrationRunner(fake_v2_test_database, migration_directory=drifted)
    with pytest.raises(MigrationError, match="checksum changed"):
        drifted_runner.apply()


def test_dry_run_executes_nothing_on_test_db(fake_v2_test_database) -> None:
    runner = MigrationRunner(fake_v2_test_database, migration_directory=_MIGRATION_DIRECTORY)

    result = runner.apply(dry_run=True)

    assert result.executed == ()
    assert fake_v2_test_database.executed_scripts == []
    assert fake_v2_test_database.tables["schema_migration"].empty
    assert len(result.planned) >= 13
