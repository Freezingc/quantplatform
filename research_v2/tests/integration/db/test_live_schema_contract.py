"""Live V2Test schema contract evidence (A6.6).

Runs only when QRV2_DDB credentials are configured; verifies the applied
migration audit rows and every declared table schema against the frozen
``.dos`` contracts without mutating the database.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import dolphindb as ddb
import pytest

from research_v2.db.migrations import discover_migrations
from verify_test_db_schemas import parse_expected_schemas

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("QRV2_INTEGRATION") != "1"
        or not os.environ.get("QRV2_DDB_USERNAME")
        or not os.environ.get("QRV2_DDB_PASSWORD"),
        reason="requires QRV2_INTEGRATION=1 and QRV2_DDB credentials",
    ),
]

_MIGRATION_DIRECTORY = Path(__file__).resolve().parents[3] / "migrations"


def _connect():
    session = ddb.session()
    connected = session.connect(
        os.environ.get("QRV2_DDB_HOST", "127.0.0.1"),
        int(os.environ.get("QRV2_DDB_PORT", "8848")),
        os.environ["QRV2_DDB_USERNAME"],
        os.environ["QRV2_DDB_PASSWORD"],
    )
    if not connected:
        pytest.skip("cannot reach the configured DolphinDB test database")
    return session


def test_live_applied_migrations_match_local_checksums() -> None:
    session = _connect()
    database = os.environ.get("QRV2_DDB_TARGET_TEST", "dfs://QuantResearchV2Test")
    local = {m.version: m for m in discover_migrations(_MIGRATION_DIRECTORY)}
    applied = session.run(
        f"select schema_version, name, checksum from loadTable('{database}', 'schema_migration')"
    )

    assert not applied.empty
    for _, row in applied.iterrows():
        version = int(row["schema_version"])
        assert version in local, f"applied version {version} is missing locally"
        assert local[version].name == row["name"]
        assert local[version].checksum == row["checksum"], f"checksum drift at v{version}"
    unapplied = sorted(set(local) - {int(v) for v in applied["schema_version"]})
    # Unapplied local migrations are pending approvals, not drift.
    assert all(version > min(local) for version in unapplied)


def test_live_table_schemas_match_migration_contracts() -> None:
    session = _connect()
    database = os.environ.get("QRV2_DDB_TARGET_TEST", "dfs://QuantResearchV2Test")
    expected_schemas = parse_expected_schemas(_MIGRATION_DIRECTORY)
    applied = session.run(
        f"select schema_version from loadTable('{database}', 'schema_migration')"
    )
    applied_versions = {int(value) for value in applied["schema_version"]}
    migrations = discover_migrations(_MIGRATION_DIRECTORY)

    verified = 0
    for table_name, expected_columns in sorted(expected_schemas.items()):
        introduced_by = next(
            migration.version
            for migration in migrations
            if re.search(rf"tableName=`{table_name}\b", migration.script)
        )
        if introduced_by not in applied_versions:
            continue
        assert bool(session.run(f"existsTable('{database}', '{table_name}')")), (
            f"table missing on live test database: {table_name}"
        )
        schema = session.run(f"schema(loadTable('{database}', '{table_name}')).colDefs")
        actual = [
            (str(row["name"]), str(row["typeString"]))
            for row in schema.loc[:, ["name", "typeString"]].to_dict("records")
        ]
        assert actual == expected_columns, f"schema drift on {table_name}"
        verified += 1
    assert verified >= 40
