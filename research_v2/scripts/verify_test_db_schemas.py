"""Verify live V2Test table schemas against the frozen migration contracts (A6.6).

Read-only: connects to the configured DolphinDB test database, enumerates every
table declared by migrations 001-012, and compares column names and types from
``schema(loadTable(...)).colDefs`` against the DDL declared in the ``.dos``
files.  Migration 013 is reported separately when the live audit table has not
received it yet.  Exits non-zero on any schema drift.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import dolphindb as ddb

from research_v2.db.migrations import discover_migrations

_TABLE_DDL = re.compile(
    r"(?P<variable>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*table\(\s*1:0\s*,\s*"
    r"`(?P<columns>[A-Za-z_][A-Za-z0-9_`]*),\s*\[(?P<types>[^\]]*)\]",
    re.DOTALL,
)
_TABLE_NAME = re.compile(
    r"createPartitionedTable\(\s*table=(?P<variable>[A-Za-z_][A-Za-z0-9_]*),"
    r"\s*tableName=`(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)


def parse_expected_schemas(migration_directory: Path) -> dict[str, list[tuple[str, str]]]:
    expected: dict[str, list[tuple[str, str]]] = {}
    for migration in discover_migrations(migration_directory):
        variables: dict[str, list[tuple[str, str]]] = {}
        for match in _TABLE_DDL.finditer(migration.script):
            columns = [
                column for column in match.group("columns").split("`") if column
            ]
            types = [
                item.strip()
                for item in match.group("types").split(",")
                if item.strip()
            ]
            if len(columns) != len(types):
                raise ValueError(
                    f"migration {migration.version:03d} table variable "
                    f"{match.group('variable')} has {len(columns)} columns "
                    f"but {len(types)} types"
                )
            variables[match.group("variable")] = list(zip(columns, types, strict=True))
        for match in _TABLE_NAME.finditer(migration.script):
            variable = match.group("variable")
            if variable not in variables:
                raise ValueError(
                    f"migration {migration.version:03d} creates table "
                    f"{match.group('name')} from unknown variable {variable}"
                )
            expected[match.group("name")] = variables[variable]
    return expected


def main() -> int:
    session = ddb.session()
    connected = session.connect(
        os.getenv("QRV2_DDB_HOST", "127.0.0.1"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", ""),
        os.getenv("QRV2_DDB_PASSWORD", ""),
    )
    if not connected:
        print("cannot connect to the configured DolphinDB test database")
        return 2
    database = os.getenv("QRV2_DDB_TARGET_TEST", "dfs://QuantResearchV2Test")

    migration_directory = Path(__file__).resolve().parents[1] / "migrations"
    expected_schemas = parse_expected_schemas(migration_directory)

    applied = session.run(
        f"select schema_version from loadTable('{database}', 'schema_migration')"
    )
    applied_versions = {int(value) for value in applied["schema_version"]}
    expected_versions = {
        migration.version
        for migration in discover_migrations(migration_directory)
        if migration.version <= max(applied_versions)
    }
    unapplied = sorted(expected_versions - applied_versions)

    drift: list[str] = []
    verified = 0
    for table_name, expected_columns in sorted(expected_schemas.items()):
        # Tables introduced by unapplied migrations are legitimately absent.
        introduced_by = next(
            migration.version
            for migration in discover_migrations(migration_directory)
            if re.search(rf"tableName=`{table_name}\b", migration.script)
        )
        if introduced_by not in applied_versions:
            continue
        exists = session.run(f"existsTable('{database}', '{table_name}')")
        if not exists:
            drift.append(f"{table_name}: missing on live test database")
            continue
        schema = session.run(f"schema(loadTable('{database}', '{table_name}')).colDefs")
        actual = [
            (str(row["name"]), str(row["typeString"]))
            for row in schema.loc[:, ["name", "typeString"]].to_dict("records")
        ]
        if actual != expected_columns:
            drift.append(
                f"{table_name}: schema drift; expected {expected_columns}, "
                f"actual {actual}"
            )
            continue
        verified += 1

    print(f"applied migration versions: {sorted(applied_versions)}")
    if unapplied:
        print(f"local migrations not yet applied: {unapplied}")
    print(f"tables verified against migration contracts: {verified}")
    if drift:
        print("SCHEMA DRIFT DETECTED:")
        for item in drift:
            print(" -", item)
        return 1
    print("no schema drift detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
