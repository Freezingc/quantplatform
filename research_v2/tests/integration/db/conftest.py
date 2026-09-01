"""Protocol-level in-memory stand-in for the V2Test DolphinDB database.

The harness simulates exactly the surface the migration runner depends on:
an allowlisted target database, script execution with idempotent table
creation, and the ``schema_migration`` audit table used for checksum
verification.  It lets migration integration tests run without a live
DolphinDB instance while preserving ordering, idempotency, and drift rules.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

_TABLE_NAME = re.compile(r"tableName=`(?P<name>[A-Za-z_][A-Za-z0-9_]*)")
_AUDIT_ROW = re.compile(
    r"table\(\[(?P<version>\d+)\] as schema_version, "
    r"\['(?P<name>[a-z0-9_]+)'\] as name, "
    r"\['(?P<checksum>[0-9a-f]{64})'\] as checksum"
)


class FakeV2TestDatabase:
    """In-memory migration execution target honouring the V2 session protocol."""

    def __init__(self, target_database: str = "dfs://QuantResearchV2Test") -> None:
        self._target_database = target_database
        self.tables: dict[str, pd.DataFrame] = {
            "schema_migration": pd.DataFrame(
                {
                    "schema_version": pd.Series(dtype="int64"),
                    "name": pd.Series(dtype="object"),
                    "checksum": pd.Series(dtype="object"),
                }
            )
        }
        self.executed_scripts: list[str] = []

    @property
    def target_database(self) -> str:
        return self._target_database

    def execute(self, script: str) -> None:
        self.executed_scripts.append(script)
        self._apply_script(script)

    def query(self, script: str) -> object:
        if "schema_migration" in script and "loadTable" in script:
            return self.tables["schema_migration"].copy()
        raise ValueError(f"fake V2Test database cannot answer query: {script[:80]!r}")

    def _apply_script(self, script: str) -> None:
        declared_target = re.search(r"targetDatabase = '(?P<db>[^']+)'", script)
        if declared_target is not None and declared_target.group("db") != self._target_database:
            raise ValueError("migration script targets an unexpected database")
        for match in _TABLE_NAME.finditer(script):
            table_name = match.group("name")
            if table_name not in self.tables:
                self.tables[table_name] = pd.DataFrame()
        audit_match = _AUDIT_ROW.search(script)
        if audit_match is not None and "tableInsert" in script:
            row = {
                "schema_version": int(audit_match.group("version")),
                "name": audit_match.group("name"),
                "checksum": audit_match.group("checksum"),
            }
            self.tables["schema_migration"] = pd.concat(
                [self.tables["schema_migration"], pd.DataFrame([row])],
                ignore_index=True,
            )


@pytest.fixture()
def fake_v2_test_database() -> FakeV2TestDatabase:
    return FakeV2TestDatabase()
