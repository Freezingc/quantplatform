"""Protocol-level in-memory stand-in for the V2Test DolphinDB database.

The harness simulates the surface the migration runner and batch publisher
depend on: an allowlisted target database, idempotent table creation, the
``schema_migration`` audit table, staged variable uploads, table inserts, and
the ``published_version`` visibility pointer.  It lets database integration
tests run without a live DolphinDB instance while preserving ordering,
idempotency, checksum drift, and pointer-last visibility rules.
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
_INSERT_TARGET = re.compile(
    r"tableInsert\{loadTable\('(?P<db>[^']+)', '(?P<table>[A-Za-z_][A-Za-z0-9_]*)'\)\}"
    r"\((?P<stage>[A-Za-z_][A-Za-z0-9_]*)\)"
)
_POINTER_FIELD = re.compile(
    r"\[(?P<value>[^\]]*)\] as (?P<field>object_name|logical_partition|data_version"
    r"|previous_data_version)"
)
_ROW_COUNT_QUERY = re.compile(
    r"select count\(\*\) as row_count from loadTable\('(?P<db>[^']+)', "
    r"'(?P<table>[A-Za-z_][A-Za-z0-9_]*)'\) where data_version='(?P<version>(?:[^']|'')*)'"
)
_SCHEMA_QUERY = re.compile(
    r"schema\(loadTable\('(?P<db>[^']+)', '(?P<table>[A-Za-z_][A-Za-z0-9_]*)'\)\)\.colDefs"
)
_PREVIOUS_VERSION_QUERY = re.compile(
    r"select top 1 data_version from loadTable\('(?P<db>[^']+)', 'published_version'\) "
    r"where object_name=`(?P<object>[A-Za-z_][A-Za-z0-9_]*), "
    r"logical_partition='(?P<partition>(?:[^']|'')*)'"
)


def _unquote(raw: str) -> str:
    text = raw.strip()
    if len(text) >= 2 and text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    return text.replace("''", "'")


class FakeV2TestDatabase:
    """In-memory V2 database honouring the migration and publish session protocols."""

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
        self._staged: dict[str, pd.DataFrame] = {}
        self._pointer_sequence = 0

    @property
    def target_database(self) -> str:
        return self._target_database

    def upload_frame(self, name: str, frame: pd.DataFrame) -> None:
        self._staged[name] = frame.copy(deep=True)

    def execute(self, script: str) -> None:
        self.executed_scripts.append(script)
        self._apply_script(script)

    def query(self, script: str) -> object:
        row_count_match = _ROW_COUNT_QUERY.search(script)
        if row_count_match is not None:
            table = self._require_table(row_count_match.group("table"))
            version = _unquote("'" + row_count_match.group("version") + "'")
            if "data_version" in table.columns:
                matched = table.loc[table["data_version"] == version]
            else:
                matched = table.iloc[0:0]
            return pd.DataFrame({"row_count": [len(matched)]})
        schema_match = _SCHEMA_QUERY.search(script)
        if schema_match is not None:
            self._require_table(schema_match.group("table"))
            return pd.DataFrame({"name": pd.Series(dtype="object"),
                                 "typeString": pd.Series(dtype="object")})
        previous_match = _PREVIOUS_VERSION_QUERY.search(script)
        if previous_match is not None:
            pointers = self.tables.get("published_version", pd.DataFrame())
            if pointers.empty:
                return pd.DataFrame({"data_version": pd.Series(dtype="object")})
            partition = _unquote("'" + previous_match.group("partition") + "'")
            matched = pointers.loc[
                (pointers["object_name"] == previous_match.group("object"))
                & (pointers["logical_partition"] == partition)
            ].sort_values("published_at", ascending=False)
            if matched.empty:
                return pd.DataFrame({"data_version": pd.Series(dtype="object")})
            return pd.DataFrame({"data_version": [matched.iloc[0]["data_version"]]})
        if "schema_migration" in script and "loadTable" in script:
            return self.tables["schema_migration"].copy()
        raise ValueError(f"fake V2Test database cannot answer query: {script[:80]!r}")

    def _require_table(self, table_name: str) -> pd.DataFrame:
        if table_name not in self.tables:
            raise ValueError(f"fake V2Test database has no table: {table_name}")
        return self.tables[table_name]

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
            return
        pointer_match = list(_POINTER_FIELD.finditer(script))
        if pointer_match and "published_version" in script and "tableInsert" in script:
            fields = {match.group("field"): match.group("value") for match in pointer_match}
            self._pointer_sequence += 1
            row = {
                "object_name": fields["object_name"].strip("`"),
                "logical_partition": _unquote(fields["logical_partition"]),
                "data_version": _unquote(fields["data_version"]),
                "previous_data_version": _unquote(fields["previous_data_version"]),
                "published_at": self._pointer_sequence,
            }
            pointers = self.tables.setdefault("published_version", pd.DataFrame())
            self.tables["published_version"] = pd.concat(
                [pointers, pd.DataFrame([row])], ignore_index=True
            )
            return
        insert_match = _INSERT_TARGET.search(script)
        if insert_match is not None:
            if insert_match.group("db") != self._target_database:
                raise ValueError("insert targets an unexpected database")
            table_name = insert_match.group("table")
            stage_name = insert_match.group("stage")
            if stage_name.endswith("_typed"):
                stage_name = stage_name[: -len("_typed")]
            if stage_name not in self._staged:
                raise ValueError(f"staged variable is missing: {stage_name}")
            frame = self._staged.pop(stage_name)
            target = self._require_table(table_name)
            self.tables[table_name] = pd.concat([target, frame], ignore_index=True)
            return


@pytest.fixture()
def fake_v2_test_database() -> FakeV2TestDatabase:
    return FakeV2TestDatabase()
