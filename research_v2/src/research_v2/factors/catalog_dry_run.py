"""Schema-only dry-run of the factor source catalog against the YesLab source tables.

C1.5 asks for a per-entry check that the registered table and columns exist, that their source
types can actually be read the way the reader reads them, and that the date and code keys are
unambiguous -- plus counts of ``mapped/blocked/retired`` and the source columns no entry maps.

Two constraints shape the design:

* **It must not read values.** The only query this module can emit is
  ``schema(loadTable('dfs://YesLab', '<table>')).colDefs``. A governance check that pulls the
  source table is not a dry-run, and the YesLab factor tables hold the full history of 487
  factors. Column metadata is enough for every check C1.5 names.
* **It must not probe arbitrary tables.** Table names are matched against the frozen
  ``factor_data1..11`` set *before* any session call, so a mistyped catalog entry cannot be used
  as a way to query something else on the source cluster.

The checks themselves are pure: ``run_catalog_dry_run`` takes an already-collected schema map,
which keeps the whole thing testable offline and makes the drift cases -- the reason this exists
at all -- reachable without staging a database.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

_SOURCE_DATABASE = "dfs://YesLab"
_SAFE_TABLE = re.compile(r"factor_data(?:[1-9]|10|11)")
_TYPE_TOKEN = re.compile(r"[A-Z0-9_]+")

#: Columns the reader filters on to find one factor inside a long-form source table.
#: No catalog entry registers this as a column -- entries carry the factor *name* as a value --
#: but every read path depends on it, so its absence makes the table unreadable.
DEFAULT_DISCRIMINATOR_COLUMN = "factor_name"

_REQUIRED_ENTRY_FIELDS = (
    "factor_id",
    "factor_version",
    "source_table",
    "source_value_column",
    "source_date_column",
    "source_code_column",
)

_TEMPORAL_TYPES = frozenset(
    {
        "DATE",
        "DATETIME",
        "TIMESTAMP",
        "MONTH",
        "MINUTE",
        "NANODATE",
        "NANODATETIME",
        "NANOMINUTE",
        "NANOTIMESTAMP",
    }
)
_TEXT_TYPES = frozenset({"SYMBOL", "STRING", "CHAR"})
_NUMERIC_TYPES = frozenset(
    {
        "DOUBLE",
        "FLOAT",
        "INT",
        "LONG",
        "SHORT",
        "BYTE",
        "NUMBER",
        "DECIMAL",
        "DECIMAL32",
        "DECIMAL64",
        "DECIMAL128",
    }
)

MAPPED = "MAPPED"
BLOCKED = "BLOCKED"
RETIRED = "RETIRED"


class CatalogDryRunError(ValueError):
    """Raised when the dry-run cannot be performed at all, as opposed to a factor failing it."""


@dataclass(frozen=True)
class SourceColumn:
    """One column of a source table, as reported by ``schema(...).colDefs``."""

    name: str
    type_string: str

    @property
    def normalized_type(self) -> str:
        """DolphinDB reports parameterized types (``DECIMAL32(S)``); only the head matters."""
        match = _TYPE_TOKEN.match(self.type_string.strip().upper())
        return match.group(0) if match else self.type_string.strip().upper()


class SchemaQuerySession(Protocol):
    """The read-only session subset this module needs."""

    def query(self, script: str) -> object: ...


@dataclass(frozen=True)
class SourceSchemaSet(Mapping[str, tuple[SourceColumn, ...] | None]):
    """Collected schemas keyed by table. ``None`` means the schema query itself failed.

    Behaves as a mapping so it can be handed straight to :func:`run_catalog_dry_run`, and carries
    the failure text alongside -- an outage must be distinguishable from schema drift, or a dead
    connection reads as 487 blocked factors.
    """

    schemas: Mapping[str, tuple[SourceColumn, ...] | None]
    errors: Mapping[str, str] = field(default_factory=dict)

    def __getitem__(self, key: str) -> tuple[SourceColumn, ...] | None:
        return self.schemas[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.schemas)

    def __len__(self) -> int:
        return len(self.schemas)


@dataclass(frozen=True)
class FactorDryRunFinding:
    """The verdict for one catalog entry, with every reason rather than only the first."""

    factor_id: str
    factor_version: int
    source_table: str
    status: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CatalogDryRunReport:
    """Dry-run output: per-entry verdicts, unmapped source columns, and schema fingerprints."""

    findings: tuple[FactorDryRunFinding, ...]
    unmapped_source_columns: Mapping[str, tuple[str, ...]]
    schema_fingerprints: Mapping[str, str]

    @property
    def mapped_count(self) -> int:
        return sum(finding.status == MAPPED for finding in self.findings)

    @property
    def blocked_count(self) -> int:
        return sum(finding.status == BLOCKED for finding in self.findings)

    @property
    def retired_count(self) -> int:
        return sum(finding.status == RETIRED for finding in self.findings)

    @property
    def blocked_findings(self) -> tuple[FactorDryRunFinding, ...]:
        return tuple(finding for finding in self.findings if finding.status == BLOCKED)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "factor_id": finding.factor_id,
                    "factor_version": finding.factor_version,
                    "source_table": finding.source_table,
                    "status": finding.status,
                    "reasons": "; ".join(finding.reasons),
                }
                for finding in self.findings
            ]
        )


def _schema_query(table: str) -> str:
    return f"schema(loadTable('{_SOURCE_DATABASE}', '{table}')).colDefs"


def _assert_safe_tables(tables: Sequence[str]) -> None:
    unsafe = sorted({table for table in tables if not _SAFE_TABLE.fullmatch(table)})
    if unsafe:
        raise CatalogDryRunError(
            "refusing to query source tables outside the frozen factor_data1..11 set: "
            + ", ".join(unsafe)
        )


def collect_source_schemas(
    session: SchemaQuerySession,
    tables: Sequence[str],
) -> SourceSchemaSet:
    """Read column metadata for ``tables``, issuing schema queries and nothing else.

    Every table name is validated before the first query, so an unsafe name costs nothing on the
    source cluster. A table whose query raises is recorded as ``None`` with its error text rather
    than aborting the run: the dry-run's job is to report, and one unreadable table should not
    hide the state of the other ten.
    """
    requested = sorted(set(tables))
    _assert_safe_tables(requested)
    schemas: dict[str, tuple[SourceColumn, ...] | None] = {}
    errors: dict[str, str] = {}
    for table in requested:
        try:
            result = session.query(_schema_query(table))
        except Exception as exc:  # noqa: BLE001 - reported as a finding, never raised
            schemas[table] = None
            errors[table] = f"{type(exc).__name__}: {exc}"
            continue
        if not isinstance(result, pd.DataFrame):
            schemas[table] = None
            errors[table] = "TypeError: schema query did not return a DataFrame"
            continue
        if not {"name", "typeString"}.issubset(result.columns):
            schemas[table] = None
            errors[table] = "schema query returned no name/typeString columns"
            continue
        schemas[table] = tuple(
            SourceColumn(name=str(row["name"]), type_string=str(row["typeString"]))
            for row in result.to_dict(orient="records")
        )
    return SourceSchemaSet(schemas=schemas, errors=errors)


def schema_fingerprint(columns: Sequence[SourceColumn]) -> str:
    """Hash ``name``/``typeString`` the same way :mod:`research_v2.sources.yeslab` does.

    Matching that canonicalization on purpose: a dry-run fingerprint can then be compared
    directly against a ``SourceBatch.schema_fingerprint`` to attribute drift to an extract.
    """
    frame = pd.DataFrame(
        [{"name": column.name, "typeString": column.type_string} for column in columns]
    )
    canonical = frame.loc[:, ["name", "typeString"]].sort_values("name").to_csv(index=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def schema_set_to_payload(
    schemas: Mapping[str, tuple[SourceColumn, ...] | None],
    errors: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Serialize a collected schema set so a verdict can be replayed without the cluster.

    This is the single definition of that on-disk shape. Both the audit script that writes it and
    any consumer that reads it back go through here, because two independent encoders of one format
    is how a saved artifact stops loading -- and it stops loading later, at the moment someone tries
    to re-examine a verdict rather than at the moment the format changed.
    """
    return {
        "schemas": {
            table: (
                None
                if columns is None
                else [
                    {"name": column.name, "typeString": column.type_string}
                    for column in columns
                ]
            )
            for table, columns in sorted(schemas.items())
        },
        "errors": dict(sorted((errors or {}).items())),
    }


def schema_set_from_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, tuple[SourceColumn, ...] | None], dict[str, str]]:
    """Inverse of :func:`schema_set_to_payload`."""
    schemas: dict[str, tuple[SourceColumn, ...] | None] = {}
    for table, columns in (payload.get("schemas") or {}).items():
        schemas[str(table)] = (
            None
            if columns is None
            else tuple(
                SourceColumn(name=str(column["name"]), type_string=str(column["typeString"]))
                for column in columns
            )
        )
    errors = {str(key): str(value) for key, value in (payload.get("errors") or {}).items()}
    return schemas, errors


def _role_columns(entry: Mapping[str, Any], discriminator: str) -> dict[str, str]:
    """The columns one entry consumes, keyed by the role each plays in the read path."""
    return {
        "date key": str(entry.get("source_date_column", "")),
        "security key": str(entry.get("source_code_column", "")),
        "value": str(entry.get("source_value_column", "")),
        "factor discriminator": discriminator,
    }


def registered_source_columns(
    entries: Sequence[Mapping[str, Any]],
    source_table: str,
    *,
    discriminator_column: str = DEFAULT_DISCRIMINATOR_COLUMN,
    retired: frozenset[tuple[str, int]] = frozenset(),
) -> tuple[str, ...]:
    """Columns the pipeline still consumes from ``source_table``, sorted.

    Retired entries are excluded: retirement is the deliberate end of consumption, so a column
    only a retired entry used is genuinely unmapped afterwards and should be reported as such.
    """
    referenced: set[str] = set()
    for entry in entries:
        if _entry_key(entry) in retired:
            continue
        if str(entry.get("source_table", "")) != source_table:
            continue
        referenced.update(
            column for column in _role_columns(entry, discriminator_column).values() if column
        )
    return tuple(sorted(referenced))


def _entry_key(entry: Mapping[str, Any]) -> tuple[str, int]:
    return (str(entry.get("factor_id", "")), int(entry.get("factor_version", 0)))


_EXPECTED_TYPE_CLASS = {
    "date key": ("a temporal type", _TEMPORAL_TYPES),
    "security key": ("a symbol/text type", _TEXT_TYPES),
    "value": ("a numeric type", _NUMERIC_TYPES),
    "factor discriminator": ("a symbol/text type", _TEXT_TYPES),
}


def _check_entry(
    entry: Mapping[str, Any],
    schemas: Mapping[str, tuple[SourceColumn, ...] | None],
    schema_errors: Mapping[str, str],
    discriminator: str,
) -> FactorDryRunFinding:
    factor_id = str(entry.get("factor_id", ""))
    try:
        factor_version = int(entry.get("factor_version", 0))
    except (TypeError, ValueError):
        factor_version = 0
    source_table = str(entry.get("source_table", ""))

    reasons: list[str] = []
    for required in _REQUIRED_ENTRY_FIELDS:
        if required not in entry or entry[required] in (None, ""):
            reasons.append(f"catalog entry is missing required field: {required}")

    if not _SAFE_TABLE.fullmatch(source_table):
        reasons.append(
            f"source table is outside the frozen factor_data1..11 set: {source_table}"
        )
    elif source_table not in schemas:
        reasons.append(
            f"source table was not present in the dry-run schema set: {source_table}"
        )
    elif (columns := schemas[source_table]) is None:
        detail = schema_errors.get(source_table, "unknown schema query failure")
        reasons.append(f"source table schema could not be read: {detail}")
    else:
        reasons.extend(_check_table(entry, source_table, columns, discriminator))

    status = BLOCKED if reasons else MAPPED
    return FactorDryRunFinding(
        factor_id=factor_id,
        factor_version=factor_version,
        source_table=source_table,
        status=status,
        reasons=tuple(reasons),
    )


def _check_table(
    entry: Mapping[str, Any],
    source_table: str,
    columns: Sequence[SourceColumn],
    discriminator: str,
) -> list[str]:
    reasons: list[str] = []
    by_name: dict[str, list[str]] = {}
    for source_column in columns:
        by_name.setdefault(source_column.name, []).append(source_column.normalized_type)
    for name, types in sorted(by_name.items()):
        if len(types) > 1:
            reasons.append(
                f"source table {source_table} declares column more than once: {name}"
            )
    present = {name: types[0] for name, types in by_name.items()}

    roles = _role_columns(entry, discriminator)

    # A role must name one column and one column must serve one role, or the read path has no
    # single answer to "which column is the date key".
    by_column: dict[str, list[str]] = {}
    for role, column_name in roles.items():
        if column_name:
            by_column.setdefault(column_name, []).append(role)
    for column_name, role_list in sorted(by_column.items()):
        if len(role_list) > 1:
            reasons.append(
                f"catalog entry is ambiguous: source column '{column_name}' is registered for "
                f"more than one role ({', '.join(sorted(role_list))})"
            )

    for role, column_name in sorted(roles.items()):
        if not column_name:
            continue
        if column_name not in present:
            if role == "factor discriminator":
                reasons.append(
                    f"source table {source_table} is missing the factor discriminator column: "
                    f"{column_name}"
                )
            else:
                reasons.append(
                    f"source table {source_table} is missing the registered {role} column: "
                    f"{column_name}"
                )
            continue
        expected_label, allowed = _EXPECTED_TYPE_CLASS[role]
        actual = present[column_name]
        if actual not in allowed:
            reasons.append(
                f"registered {role} column '{column_name}' has source type {actual}, which cannot "
                f"be read as {expected_label}"
            )
    return reasons


def run_catalog_dry_run(
    entries: Sequence[Mapping[str, Any]],
    schemas: Mapping[str, tuple[SourceColumn, ...] | None],
    *,
    retired: frozenset[tuple[str, int]] = frozenset(),
    schema_errors: Mapping[str, str] | None = None,
    discriminator_column: str = DEFAULT_DISCRIMINATOR_COLUMN,
) -> CatalogDryRunReport:
    """Classify every catalog entry against the collected source schemas.

    Retired entries short-circuit to ``RETIRED`` and are not re-checked: a factor that is
    intentionally out of the set should not be reported as a schema blocker when its old table
    drifts, which would otherwise put retirement and drift in the same bucket and hide real drift.
    """
    errors: Mapping[str, str] = (
        schema_errors if schema_errors is not None else getattr(schemas, "errors", {})
    )

    findings: list[FactorDryRunFinding] = []
    for entry in sorted(entries, key=_entry_key):
        key = _entry_key(entry)
        if key in retired:
            findings.append(
                FactorDryRunFinding(
                    factor_id=key[0],
                    factor_version=key[1],
                    source_table=str(entry.get("source_table", "")),
                    status=RETIRED,
                    reasons=(),
                )
            )
            continue
        findings.append(_check_entry(entry, schemas, errors, discriminator_column))

    unmapped: dict[str, tuple[str, ...]] = {}
    fingerprints: dict[str, str] = {}
    for table in sorted(schemas):
        columns = schemas[table]
        if columns is None:
            continue
        fingerprints[table] = schema_fingerprint(columns)
        referenced = set(
            registered_source_columns(
                entries, table, discriminator_column=discriminator_column, retired=retired
            )
        )
        unmapped[table] = tuple(
            sorted({column.name for column in columns}.difference(referenced))
        )

    return CatalogDryRunReport(
        findings=tuple(findings),
        unmapped_source_columns=unmapped,
        schema_fingerprints=fingerprints,
    )


def catalog_tables(entries: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Distinct, safe source tables an entry set would need schemas for."""
    tables = sorted({str(entry.get("source_table", "")) for entry in entries})
    _assert_safe_tables([table for table in tables if table])
    return tuple(table for table in tables if table)
