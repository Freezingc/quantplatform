"""Latest-only compact-table publication and read helpers.

The existing V2 publisher remains responsible for immutable, lineage-rich source
tables.  This module deliberately uses a separate publication path: compact rows
are appended to a new ``*_latest`` table and become visible only through the
``latest_manifest`` batch pointer.
"""

from __future__ import annotations

import gc
import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, Protocol

import pandas as pd

from research_v2.compact.contracts import CompactManifestStatus, LatestManifestEntry
from research_v2.compact.materializer import (
    CompactMaterialization,
    build_compact_materialization,
    compact_business_columns,
)

_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_DATE_LITERAL = re.compile(r"^[0-9]{4}-[0-9]{2}$")
_FACTOR_ID = re.compile(r"^[A-Za-z0-9_.:-]+$")


class CompactPublishError(ValueError):
    """Raised when a compact batch cannot be published safely."""


class CompactPublishSession(Protocol):
    """Minimal database capability needed by compact publication."""

    @property
    def target_database(self) -> str: ...

    def upload_frame(self, name: str, frame: pd.DataFrame) -> None: ...

    def execute(self, script: str) -> object: ...

    def query(self, script: str) -> object: ...


@dataclass(frozen=True)
class CompactObjectSpec:
    """Source/target contract for one compact object."""

    source_object: str
    target_object: str
    business_keys: tuple[str, ...]
    partition_date_column: str

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.source_object):
            raise ValueError("source_object must be a simple table identifier")
        if not _IDENTIFIER.fullmatch(self.target_object):
            raise ValueError("target_object must be a simple table identifier")
        if self.target_object != f"{self.source_object}_latest":
            raise ValueError("target_object must be the source object's *_latest table")
        allowed = compact_business_columns(self.source_object)
        if not self.business_keys or not set(self.business_keys).issubset(allowed):
            raise ValueError("business_keys must be allowlisted source business columns")
        if self.partition_date_column not in allowed:
            raise ValueError("partition_date_column must be an allowlisted source column")


DEFAULT_COMPACT_OBJECTS: Final[tuple[CompactObjectSpec, ...]] = (
    CompactObjectSpec(
        source_object="tradability_daily",
        target_object="tradability_daily_latest",
        business_keys=("trade_date", "ts_code"),
        partition_date_column="trade_date",
    ),
    CompactObjectSpec(
        source_object="universe_daily",
        target_object="universe_daily_latest",
        business_keys=("signal_date", "universe_id", "universe_version", "ts_code"),
        partition_date_column="signal_date",
    ),
    CompactObjectSpec(
        source_object="label_daily",
        target_object="label_daily_latest",
        business_keys=("signal_date", "ts_code", "label_id", "label_version"),
        partition_date_column="signal_date",
    ),
    CompactObjectSpec(
        source_object="factor_observation_clean",
        target_object="factor_observation_clean_latest",
        business_keys=("signal_date", "ts_code", "factor_id", "factor_version"),
        partition_date_column="signal_date",
    ),
)


@dataclass(frozen=True)
class SourcePointer:
    """Current source version for one logical partition."""

    source_object: str
    logical_partition: str
    data_version: str
    published_at: pd.Timestamp


@dataclass(frozen=True)
class CompactPublishResult:
    """Outcome of one compact publication attempt."""

    object_name: str
    logical_partition: str
    batch_id: int
    row_count: int
    status: str


def list_current_source_pointers(
    session: CompactPublishSession,
    source_object: str,
) -> tuple[SourcePointer, ...]:
    """Resolve the newest immutable source version for every source partition."""
    _require_identifier(source_object, "source_object")
    result = session.query(
        "select object_name, logical_partition, data_version, published_at "
        "from loadTable("
        f"'{session.target_database}', 'published_version') "
        f"where object_name=`{source_object} order by published_at desc"
    )
    if not isinstance(result, pd.DataFrame):
        raise CompactPublishError("published-version query did not return a DataFrame")
    required = {"object_name", "logical_partition", "data_version", "published_at"}
    if not required.issubset(result.columns):
        raise CompactPublishError("published-version query returned an invalid schema")
    if result.empty:
        return ()
    working = result.loc[:, sorted(required)].copy()
    working["logical_partition"] = working["logical_partition"].astype(str)
    working["data_version"] = working["data_version"].astype(str)
    working["published_at"] = pd.to_datetime(working["published_at"], errors="raise")
    working = working.sort_values("published_at", ascending=False, kind="mergesort")
    working = working.drop_duplicates("logical_partition", keep="first")
    pointers = [
        SourcePointer(
            source_object=source_object,
            logical_partition=str(row.logical_partition),
            data_version=str(row.data_version),
            published_at=pd.Timestamp(str(row.published_at)),
        )
        for row in working.itertuples(index=False)
    ]
    return tuple(sorted(pointers, key=lambda pointer: pointer.logical_partition))


def fetch_source_partition(
    session: CompactPublishSession,
    spec: CompactObjectSpec,
    pointer: SourcePointer,
) -> pd.DataFrame:
    """Read one source version and calendar-month partition."""
    if pointer.source_object != spec.source_object:
        raise CompactPublishError("source pointer does not match object specification")
    if not _DATE_LITERAL.fullmatch(pointer.logical_partition):
        raise CompactPublishError("source logical partition must use YYYY-MM")
    start, end = _month_bounds(pointer.logical_partition)
    selected_columns = (*compact_business_columns(spec.source_object), "data_version")
    selected = ", ".join(selected_columns)
    table_reference = (
        "loadTable(" f"'{session.target_database}', '{spec.source_object}')"
    )
    version_filter = f"data_version='{_quoted(pointer.data_version)}'"

    # Factor observations can contain millions of rows in one calendar month.
    # Daily reads keep the DolphinDB query working set bounded while retaining
    # exactly the same source-version and business-column contract.
    if spec.source_object == "factor_observation_clean":
        daily_frames: list[pd.DataFrame] = []
        for current_day in pd.date_range(start.replace(".", "-"), end.replace(".", "-"), freq="D"):
            day = current_day.strftime("%Y.%m.%d")
            query = (
                f"select {selected} from {table_reference} "
                f"where {version_filter} and {spec.partition_date_column}=date('{day}')"
            )
            daily = session.query(query)
            if not isinstance(daily, pd.DataFrame):
                raise CompactPublishError("source partition query did not return a DataFrame")
            if not daily.empty:
                daily_frames.append(daily.copy(deep=True))
        if not daily_frames:
            return pd.DataFrame(columns=selected_columns)
        return pd.concat(daily_frames, ignore_index=True)

    query = (
        f"select {selected} from {table_reference} "
        f"where {version_filter} "
        f"and {spec.partition_date_column} between date('{start}') and date('{end}')"
    )
    result = session.query(query)
    if not isinstance(result, pd.DataFrame):
        raise CompactPublishError("source partition query did not return a DataFrame")
    return result.copy(deep=True)


def publish_compact_materialization(
    session: CompactPublishSession,
    materialization: CompactMaterialization,
) -> CompactPublishResult:
    """Append one compact batch, then publish its manifest pointer.

    A matching published manifest is treated as an idempotent retry.  The old
    compact rows are never deleted or changed, so a failed retry cannot make a
    partially uploaded batch visible.
    """
    manifest = materialization.manifest
    _require_identifier(manifest.object_name, "manifest.object_name")
    current = _current_manifest(session, manifest.object_name, manifest.logical_partition)
    if (
        current is not None
        and current.batch_id == manifest.batch_id
        and current.business_hash == manifest.business_hash
        and current.row_count == manifest.row_count
        and current.status is CompactManifestStatus.PUBLISHED
    ):
        return CompactPublishResult(
            object_name=manifest.object_name,
            logical_partition=manifest.logical_partition,
            batch_id=manifest.batch_id,
            row_count=manifest.row_count,
            status="skipped",
        )

    stage_name = _stage_name(manifest)
    target_types = _target_types(session, manifest.object_name)
    session.upload_frame(stage_name, _uploadable_frame(materialization.frame))
    session.execute(_insert_script(session, manifest.object_name, stage_name, target_types))
    session.execute(_cleanup_stage_script(stage_name, target_types, materialization.frame.columns))
    session.execute(_manifest_script(session, manifest))
    gc.collect()
    return CompactPublishResult(
        object_name=manifest.object_name,
        logical_partition=manifest.logical_partition,
        batch_id=manifest.batch_id,
        row_count=manifest.row_count,
        status="published",
    )


def materialize_and_publish_current_partition(
    session: CompactPublishSession,
    spec: CompactObjectSpec,
    pointer: SourcePointer,
) -> CompactPublishResult:
    """Fetch, validate, compact, and publish one current source partition."""
    source = fetch_source_partition(session, spec, pointer)
    materialization = build_compact_materialization(
        source,
        object_name=spec.source_object,
        source_data_version=pointer.data_version,
        logical_partition=pointer.logical_partition,
        business_columns=compact_business_columns(spec.source_object),
        business_keys=spec.business_keys,
        partition_date_column=spec.partition_date_column,
        published_at=pointer.published_at.to_pydatetime(),
    )
    return publish_compact_materialization(session, materialization)


def backfill_current_compact(
    session: CompactPublishSession,
    *,
    objects: tuple[CompactObjectSpec, ...] = DEFAULT_COMPACT_OBJECTS,
) -> tuple[CompactPublishResult, ...]:
    """Backfill all current source pointers without recomputing source history."""
    results: list[CompactPublishResult] = []
    for spec in objects:
        for pointer in list_current_source_pointers(session, spec.source_object):
            results.append(materialize_and_publish_current_partition(session, spec, pointer))
    return tuple(results)


def read_latest_partition(
    session: CompactPublishSession,
    *,
    object_name: str,
    logical_partition: str,
    columns: tuple[str, ...] = (),
    factor_keys: tuple[tuple[str, int], ...] = (),
) -> pd.DataFrame:
    """Read only the batch selected by ``latest_manifest`` for a partition."""
    _require_identifier(object_name, "object_name")
    if not _DATE_LITERAL.fullmatch(logical_partition):
        raise ValueError("logical_partition must use YYYY-MM")
    if factor_keys and object_name != "factor_observation_clean_latest":
        raise ValueError("factor_keys can only filter factor_observation_clean_latest")
    entry = _current_manifest(session, object_name, logical_partition)
    if entry is None or entry.status is not CompactManifestStatus.PUBLISHED:
        return pd.DataFrame()
    selected = "*" if not columns else ", ".join(_safe_column(column) for column in columns)
    predicate = _factor_key_predicate(factor_keys)
    result = session.query(
        f"select {selected} from loadTable('{session.target_database}', '{object_name}') "
        f"where batch_id={entry.batch_id}{predicate}"
    )
    if not isinstance(result, pd.DataFrame):
        raise CompactPublishError("compact latest query did not return a DataFrame")
    return result.copy(deep=True)


def _current_manifest(
    session: CompactPublishSession,
    object_name: str,
    logical_partition: str,
) -> LatestManifestEntry | None:
    result = session.query(
        "select top 1 object_name, logical_partition, batch_id, source_object, "
        "source_data_version, published_at, row_count, business_hash, status "
        "from loadTable("
        f"'{session.target_database}', 'latest_manifest') "
        f"where object_name=`{object_name} and "
        f"logical_partition='{_quoted(logical_partition)}' order by published_at desc"
    )
    if not isinstance(result, pd.DataFrame):
        raise CompactPublishError("latest manifest query did not return a DataFrame")
    if result.empty:
        return None
    row = result.iloc[0]
    try:
        status = CompactManifestStatus(str(row["status"]))
        return LatestManifestEntry(
            object_name=str(row["object_name"]),
            logical_partition=str(row["logical_partition"]),
            batch_id=int(row["batch_id"]),
            source_object=str(row["source_object"]),
            source_data_version=str(row["source_data_version"]),
            published_at=pd.Timestamp(row["published_at"]).to_pydatetime(),
            row_count=int(row["row_count"]),
            business_hash=str(row["business_hash"]),
            status=status,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CompactPublishError("latest manifest contains invalid values") from error


def _target_types(session: CompactPublishSession, object_name: str) -> dict[str, str]:
    result = session.query(
        "schema(loadTable("
        f"'{session.target_database}', '{object_name}')).colDefs"
    )
    if not isinstance(result, pd.DataFrame) or not {"name", "typeString"}.issubset(result.columns):
        raise CompactPublishError("compact target schema query returned an invalid result")
    return {
        str(row["name"]): str(row["typeString"])
        for row in result.loc[:, ["name", "typeString"]].to_dict("records")
    }


def _insert_script(
    session: CompactPublishSession,
    object_name: str,
    stage_name: str,
    target_types: dict[str, str],
) -> str:
    castable_types = {
        "BOOL": "bool",
        "BYTE": "byte",
        "SHORT": "short",
        "INT": "int",
        "LONG": "long",
        "FLOAT": "float",
        "DOUBLE": "double",
        "DATE": "date",
        "TIMESTAMP": "timestamp",
    }
    columns = list(target_types)
    selected_columns = [
        f"{castable_types[target_types[column]]}({column}) as {column}"
        if target_types.get(column) in castable_types
        else column
        for column in columns
    ]
    typed_stage = f"{stage_name}_typed"
    return (
        f"{typed_stage} = select {', '.join(selected_columns)} from {stage_name}\n"
        "tableInsert{loadTable("
        f"'{session.target_database}', '{object_name}')"
        f"}}({typed_stage})"
    )


def _manifest_script(session: CompactPublishSession, manifest: LatestManifestEntry) -> str:
    return (
        "compactManifest = table("
        f"[`{manifest.object_name}] as object_name, "
        f"['{_quoted(manifest.logical_partition)}'] as logical_partition, "
        f"[{manifest.batch_id}] as batch_id, "
        f"[`{manifest.source_object}] as source_object, "
        f"['{_quoted(manifest.source_data_version)}'] as source_data_version, "
        "[now()] as published_at, "
        f"[{manifest.row_count}] as row_count, "
        f"['{manifest.business_hash}'] as business_hash, "
        "[`PUBLISHED] as status)\n"
        "tableInsert{loadTable("
        f"'{session.target_database}', 'latest_manifest')"
        "}(compactManifest)"
    )


def _cleanup_stage_script(
    stage_name: str,
    target_types: dict[str, str],
    columns: Iterable[str],
) -> str:
    castable_types = {
        "BOOL",
        "BYTE",
        "SHORT",
        "INT",
        "LONG",
        "FLOAT",
        "DOUBLE",
        "DATE",
        "TIMESTAMP",
    }
    has_typed_stage = any(target_types.get(column) in castable_types for column in columns)
    cleanup = [f"undef(`{stage_name}, VAR)"]
    if has_typed_stage:
        cleanup.append(f"undef(`{stage_name}_typed, VAR)")
    return "\n".join(cleanup)


def _stage_name(manifest: LatestManifestEntry) -> str:
    identity = f"{manifest.object_name}|{manifest.logical_partition}|{manifest.batch_id}"
    return f"compact_stage_{hashlib.sha256(identity.encode()).hexdigest()[:16]}"


def _uploadable_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.copy(deep=True)


def _month_bounds(value: str) -> tuple[str, str]:
    period = pd.Period(value, freq="M")
    return period.start_time.strftime("%Y.%m.%d"), period.end_time.strftime("%Y.%m.%d")


def _quoted(value: str) -> str:
    return value.replace("'", "''")


def _require_identifier(value: str, name: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise CompactPublishError(f"{name} must be a simple table identifier")


def _safe_column(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError("columns must be simple identifiers")
    return value


def _factor_key_predicate(factor_keys: tuple[tuple[str, int], ...]) -> str:
    if not factor_keys:
        return ""
    clauses: list[str] = []
    for factor_id, factor_version in factor_keys:
        if not _FACTOR_ID.fullmatch(factor_id):
            raise ValueError("factor IDs contain unsafe characters")
        if factor_version < 1:
            raise ValueError("factor versions must be positive")
        clauses.append(
            f"(factor_id='{_quoted(factor_id)}' and factor_version={int(factor_version)})"
        )
    return " and (" + " or ".join(clauses) + ")"
