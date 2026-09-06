"""Strict, month-scoped backfill of frozen technical factors from YesLab.

This module fills the gap between a frozen factor set and the current V2 factor
observation table. It never mutates YesLab and publishes one V2 source version
only after every requested factor in the month has produced a clean batch.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Final, Protocol, cast

import pandas as pd

from research_v2.contracts.factor import (
    AvailabilityPolicy,
    FactorDefinition,
    FactorType,
)
from research_v2.db.publish import StagedBatch, V2BatchPublisher
from research_v2.factors.catalog import build_yeslab_catalog
from research_v2.factors.materializer import (
    FactorMaterializationBatches,
    _record_hashes,
    build_technical_factor_batches,
)
from research_v2.factors.repair import RepairOverlayResult
from research_v2.factors.repair_overlay import (
    RepairOverlayBuildResult,
    RepairProvider,
    RepairStatus,
    build_repair_overlay,
)
from research_v2.quality.engine import CheckResult, CheckStatus, Severity
from research_v2.runs.context import RunContext


class FactorBackfillError(ValueError):
    """Raised when a requested frozen factor cannot be materialized safely."""


@dataclass(frozen=True)
class RepairFactorMaterializationBatches(FactorMaterializationBatches):
    """Materialization pair carrying optional in-memory repair evidence."""

    repair_overlay: RepairOverlayBuildResult | None = None
    repair_audit: RepairOverlayResult | None = None


class YesLabFactorSession(Protocol):
    """Read-only source capability required by the factor backfiller."""

    def query(self, script: str) -> object: ...


class FactorBackfillTargetSession(Protocol):
    """V2 write capability consumed by :class:`V2BatchPublisher`."""

    @property
    def target_database(self) -> str: ...

    def upload_frame(self, name: str, frame: pd.DataFrame) -> None: ...

    def execute(self, script: str) -> object: ...

    def query(self, script: str) -> object: ...


_FACTOR_ID = re.compile(r"^yeslab_[0-9]+$")
_SOURCE_COLUMNS = ("ts_code", "trade_date", "factor_name", "value")


@dataclass(frozen=True)
class FactorMonth:
    """Closed calendar-month scope for one factor publication."""

    logical_partition: str
    start_date: date
    end_date: date

    @classmethod
    def from_partition(cls, logical_partition: str) -> FactorMonth:
        try:
            period = pd.Period(logical_partition, freq="M")
        except (TypeError, ValueError) as error:
            raise FactorBackfillError("logical_partition must use YYYY-MM") from error
        if str(period) != logical_partition:
            raise FactorBackfillError("logical_partition must use YYYY-MM")
        return cls(
            logical_partition=logical_partition,
            start_date=period.start_time.date(),
            end_date=period.end_time.date(),
        )


@dataclass(frozen=True)
class FactorBackfillResult:
    """Identity of one successfully published V2 factor source version."""

    logical_partition: str
    data_version: str
    factor_ids: tuple[str, ...]


@dataclass(frozen=True)
class FactorMaterializationAudit:
    """Read-only row-count evidence for one month-scoped factor materialization."""

    logical_partition: str
    data_version: str
    clean_rows_by_factor: tuple[tuple[str, int], ...]
    quarantine_rows_by_factor: tuple[tuple[str, int], ...]
    quarantine_reasons_by_factor: tuple[tuple[str, str, int], ...]

    @property
    def total_clean_rows(self) -> int:
        """Return the number of clean rows across the requested factors."""
        return sum(row_count for _, row_count in self.clean_rows_by_factor)

    @property
    def total_quarantine_rows(self) -> int:
        """Return the number of quarantined groups across the requested factors."""
        return sum(row_count for _, row_count in self.quarantine_rows_by_factor)


def audit_factor_materialization_batches(
    batches: Iterable[FactorMaterializationBatches],
    *,
    factor_ids: tuple[str, ...] = (),
) -> FactorMaterializationAudit:
    """Summarize a materialization stream without writing any database rows.

    The audit consumes the stream once, checks that all clean and quarantine
    batches share one publication identity, and reports per-factor counts. It
    is intentionally independent of a target session so callers can run this
    preflight before deciding whether a source version is eligible to publish.
    """
    iterator = iter(batches)
    first: FactorMaterializationBatches | None = None
    clean_counts: Counter[str] = Counter()
    quarantine_counts: Counter[str] = Counter()
    quarantine_reason_counts: Counter[tuple[str, str]] = Counter()
    for materialized in iterator:
        clean = materialized.clean_batch
        if clean.object_name != "factor_observation_clean":
            raise FactorBackfillError("clean batch has an invalid publish object")
        if first is None:
            first = materialized
        else:
            expected = first.clean_batch
            if (
                clean.logical_partition != expected.logical_partition
                or clean.data_version != expected.data_version
                or clean.business_keys != expected.business_keys
            ):
                raise FactorBackfillError("materialization batches have mixed publish identity")
        _count_factor_rows(clean.frame, clean_counts, "clean")

        quarantine = materialized.quarantine_batch
        if quarantine is None:
            continue
        if quarantine.object_name != "factor_quarantine":
            raise FactorBackfillError("quarantine batch has an invalid publish object")
        if quarantine.data_version != clean.data_version:
            raise FactorBackfillError("materialization batches have mixed publish identity")
        if "reason" not in quarantine.frame.columns:
            raise FactorBackfillError("quarantine batch is missing reason")
        _count_factor_rows(quarantine.frame, quarantine_counts, "quarantine")
        for row in quarantine.frame.loc[:, ["factor_id", "reason"]].itertuples(index=False):
            quarantine_reason_counts[(str(row.factor_id), str(row.reason))] += 1

    if first is None:
        raise FactorBackfillError("materialization stream is empty")
    requested = tuple(sorted(set(factor_ids)))
    if factor_ids and len(requested) != len(factor_ids):
        raise FactorBackfillError("factor_ids must be unique for materialization audit")
    all_factor_ids = tuple(sorted(set(requested).union(clean_counts).union(quarantine_counts)))
    return FactorMaterializationAudit(
        logical_partition=first.clean_batch.logical_partition,
        data_version=first.clean_batch.data_version,
        clean_rows_by_factor=tuple(
            (factor_id, int(clean_counts.get(factor_id, 0))) for factor_id in all_factor_ids
        ),
        quarantine_rows_by_factor=tuple(
            (factor_id, int(quarantine_counts.get(factor_id, 0))) for factor_id in all_factor_ids
        ),
        quarantine_reasons_by_factor=tuple(
            (factor_id, reason, int(count))
            for (factor_id, reason), count in sorted(quarantine_reason_counts.items())
        ),
    )


def _count_factor_rows(frame: pd.DataFrame, counts: Counter[str], kind: str) -> None:
    if "factor_id" not in frame.columns:
        raise FactorBackfillError(f"{kind} batch is missing factor_id")
    if frame["factor_id"].isna().any():
        raise FactorBackfillError(f"{kind} batch contains a null factor_id")
    counts.update(frame["factor_id"].astype(str).tolist())


# Columns this recovery path republishes exactly as it read them. ``run_id``,
# ``data_version``, ``created_at`` and ``record_hash`` are deliberately absent: the path
# overwrites all four, so duplicate rows left behind by an interrupted chunk stream are
# *expected* to disagree on them and the disagreement cannot reach the published version.
# Every other selected column is carried through unchanged, so a disagreement on one of
# these means the row that gets published depends on the order the database returned them.
_REPUBLISHED_UNCHANGED: Final[tuple[str, ...]] = (
    "value",
    "observation_date",
    "available_date",
    "report_period",
    "stale_days",
    "dedup_resolution",
    "source_revision_id",
    "source_watermark_id",
)


def _duplicate_payload_conflicts(
    frame: pd.DataFrame,
    keys: Sequence[str],
    payload_columns: Sequence[str],
) -> list[tuple[tuple[object, ...], list[str]]]:
    """Business keys whose duplicate rows disagree on a column published unchanged.

    The deduplication in :func:`_iter_client_repair_batches` sorts by ``keys`` and then
    drops duplicates on ``keys``, so every row in a duplicate group is equal on all of the
    sort keys by construction -- that sort cannot break the tie, and ``keep="first"`` is
    really "whichever row the query returned first". That is harmless when the tied rows
    agree on everything this path republishes, and unsafe the moment they do not, because
    the published value would then depend on a row order nothing guarantees. This finds the
    unsafe case so the caller can refuse it instead of resolving it silently; refusing
    matches ``pit.market``'s handling of an unequal duplicate group and keeps this path
    from having to choose a semantic that belongs to nobody's decision.
    """
    present = [column for column in payload_columns if column in frame.columns]
    if frame.empty or not present:
        return []
    counts = frame.groupby(list(keys), sort=True, dropna=False)[present].nunique(dropna=False)
    conflicts: list[tuple[tuple[object, ...], list[str]]] = []
    for key, row in counts.iterrows():
        differing = sorted(row.index[row > 1].tolist())
        if differing:
            conflicts.append((key if isinstance(key, tuple) else (key,), differing))
    return conflicts


def publish_client_deduplicated_repair(
    target_session: FactorBackfillTargetSession,
    *,
    old_data_version: str,
    new_data_version: str,
    logical_partition: str,
) -> None:
    """Publish a deduplicated repair using bounded client-side daily reads.

    This recovery path is for an untrusted source version that contains
    duplicate rows from an interrupted chunk stream. It never deletes the old
    version; the new version becomes visible only after all daily chunks pass
    duplicate-key validation.
    """
    if not old_data_version.strip() or not new_data_version.strip():
        raise FactorBackfillError("repair data versions cannot be empty")
    if old_data_version == new_data_version:
        raise FactorBackfillError("repair data versions must differ")
    month = FactorMonth.from_partition(logical_partition)
    checks = [
        CheckResult("repair_source_rows", Severity.BLOCKER, CheckStatus.PASS),
        CheckResult("repair_keys_unique", Severity.BLOCKER, CheckStatus.PASS),
        CheckResult("repair_finite_values", Severity.BLOCKER, CheckStatus.PASS),
    ]
    V2BatchPublisher(target_session).publish_chunk_stream(
        _iter_client_repair_batches(
            target_session,
            old_data_version=old_data_version,
            new_data_version=new_data_version,
            month=month,
        ),
        checks,
    )


def _iter_client_repair_batches(
    target_session: FactorBackfillTargetSession,
    *,
    old_data_version: str,
    new_data_version: str,
    month: FactorMonth,
) -> Iterator[StagedBatch]:
    run_id = str(uuid.uuid4())
    created_at = pd.Timestamp.now(tz="UTC").tz_localize(None)
    current = month.start_date
    yielded = False
    while current <= month.end_date:
        day = current.strftime("%Y.%m.%d")
        query = (
            "select signal_date,ts_code,factor_id,factor_version,value,observation_date,"
            "available_date,report_period,stale_days,dedup_resolution,source_revision_id,"
            "run_id,data_version,created_at,source_watermark_id,record_hash "
            "from loadTable("
            f"'{target_session.target_database}', 'factor_observation_clean') "
            f"where data_version='{_quoted(old_data_version)}' and signal_date=date('{day}')"
        )
        result = target_session.query(query)
        if not isinstance(result, pd.DataFrame):
            raise FactorBackfillError("factor repair query did not return a DataFrame")
        if not result.empty:
            frame = result.copy(deep=True)
            keys = ["signal_date", "ts_code", "factor_id", "factor_version"]
            # The sort below cannot break a tie: its keys are the dedup keys, so rows in a
            # duplicate group are equal on every one of them. Refuse the case where that
            # would matter rather than let the query's row order decide a published value.
            conflicts = _duplicate_payload_conflicts(frame, keys, _REPUBLISHED_UNCHANGED)
            if conflicts:
                first_key, columns = conflicts[0]
                raise FactorBackfillError(
                    f"repair source has {len(conflicts)} business key(s) whose duplicate rows "
                    f"disagree on columns this path republishes unchanged, so the surviving row "
                    f"would be chosen by the query's row order; first such key {first_key} "
                    f"differing on {columns}. This path collapses duplicates left by an "
                    f"interrupted chunk stream, which are expected to differ only on run_id, "
                    f"data_version, created_at and record_hash -- all four are overwritten. "
                    f"A disagreement on anything else is a source conflict, and resolving it "
                    f"is not this recovery path's to decide."
                )
            frame = frame.sort_values(keys, kind="mergesort").drop_duplicates(
                keys, keep="first"
            )
            frame["run_id"] = run_id
            frame["data_version"] = new_data_version
            frame["created_at"] = created_at
            frame["record_hash"] = _record_hashes(frame, strategy="fast")
            yielded = True
            yield StagedBatch.create(
                object_name="factor_observation_clean",
                logical_partition=month.logical_partition,
                data_version=new_data_version,
                frame=frame.reset_index(drop=True),
                business_keys=tuple(keys),
                partition_date_column="signal_date",
            )
        current += pd.Timedelta(days=1).to_pytimedelta()
    if not yielded:
        raise FactorBackfillError(
            f"repair source has no rows in {month.logical_partition}: {old_data_version}"
        )


def build_factor_definitions(
    metadata: pd.DataFrame,
    factor_ids: Iterable[str],
) -> tuple[FactorDefinition, ...]:
    """Convert selected YesLab metadata rows into immutable source definitions."""
    requested = tuple(factor_ids)
    if not requested or len(set(requested)) != len(requested):
        raise FactorBackfillError("factor_ids must be non-empty and unique")
    if any(_FACTOR_ID.fullmatch(factor_id) is None for factor_id in requested):
        raise FactorBackfillError("factor_ids must use yeslab_<integer> form")
    required = {
        "factor_id",
        "factor_name",
        "creator",
        "update_frequency",
        "store_factor",
    }
    if missing := sorted(required.difference(metadata.columns)):
        raise FactorBackfillError(f"factor metadata is missing columns: {missing}")
    numeric_ids = {int(factor_id.removeprefix("yeslab_")) for factor_id in requested}
    selected = metadata.loc[metadata["factor_id"].astype(int).isin(numeric_ids)].copy()
    catalog = build_yeslab_catalog(selected)
    by_id = {str(entry["factor_id"]): entry for entry in catalog}
    missing_ids = sorted(set(requested).difference(by_id))
    if missing_ids:
        raise FactorBackfillError(f"factor metadata is missing requested IDs: {missing_ids}")
    definitions: list[FactorDefinition] = []
    for factor_id in sorted(requested):
        entry = by_id[factor_id]
        definitions.append(
            FactorDefinition(
                factor_id=factor_id,
                factor_version=1,
                display_name=str(entry["display_name"]),
                source_database=str(entry["source_database"]),
                source_table=str(entry["source_table"]),
                source_value_column=str(entry["source_value_column"]),
                source_date_column=str(entry["source_date_column"]),
                source_code_column=str(entry["source_code_column"]),
                factor_type=FactorType(str(entry["factor_type"])),
                frequency=str(entry["frequency"]),
                expected_direction=int(cast(Any, entry["expected_direction"])),
                availability_policy=AvailabilityPolicy(str(entry["availability_policy"])),
                max_staleness_calendar_days=int(
                    cast(Any, entry["max_staleness_calendar_days"])
                ),
                unit=str(entry["unit"]),
                owner=str(entry["owner"]),
            )
        )
    names = [definition.display_name for definition in definitions]
    if len(set(names)) != len(names):
        raise FactorBackfillError("requested factors have duplicate source factor names")
    return tuple(definitions)


def build_factor_month_context(
    month: FactorMonth,
    definitions: tuple[FactorDefinition, ...],
    *,
    schema_version: str = "8",
    attempt_nonce: str | None = None,
) -> RunContext:
    """Build a deterministic data version for one factor/month selection."""
    config = {
        "factor_ids": tuple(definition.factor_id for definition in definitions),
        "factor_versions": tuple(definition.factor_version for definition in definitions),
        "logical_partition": month.logical_partition,
        "source_tables": tuple(sorted({definition.source_table for definition in definitions})),
        "operation": "selected_technical_factor_backfill",
    }
    if attempt_nonce is not None:
        if not attempt_nonce.strip():
            raise FactorBackfillError("attempt_nonce cannot be empty")
        config["attempt_nonce"] = attempt_nonce
    return RunContext.build(
        config=config,
        watermarks=(
            f"YesLab|{table}|{month.logical_partition}"
            for table in sorted({definition.source_table for definition in definitions})
        ),
        code_commit="factor-source-backfill-v1",
        schema_version=schema_version,
    )


def preflight_factor_month(
    source_session: YesLabFactorSession,
    metadata: pd.DataFrame,
    *,
    factor_ids: tuple[str, ...],
    logical_partition: str,
    schema_version: str = "8",
    attempt_nonce: str | None = None,
    repair_provider: RepairProvider | None = None,
) -> FactorMaterializationAudit:
    """Materialize one factor month in memory and return write-free audit counts."""
    month = FactorMonth.from_partition(logical_partition)
    definitions = build_factor_definitions(metadata, factor_ids)
    context = build_factor_month_context(
        month,
        definitions,
        schema_version=schema_version,
        attempt_nonce=attempt_nonce,
    )
    return audit_factor_materialization_batches(
        iter_factor_materialization_batches(
            source_session,
            definitions,
            month,
            context,
            include_lineage=False,
            repair_provider=repair_provider,
        ),
        factor_ids=factor_ids,
    )


def iter_clean_factor_batches(
    source_session: YesLabFactorSession,
    definitions: tuple[FactorDefinition, ...],
    month: FactorMonth,
    context: RunContext,
    *,
    fail_on_quarantine: bool = True,
    repair_provider: RepairProvider | None = None,
) -> Iterator[StagedBatch]:
    """Lazily query source tables and yield one clean batch per factor.

    The generator raises before completion if any requested factor is absent or
    produces a quarantine row. When used with ``publish_chunk_stream`` any rows
    already inserted remain invisible because the V2 pointer is advanced last.
    """
    for materialized in iter_factor_materialization_batches(
        source_session,
        definitions,
        month,
        context,
        repair_provider=repair_provider,
    ):
        repair_result = getattr(materialized, "repair_overlay", None)
        if fail_on_quarantine and repair_result is not None:
            repair_reasons = set(repair_result.quarantine["reason"].astype(str))
            blocking_reasons = repair_reasons.difference(
                {"WARMUP_OUT_OF_SCOPE", "NO_EVIDENCE_BACKFILL", "NO_FINITE_CANDIDATE"}
            )
            if repair_result.status is RepairStatus.BLOCKED or blocking_reasons:
                factor_id = str(materialized.clean_batch.frame["factor_id"].iloc[0])
                raise FactorBackfillError(
                    f"repair overlay blocked factor materialization: {factor_id} "
                    f"({sorted(blocking_reasons or {'REPAIR_BLOCKED'})})"
                )
        if (
            fail_on_quarantine
            and materialized.quarantine_batch is not None
            and not materialized.quarantine_batch.frame.empty
        ):
            quarantine = materialized.quarantine_batch.frame
            if "reason" not in quarantine.columns:
                raise FactorBackfillError("factor quarantine is missing its reason column")
            reasons = set(quarantine["reason"].astype(str))
            blocking_reasons = reasons.difference({"NO_FINITE_VALUE"})
            if not blocking_reasons:
                yield materialized.clean_batch
                continue
            factor_id = str(materialized.clean_batch.frame["factor_id"].iloc[0])
            raise FactorBackfillError(
                f"factor has quarantined conflict rows: {factor_id} "
                f"({sorted(blocking_reasons)})"
            )
        yield materialized.clean_batch


def iter_quarantine_factor_batches(
    source_session: YesLabFactorSession,
    definitions: tuple[FactorDefinition, ...],
    month: FactorMonth,
    context: RunContext,
    *,
    repair_provider: RepairProvider | None = None,
) -> Iterator[StagedBatch]:
    """Lazily yield only quarantined rows for a factor/month preflight."""
    for materialized in iter_factor_materialization_batches(
        source_session,
        definitions,
        month,
        context,
        repair_provider=repair_provider,
    ):
        if (
            materialized.quarantine_batch is not None
            and not materialized.quarantine_batch.frame.empty
        ):
            yield materialized.quarantine_batch


def iter_factor_materialization_batches(
    source_session: YesLabFactorSession,
    definitions: tuple[FactorDefinition, ...],
    month: FactorMonth,
    context: RunContext,
    *,
    include_lineage: bool = True,
    repair_provider: RepairProvider | None = None,
) -> Iterator[FactorMaterializationBatches]:
    """Yield clean/quarantine materialization pairs while retaining no source table."""
    grouped: dict[str, list[FactorDefinition]] = defaultdict(list)
    for definition in definitions:
        grouped[definition.source_table].append(definition)
    seen: set[str] = set()
    for source_table in sorted(grouped):
        table_definitions = tuple(sorted(grouped[source_table], key=lambda item: item.factor_id))
        source = _query_source_month(source_session, source_table, table_definitions, month)
        if source.empty:
            continue
        watermark = _source_watermark(source_table, month, source, table_definitions)
        for definition in table_definitions:
            materialization_source = source
            repair_result = None
            if repair_provider is not None:
                factor_source = source.loc[
                    source["factor_name"].astype(str) == definition.display_name
                ].copy(deep=True)
                repair_result = build_repair_overlay(
                    factor_source,
                    repair_provider,
                    definition=definition,
                    logical_partition=month.logical_partition,
                    source_watermark_id=watermark,
                )
                if repair_result.status is RepairStatus.BLOCKED:
                    raise FactorBackfillError(
                        "repair overlay blocked factor materialization: "
                        f"{definition.factor_id} ({month.logical_partition})"
                    )
                # The materializer consumes the existing canonical business
                # columns.  Repair lineage/audit remains attached to the
                # returned materialization object for the caller.
                materialization_source = repair_result.clean.loc[
                    :, ["trade_date", "ts_code", "factor_name", "value"]
                ].copy(deep=True)
            try:
                materialized = build_technical_factor_batches(
                    materialization_source,
                    definition,
                    source_factor_name=definition.display_name,
                    source_watermark_id=watermark,
                    context=context,
                    record_hash_strategy="fast",
                    include_lineage=include_lineage,
                )
            except ValueError as error:
                raise FactorBackfillError(
                    f"factor has no clean observations (source absent or fully quarantined) "
                    f"in {month.logical_partition}: "
                    f"{definition.factor_id}"
                ) from error
            if repair_result is not None:
                materialized = RepairFactorMaterializationBatches(
                    clean_batch=materialized.clean_batch,
                    quarantine_batch=materialized.quarantine_batch,
                    repair_overlay=repair_result,
                    repair_audit=repair_result.decision,
                )
            seen.add(definition.factor_id)
            yield materialized
    missing = sorted({definition.factor_id for definition in definitions}.difference(seen))
    if missing:
        raise FactorBackfillError(
            f"requested factors have no clean observations in {month.logical_partition}: {missing}"
        )


def publish_factor_month(
    source_session: YesLabFactorSession,
    target_session: FactorBackfillTargetSession,
    metadata: pd.DataFrame,
    *,
    factor_ids: tuple[str, ...],
    logical_partition: str,
    fail_on_quarantine: bool = True,
    publish_quarantine: bool = False,
    attempt_nonce: str | None = None,
    repair_current: bool = False,
    repair_provider: RepairProvider | None = None,
) -> FactorBackfillResult:
    """Materialize and publish one selected frozen-factor month to V2."""
    month = FactorMonth.from_partition(logical_partition)
    definitions = build_factor_definitions(metadata, factor_ids)
    context = build_factor_month_context(month, definitions, attempt_nonce=attempt_nonce)
    checks = [
        CheckResult("factor_source_schema", Severity.BLOCKER, CheckStatus.PASS),
        CheckResult("factor_keys_unique", Severity.BLOCKER, CheckStatus.PASS),
        CheckResult("factor_finite_values", Severity.BLOCKER, CheckStatus.PASS),
    ]
    if publish_quarantine and fail_on_quarantine:
        raise FactorBackfillError(
            "publish_quarantine requires fail_on_quarantine=False"
        )
    if publish_quarantine:
        quarantine_batches = tuple(
            iter_quarantine_factor_batches(
                source_session,
                definitions,
                month,
                context,
                repair_provider=repair_provider,
            )
        )
        if quarantine_batches:
            V2BatchPublisher(target_session).publish_chunk_stream(quarantine_batches, checks)
    batches = iter_clean_factor_batches(
        source_session,
        definitions,
        month,
        context,
        fail_on_quarantine=fail_on_quarantine,
        repair_provider=repair_provider,
    )
    V2BatchPublisher(target_session).publish_chunk_stream(
        batches,
        checks,
        allow_current_repair=repair_current,
    )
    return FactorBackfillResult(
        logical_partition=logical_partition,
        data_version=context.data_version,
        factor_ids=tuple(definition.factor_id for definition in definitions),
    )


def _query_source_month(
    session: YesLabFactorSession,
    source_table: str,
    definitions: tuple[FactorDefinition, ...],
    month: FactorMonth,
) -> pd.DataFrame:
    if not re.fullmatch(r"factor_data(?:[1-9]|10|11)", source_table):
        raise FactorBackfillError(f"unsafe YesLab factor table: {source_table}")
    factor_names = ", ".join(
        f"'{_quoted(definition.display_name)}'" for definition in definitions
    )
    query = (
        "select ts_code, trade_date, factor_name, value from "
        f"loadTable('dfs://YesLab', '{source_table}') "
        f"where factor_name in [{factor_names}] "
        f"and trade_date between date('{month.start_date.strftime('%Y.%m.%d')}') "
        f"and date('{month.end_date.strftime('%Y.%m.%d')}')"
    )
    result = session.query(query)
    if not isinstance(result, pd.DataFrame):
        raise FactorBackfillError("YesLab factor query did not return a DataFrame")
    if set(_SOURCE_COLUMNS).difference(result.columns):
        raise FactorBackfillError("YesLab factor query returned an invalid schema")
    return result.loc[:, list(_SOURCE_COLUMNS)].copy(deep=True)


def _source_watermark(
    source_table: str,
    month: FactorMonth,
    source: pd.DataFrame,
    definitions: tuple[FactorDefinition, ...],
) -> str:
    payload = {
        "factor_ids": [definition.factor_id for definition in definitions],
        "logical_partition": month.logical_partition,
        "max_date": str(source["trade_date"].max()),
        "min_date": str(source["trade_date"].min()),
        "row_count": len(source),
        "source_table": source_table,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _quoted(value: str) -> str:
    return value.replace("'", "''")
