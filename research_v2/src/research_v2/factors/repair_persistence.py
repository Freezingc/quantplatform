"""Persistence contracts for factor repair audit and quarantine evidence.

The repair overlay itself is intentionally write-free.  This module is the
small boundary that turns one or more in-memory repair decisions into the two
append-only V2 tables introduced by migration ``009``.  Identity is derived
from the complete repair scope, rather than from iteration order, so audit and
quarantine batches can be built independently and still share one run/data
version/watermark.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from enum import StrEnum
from typing import Any, cast

import numpy as np
import pandas as pd

from research_v2.db.publish import StagedBatch
from research_v2.quality.engine import CheckResult, CheckStatus, Severity


class RepairPersistenceError(ValueError):
    """Raised when repair evidence cannot be mapped to migration 009."""


class RepairDecisionStatus(StrEnum):
    """Stable names used in ``factor_repair_audit.decision_status``."""

    PASS = "REPAIR_AUDIT_PASS"
    PASS_WITH_QUARANTINE = "REPAIR_AUDIT_PASS_WITH_UNRESOLVED_QUARANTINE"
    BLOCKED = "REPAIR_BLOCKED"
    PENDING_RECHECK = "REPAIRED_PENDING_QUALITY_RECHECK"


@dataclass(frozen=True)
class RepairAuditRecord:
    """One factor/month repair decision and its aggregate audit counters.

    The first fields mirror the required columns of ``factor_repair_audit``.
    Identity metadata (definition/provider hashes and date range) is retained
    on the Python contract so the builders can derive a reproducible lineage
    identity; those helper fields are not uploaded to migration 009.
    """

    as_of_date: date
    logical_partition: str
    factor_id: str
    factor_version: int
    source_table: str
    source_factor_name: str
    repair_method: str
    source_rows: int = 0
    candidate_rows: int = 0
    exact_duplicate_groups: int = 0
    exact_duplicate_rows: int = 0
    conflict_groups: int = 0
    nonfinite_rows: int = 0
    overlap_finite_rows: int = 0
    fillable_rows: int = 0
    filled_rows: int = 0
    added_rows: int = 0
    remaining_nonfinite_rows: int = 0
    mismatch_rows: int = 0
    max_abs_diff: float = 0.0
    max_rel_diff: float = 0.0
    pit_violation_rows: int = 0
    unresolved_rows: int = 0
    decision_status: str = RepairDecisionStatus.PASS.value
    rules_version: int = 1
    run_id: str | uuid.UUID | None = None
    data_version: str | None = None
    created_at: datetime | pd.Timestamp | None = None
    source_watermark_id: str | None = None
    record_hash: str | None = None
    source_definition_hash: str = ""
    # ``definition_hash`` is accepted as a convenient alias by callers that
    # do not use the longer migration-oriented name.
    definition_hash: str = ""
    provider_hash: str = ""
    candidate_hash: str = ""
    start_date: date | None = None
    end_date: date | None = None


@dataclass(frozen=True)
class RepairQuarantineRow:
    """One rejected source/candidate observation for migration 009."""

    observation_date: date
    ts_code: str
    factor_id: str
    factor_version: int
    factor_name: str
    reason: str
    source_value: float | int | None
    candidate_value: float | int | None
    repair_method: str
    candidate_hash: str
    run_id: str | uuid.UUID | None = None
    data_version: str | None = None
    created_at: datetime | pd.Timestamp | None = None
    source_watermark_id: str | None = None
    record_hash: str | None = None
    source_definition_hash: str = ""
    definition_hash: str = ""
    provider_hash: str = ""
    rules_version: int = 1
    start_date: date | None = None
    end_date: date | None = None


_AUDIT_COLUMNS: tuple[str, ...] = (
    "as_of_date",
    "logical_partition",
    "factor_id",
    "factor_version",
    "source_table",
    "source_factor_name",
    "repair_method",
    "source_rows",
    "candidate_rows",
    "exact_duplicate_groups",
    "exact_duplicate_rows",
    "conflict_groups",
    "nonfinite_rows",
    "overlap_finite_rows",
    "fillable_rows",
    "filled_rows",
    "added_rows",
    "remaining_nonfinite_rows",
    "mismatch_rows",
    "max_abs_diff",
    "max_rel_diff",
    "pit_violation_rows",
    "unresolved_rows",
    "decision_status",
    "rules_version",
    "run_id",
    "data_version",
    "created_at",
    "source_watermark_id",
    "record_hash",
)
_QUARANTINE_COLUMNS: tuple[str, ...] = (
    "observation_date",
    "ts_code",
    "factor_id",
    "factor_version",
    "factor_name",
    "reason",
    "source_value",
    "candidate_value",
    "repair_method",
    "candidate_hash",
    "run_id",
    "data_version",
    "created_at",
    "source_watermark_id",
    "record_hash",
)


def build_repair_audit_batch(records: Iterable[RepairAuditRecord]) -> StagedBatch:
    """Map repair decisions to one deterministic ``factor_repair_audit`` batch."""

    materialized = tuple(records)
    if not materialized:
        raise RepairPersistenceError("repair audit batch cannot be empty")
    _validate_audit_records(materialized)
    identity = _identity(materialized, ())
    frame = pd.DataFrame(
        [_audit_row(record, identity) for record in materialized],
        columns=list(_AUDIT_COLUMNS),
    )
    frame = frame.sort_values(
        ["as_of_date", "factor_id", "factor_version"], kind="mergesort"
    ).reset_index(drop=True)
    partition, date_column = _partition(frame, "as_of_date", "logical_partition")
    return StagedBatch.create(
        object_name="factor_repair_audit",
        logical_partition=partition,
        data_version=identity.data_version,
        frame=frame,
        business_keys=("as_of_date", "factor_id", "factor_version"),
        partition_date_column=date_column,
    )


def build_repair_quarantine_batch(
    rows: Iterable[RepairQuarantineRow],
) -> StagedBatch | None:
    """Map rejected observations to ``factor_repair_quarantine``.

    No quarantine rows is a normal outcome for a clean repair and therefore
    returns ``None`` instead of manufacturing an empty (unpublishable) batch.
    """

    materialized = tuple(rows)
    if not materialized:
        return None
    _validate_quarantine_rows(materialized)
    identity = _identity((), materialized)
    frame = pd.DataFrame(
        [_quarantine_row(row, identity) for row in materialized],
        columns=list(_QUARANTINE_COLUMNS),
    )
    frame = frame.sort_values(
        ["observation_date", "factor_id", "ts_code", "reason"], kind="mergesort"
    ).reset_index(drop=True)
    partition, date_column = _partition(frame, "observation_date", None)
    return StagedBatch.create(
        object_name="factor_repair_quarantine",
        logical_partition=partition,
        data_version=identity.data_version,
        frame=frame,
        business_keys=("observation_date", "ts_code", "factor_id", "factor_version", "reason"),
        partition_date_column=date_column,
    )


def repair_quality_checks(audit: RepairAuditRecord) -> list[CheckResult]:
    """Return fail-closed blocker checks for one repair decision.

    Quarantine and unresolved evidence never gets silently promoted: any
    non-zero conflict, mismatch, PIT violation, or unresolved/non-finite row
    fails a blocker.  ``REPAIRED_PENDING_QUALITY_RECHECK`` is intentionally
    accepted here; the predictive quality gate remains a separate workflow.
    """

    status = str(audit.decision_status)
    checks = [
        _check("repair_source_rows", audit.source_rows > 0),
        _check("repair_keys_unique", audit.conflict_groups == 0),
        _check("repair_finite_values", audit.remaining_nonfinite_rows == 0),
        _check("repair_overlap_mismatch", audit.mismatch_rows == 0),
        _check("repair_pit", audit.pit_violation_rows == 0),
        _check("repair_unresolved_rows", audit.unresolved_rows == 0),
        _check("repair_definition_hash", bool(_definition_hash(audit).strip())),
        _check("repair_rules_version", audit.rules_version >= 1),
        _check("repair_watermark", bool((audit.source_watermark_id or "").strip())),
        _check("repair_decision_status", status != RepairDecisionStatus.BLOCKED.value),
    ]
    return checks


@dataclass(frozen=True)
class _RepairIdentity:
    data_version: str
    source_watermark_id: str
    run_id: str
    created_at: datetime


def _identity(
    audits: Sequence[RepairAuditRecord],
    quarantines: Sequence[RepairQuarantineRow],
) -> _RepairIdentity:
    factor_ids = sorted(
        {
            *(record.factor_id for record in audits),
            *(row.factor_id for row in quarantines),
        }
    )
    definition_hashes = sorted(
        {
            *(_definition_hash(record) for record in audits),
            *(_definition_hash(row) for row in quarantines),
        }
        - {""}
    )
    provider_hashes = sorted(
        {
            *(record.provider_hash for record in audits),
            *(row.provider_hash for row in quarantines),
        }
        - {""}
    )
    dates = [
        *(
            _date_range(
                record.start_date,
                record.end_date,
                _logical_partition_range(record.logical_partition, record.as_of_date),
            )
            for record in audits
        ),
        *(
            _date_range(
                row.start_date,
                row.end_date,
                _month_range(row.observation_date),
            )
            for row in quarantines
        ),
    ]
    if not dates:
        raise RepairPersistenceError("repair identity requires a date range")
    start = min(item[0] for item in dates)
    end = max(item[1] for item in dates)
    rules_versions = sorted(
        {
            *(record.rules_version for record in audits),
            *(row.rules_version for row in quarantines),
        }
        - {""}
    )
    payload = {
        "factor_ids": factor_ids,
        "definition_hashes": definition_hashes,
        "date_range": [start.isoformat(), end.isoformat()],
        "provider_hashes": provider_hashes,
        "rules_versions": rules_versions,
    }
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    data_version = hashlib.sha256(("repair-data:" + canonical).encode("utf-8")).hexdigest()
    source_watermark_id = hashlib.sha256(
        ("repair-source:" + canonical).encode("utf-8")
    ).hexdigest()
    explicit_runs = {
        str(value)
        for value in [
            *(record.run_id for record in audits),
            *(row.run_id for row in quarantines),
        ]
        if value is not None and str(value).strip()
    }
    if len(explicit_runs) > 1:
        raise RepairPersistenceError("repair records contain mixed run_id values")
    run_id = next(iter(explicit_runs), str(uuid.uuid5(uuid.NAMESPACE_URL, data_version)))
    explicit_created = [
        _as_datetime(value)
        for value in [
            *(record.created_at for record in audits),
            *(row.created_at for row in quarantines),
        ]
        if value is not None
    ]
    created_at = max(explicit_created, default=datetime.combine(end, time.min))
    return _RepairIdentity(
        data_version=data_version,
        source_watermark_id=source_watermark_id,
        run_id=run_id,
        created_at=created_at,
    )


def _audit_row(record: RepairAuditRecord, identity: _RepairIdentity) -> dict[str, object]:
    row = {
        "as_of_date": _as_date(record.as_of_date),
        "logical_partition": record.logical_partition,
        "factor_id": record.factor_id,
        "factor_version": int(record.factor_version),
        "source_table": record.source_table,
        "source_factor_name": record.source_factor_name,
        "repair_method": record.repair_method,
        "source_rows": int(record.source_rows),
        "candidate_rows": int(record.candidate_rows),
        "exact_duplicate_groups": int(record.exact_duplicate_groups),
        "exact_duplicate_rows": int(record.exact_duplicate_rows),
        "conflict_groups": int(record.conflict_groups),
        "nonfinite_rows": int(record.nonfinite_rows),
        "overlap_finite_rows": int(record.overlap_finite_rows),
        "fillable_rows": int(record.fillable_rows),
        "filled_rows": int(record.filled_rows),
        "added_rows": int(record.added_rows),
        "remaining_nonfinite_rows": int(record.remaining_nonfinite_rows),
        "mismatch_rows": int(record.mismatch_rows),
        "max_abs_diff": float(record.max_abs_diff),
        "max_rel_diff": float(record.max_rel_diff),
        "pit_violation_rows": int(record.pit_violation_rows),
        "unresolved_rows": int(record.unresolved_rows),
        "decision_status": str(record.decision_status),
        "rules_version": record.rules_version,
        "run_id": identity.run_id,
        "data_version": identity.data_version,
        "created_at": identity.created_at,
        "source_watermark_id": identity.source_watermark_id,
    }
    row["record_hash"] = _record_hash(row)
    return row


def _quarantine_row(row: RepairQuarantineRow, identity: _RepairIdentity) -> dict[str, object]:
    mapped = {
        "observation_date": _as_date(row.observation_date),
        "ts_code": row.ts_code,
        "factor_id": row.factor_id,
        "factor_version": int(row.factor_version),
        "factor_name": row.factor_name,
        "reason": row.reason,
        "source_value": _as_float_or_nan(row.source_value),
        "candidate_value": _as_float_or_nan(row.candidate_value),
        "repair_method": row.repair_method,
        "candidate_hash": row.candidate_hash,
        "run_id": identity.run_id,
        "data_version": identity.data_version,
        "created_at": identity.created_at,
        "source_watermark_id": identity.source_watermark_id,
    }
    mapped["record_hash"] = _record_hash(mapped)
    return mapped


def _record_hash(row: dict[str, object]) -> str:
    payload = {
        key: value.isoformat() if isinstance(value, (date, datetime, pd.Timestamp)) else value
        for key, value in row.items()
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _validate_audit_records(records: Sequence[RepairAuditRecord]) -> None:
    for record in records:
        if not record.factor_id.strip() or not record.logical_partition.strip():
            raise RepairPersistenceError("repair audit factor identity cannot be empty")
        if record.factor_version < 1:
            raise RepairPersistenceError("factor_version must be positive")
        if not record.source_table.strip() or not record.source_factor_name.strip():
            raise RepairPersistenceError("repair audit source identity cannot be empty")
        if not record.repair_method.strip() or record.rules_version < 1:
            raise RepairPersistenceError("repair audit method/rules cannot be empty")
        _validate_nonnegative_counts(record)


def _validate_nonnegative_counts(record: RepairAuditRecord) -> None:
    for name in (
        "source_rows",
        "candidate_rows",
        "exact_duplicate_groups",
        "exact_duplicate_rows",
        "conflict_groups",
        "nonfinite_rows",
        "overlap_finite_rows",
        "fillable_rows",
        "filled_rows",
        "added_rows",
        "remaining_nonfinite_rows",
        "mismatch_rows",
        "pit_violation_rows",
        "unresolved_rows",
    ):
        if getattr(record, name) < 0:
            raise RepairPersistenceError(f"{name} cannot be negative")
    if not np.isfinite(float(record.max_abs_diff)) or not np.isfinite(float(record.max_rel_diff)):
        raise RepairPersistenceError("repair mismatch maxima must be finite")


def _validate_quarantine_rows(rows: Sequence[RepairQuarantineRow]) -> None:
    for row in rows:
        if not row.ts_code.strip() or not row.factor_id.strip() or not row.factor_name.strip():
            raise RepairPersistenceError("repair quarantine business identity cannot be empty")
        if row.factor_version < 1 or not row.reason.strip() or not row.repair_method.strip():
            raise RepairPersistenceError("repair quarantine identity is invalid")
        if not row.candidate_hash.strip():
            raise RepairPersistenceError("repair quarantine candidate_hash cannot be empty")
        for value in (row.source_value, row.candidate_value):
            if value is not None and not np.isfinite(float(value)):
                raise RepairPersistenceError("repair quarantine values must be finite or null")


def _partition(
    frame: pd.DataFrame,
    date_column: str,
    logical_partitions: str | None,
) -> tuple[str, str | None]:
    months = sorted(
        pd.to_datetime(frame[date_column], errors="raise").dt.strftime("%Y-%m").unique()
    )
    if len(months) == 1:
        return str(months[0]), date_column
    return str(logical_partitions or f"{months[0]}_{months[-1]}"), None


def _definition_hash(value: Any) -> str:
    return str(
        getattr(value, "source_definition_hash", "")
        or getattr(value, "definition_hash", "")
    )


def _date_range(
    start: date | None,
    end: date | None,
    fallback: date | tuple[date, date],
) -> tuple[date, date]:
    fallback_start, fallback_end = (
        fallback if isinstance(fallback, tuple) else (_as_date(fallback), _as_date(fallback))
    )
    lower = _as_date(start) if start is not None else fallback_start
    upper = _as_date(end) if end is not None else fallback_end
    if lower > upper:
        raise RepairPersistenceError("repair date range cannot be reversed")
    return lower, upper


def _month_range(value: date) -> tuple[date, date]:
    period = pd.Period(value, freq="M")
    return period.start_time.date(), period.end_time.date()


def _logical_partition_range(logical_partition: str, fallback: date) -> tuple[date, date]:
    try:
        period = pd.Period(logical_partition, freq="M")
    except (TypeError, ValueError):
        return _month_range(fallback)
    if str(period) != logical_partition:
        return _month_range(fallback)
    return period.start_time.date(), period.end_time.date()


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(cast(Any, value)).date()


def _as_datetime(value: object) -> datetime:
    timestamp = pd.Timestamp(cast(Any, value))
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(UTC).tz_localize(None)
    return timestamp.to_pydatetime()


def _as_float_or_nan(value: float | int | None) -> float:
    return float("nan") if value is None else float(value)


def _check(name: str, passed: bool) -> CheckResult:
    return CheckResult(name, Severity.BLOCKER, CheckStatus.PASS if passed else CheckStatus.FAIL)
