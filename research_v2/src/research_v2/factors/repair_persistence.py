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
    source_table: str = ""
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


def build_repair_audit_batch(
    records: Iterable[RepairAuditRecord],
    *,
    identity: RepairIdentity | None = None,
) -> StagedBatch:
    """Map repair decisions to one deterministic ``factor_repair_audit`` batch."""

    materialized = tuple(records)
    if not materialized:
        raise RepairPersistenceError("repair audit batch cannot be empty")
    _validate_audit_records(materialized)
    resolved_identity = identity or build_repair_identity(materialized)
    _validate_identity_scope(resolved_identity, audits=materialized)
    frame = pd.DataFrame(
        [_audit_row(record, resolved_identity) for record in materialized],
        columns=list(_AUDIT_COLUMNS),
    )
    frame = frame.sort_values(
        ["as_of_date", "factor_id", "factor_version"], kind="mergesort"
    ).reset_index(drop=True)
    partition, date_column = _partition(frame, "as_of_date", "logical_partition")
    return StagedBatch.create(
        object_name="factor_repair_audit",
        logical_partition=partition,
        data_version=resolved_identity.data_version,
        frame=frame,
        business_keys=("as_of_date", "factor_id", "factor_version"),
        partition_date_column=date_column,
    )


def build_repair_quarantine_batch(
    rows: Iterable[RepairQuarantineRow],
    *,
    identity: RepairIdentity | None = None,
) -> StagedBatch | None:
    """Map rejected observations to ``factor_repair_quarantine``.

    No quarantine rows is a normal outcome for a clean repair and therefore
    returns ``None`` instead of manufacturing an empty (unpublishable) batch.
    """

    materialized = tuple(rows)
    if not materialized:
        return None
    _validate_quarantine_rows(materialized)
    resolved_identity = identity or build_repair_identity((), materialized)
    _validate_identity_scope(resolved_identity, quarantines=materialized)
    frame = pd.DataFrame(
        [_quarantine_row(row, resolved_identity) for row in materialized],
        columns=list(_QUARANTINE_COLUMNS),
    )
    frame = frame.sort_values(
        ["observation_date", "factor_id", "ts_code", "reason"], kind="mergesort"
    ).reset_index(drop=True)
    partition, date_column = _partition(frame, "observation_date", None)
    return StagedBatch.create(
        object_name="factor_repair_quarantine",
        logical_partition=partition,
        data_version=resolved_identity.data_version,
        frame=frame,
        business_keys=("observation_date", "ts_code", "factor_id", "factor_version", "reason"),
        partition_date_column=date_column,
    )


def repair_quality_checks(
    audit: RepairAuditRecord,
    *,
    identity: RepairIdentity | None = None,
) -> list[CheckResult]:
    """Return fail-closed blocker checks for one repair decision.

    Quarantine and unresolved evidence never gets silently promoted: any
    non-zero conflict, mismatch, PIT violation, or unresolved/non-finite row
    fails a blocker.  ``REPAIRED_PENDING_QUALITY_RECHECK`` is intentionally
    accepted here; the predictive quality gate remains a separate workflow.
    """

    _validate_audit_records((audit,))
    status = str(audit.decision_status)
    identity_scope_valid = True
    resolved_identity = identity
    if resolved_identity is None:
        try:
            resolved_identity = build_repair_identity([audit])
        except RepairPersistenceError:
            identity_scope_valid = False
    if resolved_identity is not None:
        try:
            _validate_identity_scope(resolved_identity, audits=[audit])
        except RepairPersistenceError:
            identity_scope_valid = False
    allowed_statuses = {
        RepairDecisionStatus.PASS.value,
        RepairDecisionStatus.PENDING_RECHECK.value,
    }
    checks = [
        _check("repair_source_rows", audit.source_rows > 0),
        _check("repair_keys_unique", audit.conflict_groups == 0),
        _check("repair_finite_values", audit.remaining_nonfinite_rows == 0),
        _check("repair_overlap_mismatch", audit.mismatch_rows == 0),
        _check("repair_pit", audit.pit_violation_rows == 0),
        _check("repair_unresolved_rows", audit.unresolved_rows == 0),
        _check("repair_definition_hash", bool(_definition_hash(audit).strip())),
        _check("repair_rules_version", audit.rules_version >= 1),
        _check("repair_identity_scope", identity_scope_valid),
        _check(
            "repair_watermark",
            bool(
                (
                    audit.source_watermark_id
                    or (resolved_identity.source_watermark_id if resolved_identity else "")
                ).strip()
            ),
        ),
        _check(
            "repair_decision_status",
            status in allowed_statuses and status != RepairDecisionStatus.BLOCKED.value,
        ),
    ]
    return checks


def repair_batch_quality_checks(
    records: Iterable[RepairAuditRecord],
    *,
    identity: RepairIdentity | None = None,
) -> list[CheckResult]:
    """Build uniquely named blocker checks for a multi-factor repair scope."""

    materialized = tuple(records)
    if not materialized:
        raise RepairPersistenceError("repair quality scope cannot be empty")
    resolved_identity = identity or build_repair_identity(materialized)
    _validate_identity_scope(resolved_identity, audits=materialized)
    checks: list[CheckResult] = []
    used_names: set[str] = set()
    for record in materialized:
        prefix = f"{record.factor_id}:{record.logical_partition}"
        suffix = 1
        for result in repair_quality_checks(record, identity=resolved_identity):
            check_name = f"{prefix}:{result.check_name}"
            while check_name in used_names:
                suffix += 1
                check_name = f"{prefix}:{suffix}:{result.check_name}"
            used_names.add(check_name)
            checks.append(CheckResult(check_name, result.severity, result.status))
    return checks


@dataclass(frozen=True)
class RepairIdentity:
    data_version: str
    source_watermark_id: str
    run_id: str
    created_at: datetime
    factor_ids: tuple[str, ...]
    factor_versions: tuple[tuple[str, int], ...]
    definition_hashes: tuple[str, ...]
    provider_hashes: tuple[str, ...]
    source_tables: tuple[str, ...]
    factor_names: tuple[str, ...]
    repair_methods: tuple[str, ...]
    candidate_hashes: tuple[str, ...]
    rules_versions: tuple[int, ...]
    start_date: date
    end_date: date


def build_repair_identity(
    audits: Iterable[RepairAuditRecord],
    quarantines: Iterable[RepairQuarantineRow] = (),
) -> RepairIdentity:
    """Build one scope identity for audit rows and any quarantine subset."""

    audit_records = tuple(audits)
    quarantine_rows = tuple(quarantines)
    if not audit_records and not quarantine_rows:
        raise RepairPersistenceError("repair identity scope cannot be empty")
    if audit_records:
        _validate_audit_records(audit_records)
    if quarantine_rows:
        _validate_quarantine_rows(
            quarantine_rows,
            require_provenance=not audit_records,
        )
    return _identity(audit_records, quarantine_rows)


def _identity(
    audits: Sequence[RepairAuditRecord],
    quarantines: Sequence[RepairQuarantineRow],
) -> RepairIdentity:
    factor_ids = sorted(
        {
            *(record.factor_id for record in audits),
            *(row.factor_id for row in quarantines),
        }
    )
    factor_versions = sorted(
        {
            *((record.factor_id, int(record.factor_version)) for record in audits),
            *((row.factor_id, int(row.factor_version)) for row in quarantines),
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
    source_tables = sorted(
        {
            *(record.source_table for record in audits),
            *(row.source_table for row in quarantines),
        }
        - {""}
    )
    factor_names = sorted(
        {
            *(record.source_factor_name for record in audits),
            *(row.factor_name for row in quarantines),
        }
        - {""}
    )
    repair_methods = sorted(
        {
            *(record.repair_method for record in audits),
            *(row.repair_method for row in quarantines),
        }
        - {""}
    )
    candidate_hashes = sorted(
        {
            *(record.candidate_hash for record in audits),
            *(row.candidate_hash for row in quarantines),
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
        - {0}
    )
    payload = {
        "factor_ids": factor_ids,
        "factor_versions": factor_versions,
        "definition_hashes": definition_hashes,
        "date_range": [start.isoformat(), end.isoformat()],
        "provider_hashes": provider_hashes,
        "source_tables": source_tables,
        "factor_names": factor_names,
        "repair_methods": repair_methods,
        "candidate_hashes": candidate_hashes,
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
    return RepairIdentity(
        data_version=data_version,
        source_watermark_id=source_watermark_id,
        run_id=run_id,
        created_at=created_at,
        factor_ids=tuple(factor_ids),
        factor_versions=tuple(factor_versions),
        definition_hashes=tuple(definition_hashes),
        provider_hashes=tuple(provider_hashes),
        source_tables=tuple(source_tables),
        factor_names=tuple(factor_names),
        repair_methods=tuple(repair_methods),
        candidate_hashes=tuple(candidate_hashes),
        rules_versions=tuple(rules_versions),
        start_date=start,
        end_date=end,
    )


def _audit_row(record: RepairAuditRecord, identity: RepairIdentity) -> dict[str, object]:
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


def _quarantine_row(row: RepairQuarantineRow, identity: RepairIdentity) -> dict[str, object]:
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
        try:
            period = pd.Period(record.logical_partition, freq="M")
        except (TypeError, ValueError) as error:
            raise RepairPersistenceError(
                "repair audit logical_partition must use YYYY-MM"
            ) from error
        if str(period) != record.logical_partition or period != pd.Period(
            record.as_of_date, freq="M"
        ):
            raise RepairPersistenceError(
                "repair audit logical_partition does not match as_of_date"
            )
        if record.factor_version < 1:
            raise RepairPersistenceError("factor_version must be positive")
        if not record.source_table.strip() or not record.source_factor_name.strip():
            raise RepairPersistenceError("repair audit source identity cannot be empty")
        if not _definition_hash(record).strip() or not record.provider_hash.strip():
            raise RepairPersistenceError(
                "repair audit provenance hashes are required for every record"
            )
        if not record.repair_method.strip() or record.rules_version < 1:
            raise RepairPersistenceError("repair audit method/rules cannot be empty")
        _validate_nonnegative_counts(record)


def _validate_identity_scope(
    identity: RepairIdentity,
    *,
    audits: Sequence[RepairAuditRecord] = (),
    quarantines: Sequence[RepairQuarantineRow] = (),
) -> None:
    """Ensure records are members of the identity used to publish them."""

    if not identity.factor_ids or not identity.definition_hashes or not identity.provider_hashes:
        raise RepairPersistenceError("repair identity scope is incomplete")
    if identity.start_date > identity.end_date:
        raise RepairPersistenceError("repair identity date range is reversed")
    for record in audits:
        if (
            record.factor_id not in identity.factor_ids
            or (record.factor_id, int(record.factor_version)) not in identity.factor_versions
        ):
            raise RepairPersistenceError("audit record is outside repair identity scope")
        _validate_scope_values(
            identity,
            factor_id=record.factor_id,
            factor_version=record.factor_version,
            factor_name=record.source_factor_name,
            source_table=record.source_table,
            repair_method=record.repair_method,
            definition_hash=_definition_hash(record),
            provider_hash=record.provider_hash,
            candidate_hash=record.candidate_hash,
            rules_version=record.rules_version,
            lower=_date_range(
                record.start_date,
                record.end_date,
                _logical_partition_range(record.logical_partition, record.as_of_date),
            )[0],
            upper=_date_range(
                record.start_date,
                record.end_date,
                _logical_partition_range(record.logical_partition, record.as_of_date),
            )[1],
        )
    for row in quarantines:
        if (
            row.factor_id not in identity.factor_ids
            or (row.factor_id, int(row.factor_version)) not in identity.factor_versions
        ):
            raise RepairPersistenceError("quarantine row is outside repair identity scope")
        _validate_scope_values(
            identity,
            factor_id=row.factor_id,
            factor_version=row.factor_version,
            factor_name=row.factor_name,
            source_table=row.source_table,
            repair_method=row.repair_method,
            definition_hash=_definition_hash(row),
            provider_hash=row.provider_hash,
            candidate_hash=row.candidate_hash,
            rules_version=row.rules_version,
            lower=_date_range(
                row.start_date,
                row.end_date,
                _month_range(row.observation_date),
            )[0],
            upper=_date_range(
                row.start_date,
                row.end_date,
                _month_range(row.observation_date),
            )[1],
        )


def _validate_scope_values(
    identity: RepairIdentity,
    *,
    factor_id: str,
    factor_version: int,
    factor_name: str,
    source_table: str,
    repair_method: str,
    definition_hash: str,
    provider_hash: str,
    candidate_hash: str,
    rules_version: int,
    lower: date,
    upper: date,
) -> None:
    if (factor_id, int(factor_version)) not in identity.factor_versions:
        raise RepairPersistenceError("record is outside repair identity scope")
    if factor_name and factor_name not in identity.factor_names:
        raise RepairPersistenceError("record is outside repair identity scope")
    if source_table and source_table not in identity.source_tables:
        raise RepairPersistenceError("record is outside repair identity scope")
    if repair_method and repair_method not in identity.repair_methods:
        raise RepairPersistenceError("record is outside repair identity scope")
    if definition_hash and definition_hash not in identity.definition_hashes:
        raise RepairPersistenceError("record is outside repair identity scope")
    if provider_hash and provider_hash not in identity.provider_hashes:
        raise RepairPersistenceError("record is outside repair identity scope")
    if candidate_hash and candidate_hash not in identity.candidate_hashes:
        raise RepairPersistenceError("record is outside repair identity scope")
    if rules_version not in identity.rules_versions:
        raise RepairPersistenceError("record is outside repair identity scope")
    if lower < identity.start_date or upper > identity.end_date:
        raise RepairPersistenceError("record is outside repair identity date range")


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


def _validate_quarantine_rows(
    rows: Sequence[RepairQuarantineRow],
    *,
    require_provenance: bool = False,
) -> None:
    for row in rows:
        if not row.ts_code.strip() or not row.factor_id.strip() or not row.factor_name.strip():
            raise RepairPersistenceError("repair quarantine business identity cannot be empty")
        if (
            row.factor_version < 1
            or row.rules_version < 1
            or not row.reason.strip()
            or not row.repair_method.strip()
        ):
            raise RepairPersistenceError("repair quarantine identity is invalid")
        if not row.candidate_hash.strip():
            raise RepairPersistenceError("repair quarantine candidate_hash cannot be empty")
        if require_provenance and (
            not row.source_table.strip()
            or not _definition_hash(row).strip()
            or not row.provider_hash.strip()
        ):
            raise RepairPersistenceError(
                "repair quarantine provenance hashes are required for a quarantine-only scope"
            )
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
    if len(months) != 1:
        raise RepairPersistenceError(
            f"repair batch cannot span multiple calendar months: {months}"
        )
    return str(months[0]), date_column


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
