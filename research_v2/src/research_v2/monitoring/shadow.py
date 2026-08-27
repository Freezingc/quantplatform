"""Label-free 60-day shadow-operation qualification accounting."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import cast

from research_v2.reporting.shadow import ShadowDailyReport

_MODEL_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_EVIDENCE_HASH = re.compile(r"^[0-9a-f]{64}$")
_SHADOW_APPROVAL_KEY_ENV = "QRV2_SHADOW_APPROVAL_HMAC_KEY"
_SHADOW_RUNTIME_KEY_ENV = "QRV2_SHADOW_RUNTIME_HMAC_KEY"
_SHADOW_EVIDENCE_KEY_ENV = "QRV2_SHADOW_EVIDENCE_HMAC_KEY"


@dataclass(frozen=True)
class ShadowDeploymentApproval:
    approval_id: str
    model_id: str


@dataclass(frozen=True)
class _ShadowDeploymentApprovalRecord:
    approval_id: str
    model_id: str
    actor: str
    approved_at: str
    walk_forward_evidence_hash: str
    artifact_bundle_hash: str
    inference_contract_hash: str
    v2test_drill_hash: str
    rollback_plan_hash: str


class ShadowDeploymentApprovalStore:
    """Issue and verify signed approvals before a model can enter SHADOW."""

    def __init__(
        self,
        root: Path,
        *,
        signing_key: bytes,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(signing_key, bytes) or not signing_key:
            raise ValueError("shadow approval signing_key must be non-empty bytes")
        self._root = root
        self._signing_key = signing_key
        self._clock = clock or (lambda: datetime.now(UTC))

    @classmethod
    def from_environment(cls, root: Path) -> ShadowDeploymentApprovalStore:
        secret = os.environ.get(_SHADOW_APPROVAL_KEY_ENV, "")
        if not secret:
            raise ValueError(f"shadow deployment approval requires {_SHADOW_APPROVAL_KEY_ENV}")
        return cls(root, signing_key=secret.encode("utf-8"))

    def issue(
        self,
        *,
        model_id: str,
        actor: str,
        walk_forward_evidence_hash: str,
        artifact_bundle_hash: str,
        inference_contract_hash: str,
        v2test_drill_hash: str,
        rollback_plan_hash: str,
    ) -> ShadowDeploymentApproval:
        payload = {
            "actor": actor,
            "approved_at": _timestamp(self._clock()),
            "artifact_bundle_hash": artifact_bundle_hash,
            "inference_contract_hash": inference_contract_hash,
            "model_id": model_id,
            "rollback_plan_hash": rollback_plan_hash,
            "v2test_drill_hash": v2test_drill_hash,
            "walk_forward_evidence_hash": walk_forward_evidence_hash,
        }
        record = _shadow_approval_from_payload(_shadow_approval_id(payload), payload)
        document = {
            "approval_id": record.approval_id,
            "payload": payload,
            "signature": _shadow_approval_signature(
                self._signing_key,
                record.approval_id,
                payload,
            ),
        }
        self._root.mkdir(parents=True, exist_ok=True)
        path = self._root / f"{record.approval_id}.json"
        try:
            _write_exclusive(path, _canonical_json(document) + "\n")
        except FileExistsError as error:
            raise FileExistsError(
                f"shadow deployment approval is immutable: {record.approval_id}"
            ) from error
        return ShadowDeploymentApproval(record.approval_id, record.model_id)

    def verify(self, approval: ShadowDeploymentApproval) -> _ShadowDeploymentApprovalRecord:
        return ShadowDeploymentApprovalVerifier(
            self._root,
            verification_key=self._signing_key,
        ).verify(approval)


class ShadowDeploymentApprovalVerifier:
    """Verify deployment approvals using an evaluator-owned key rather than caller assertions."""

    def __init__(self, root: Path, *, verification_key: bytes) -> None:
        if not isinstance(verification_key, bytes) or not verification_key:
            raise ValueError("shadow approval verification_key must be non-empty bytes")
        self._root = root
        self._verification_key = verification_key

    @classmethod
    def from_environment(cls, root: Path) -> ShadowDeploymentApprovalVerifier:
        secret = os.environ.get(_SHADOW_APPROVAL_KEY_ENV, "")
        if not secret:
            raise ValueError(f"shadow deployment approval requires {_SHADOW_APPROVAL_KEY_ENV}")
        return cls(root, verification_key=secret.encode("utf-8"))

    def verify(self, approval: ShadowDeploymentApproval) -> _ShadowDeploymentApprovalRecord:
        if not isinstance(approval, ShadowDeploymentApproval):
            raise ValueError("shadow deployment approval is invalid")
        path = self._root / f"{approval.approval_id}.json"
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("shadow deployment approval cannot be loaded") from error
        payload = document.get("payload") if isinstance(document, dict) else None
        signature = document.get("signature") if isinstance(document, dict) else None
        if (
            not isinstance(payload, dict)
            or document.get("approval_id") != approval.approval_id
            or not isinstance(signature, str)
            or _shadow_approval_id(payload) != approval.approval_id
            or not hmac.compare_digest(
                signature,
                _shadow_approval_signature(self._verification_key, approval.approval_id, payload),
            )
        ):
            raise ValueError("shadow deployment approval signature verification failed")
        record = _shadow_approval_from_payload(approval.approval_id, payload)
        if record.model_id != approval.model_id:
            raise ValueError("shadow deployment approval does not match its immutable record")
        return record


class ShadowQualificationStatus(StrEnum):
    INCOMPLETE = "INCOMPLETE"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


@dataclass(frozen=True)
class ShadowRuntimeDay:
    """Opaque reference to a scheduler-signed daily SHADOW runtime record."""

    runtime_day_id: str
    model_id: str


@dataclass(frozen=True)
class _ShadowRuntimeDayRecord:
    runtime_day_id: str
    model_id: str
    model_version: int
    signal_date: date
    recorded_at: datetime
    model_status: str
    shadow_pointer_model_id: str | None
    shadow_pointer_model_version: int | None
    has_blocker: bool
    has_psi_error: bool
    report: ShadowDailyReport


class ShadowRuntimeDayStore:
    """Trusted scheduler writer for immutable daily SHADOW runtime facts."""

    def __init__(
        self,
        root: Path,
        *,
        signing_key: bytes,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(signing_key, bytes) or not signing_key:
            raise ValueError("shadow runtime signing_key must be non-empty bytes")
        self._root = root
        self._signing_key = signing_key
        self._clock = clock or (lambda: datetime.now(UTC))

    @classmethod
    def from_environment(cls, root: Path) -> ShadowRuntimeDayStore:
        secret = os.environ.get(_SHADOW_RUNTIME_KEY_ENV, "")
        if not secret:
            raise ValueError(f"shadow runtime evidence requires {_SHADOW_RUNTIME_KEY_ENV}")
        return cls(root, signing_key=secret.encode("utf-8"))

    def issue(
        self,
        report: ShadowDailyReport,
        *,
        model_status: str,
        shadow_pointer_model_id: str | None,
        shadow_pointer_model_version: int | None,
        has_blocker: bool,
    ) -> ShadowRuntimeDay:
        report.verify()
        payload = {
            "has_blocker": has_blocker,
            "has_psi_error": _has_psi_error(report),
            "model_id": report.payload["model_id"],
            "model_status": model_status,
            "model_version": report.payload["model_version"],
            "recorded_at": _timestamp(self._clock()),
            "report_id": report.report_id,
            "report_payload": dict(report.payload),
            "shadow_pointer_model_id": shadow_pointer_model_id,
            "shadow_pointer_model_version": shadow_pointer_model_version,
            "signal_date": report.payload["signal_date"],
        }
        record = _shadow_runtime_day_from_payload(_shadow_runtime_day_id(payload), payload)
        document = {
            "payload": payload,
            "runtime_day_id": record.runtime_day_id,
            "signature": _shadow_runtime_day_signature(
                self._signing_key, record.runtime_day_id, payload
            ),
        }
        self._root.mkdir(parents=True, exist_ok=True)
        path = self._root / f"{record.runtime_day_id}.json"
        try:
            _write_exclusive(path, _canonical_json(document) + "\n")
        except FileExistsError as error:
            raise FileExistsError(
                f"shadow runtime day is immutable: {record.runtime_day_id}"
            ) from error
        return ShadowRuntimeDay(record.runtime_day_id, record.model_id)


class ShadowRuntimeDayVerifier:
    """Evaluator-owned verifier for runtime facts, calendar membership, and reports."""

    def __init__(self, root: Path, *, verification_key: bytes) -> None:
        if not isinstance(verification_key, bytes) or not verification_key:
            raise ValueError("shadow runtime verification_key must be non-empty bytes")
        self._root = root
        self._verification_key = verification_key

    @classmethod
    def from_environment(cls, root: Path) -> ShadowRuntimeDayVerifier:
        secret = os.environ.get(_SHADOW_RUNTIME_KEY_ENV, "")
        if not secret:
            raise ValueError(f"shadow runtime evidence requires {_SHADOW_RUNTIME_KEY_ENV}")
        return cls(root, verification_key=secret.encode("utf-8"))

    def verify(self, day: ShadowRuntimeDay) -> _ShadowRuntimeDayRecord:
        if not isinstance(day, ShadowRuntimeDay):
            raise ValueError("shadow runtime day is invalid")
        path = self._root / f"{day.runtime_day_id}.json"
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("shadow runtime day cannot be loaded") from error
        payload = document.get("payload") if isinstance(document, dict) else None
        signature = document.get("signature") if isinstance(document, dict) else None
        if (
            not isinstance(payload, dict)
            or document.get("runtime_day_id") != day.runtime_day_id
            or not isinstance(signature, str)
            or _shadow_runtime_day_id(payload) != day.runtime_day_id
            or not hmac.compare_digest(
                signature,
                _shadow_runtime_day_signature(
                    self._verification_key, day.runtime_day_id, payload
                ),
            )
        ):
            raise ValueError("shadow runtime day signature verification failed")
        record = _shadow_runtime_day_from_payload(day.runtime_day_id, payload)
        if record.model_id != day.model_id:
            raise ValueError("shadow runtime day does not match its immutable record")
        return record

    def records_for_model(self, model_id: str) -> tuple[_ShadowRuntimeDayRecord, ...]:
        """Load every signed attempt, so a caller cannot hide blocked or PSI-error days."""
        records: list[_ShadowRuntimeDayRecord] = []
        for path in sorted(self._root.glob("*.json")):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
                payload = document.get("payload") if isinstance(document, dict) else None
                runtime_day_id = (
                    document.get("runtime_day_id") if isinstance(document, dict) else None
                )
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("shadow runtime day cannot be loaded") from error
            if (
                not isinstance(payload, dict)
                or not isinstance(runtime_day_id, str)
                or payload.get("model_id") != model_id
            ):
                continue
            records.append(self.verify(ShadowRuntimeDay(runtime_day_id, model_id)))
        return tuple(records)

@dataclass(frozen=True)
class ShadowQualification:
    model_id: str
    valid_day_count: int
    status: ShadowQualificationStatus
    max_consecutive_psi_errors: int
    valid_signal_dates: tuple[date, ...]
    invalid_signal_dates: tuple[date, ...]
    valid_report_ids: tuple[str, ...]
    valid_runtime_day_ids: tuple[str, ...]
    invalid_runtime_day_ids: tuple[str, ...]


@dataclass(frozen=True)
class ShadowEvidence:
    evidence_id: str
    payload: Mapping[str, object]
    signature: str

    def verify(self) -> None:
        _validate_shadow_evidence_payload(self.payload)
        expected = hashlib.sha256(_canonical_json(self.payload).encode("utf-8")).hexdigest()
        if self.evidence_id != expected:
            raise ValueError("shadow evidence content does not match evidence_id")

    def to_json(self) -> str:
        return _canonical_json(
            {"evidence_id": self.evidence_id, "payload": self.payload, "signature": self.signature}
        ) + "\n"


def evaluate_shadow_qualification(
    model_id: str,
    *,
    runtime_root: Path,
    trading_dates: Sequence[date],
) -> ShadowQualification:
    """Count only verified, matching SHADOW days; never infer performance from labels or returns."""
    if not _MODEL_ID.fullmatch(model_id):
        raise ValueError("shadow qualification model_id is invalid")
    calendar = _trading_calendar(trading_dates)
    verifier = ShadowRuntimeDayVerifier.from_environment(runtime_root)
    latest_by_date: dict[date, _ShadowRuntimeDayRecord] = {}
    for record in verifier.records_for_model(model_id):
        latest = latest_by_date.get(record.signal_date)
        if latest is None or (record.recorded_at, record.runtime_day_id) > (
            latest.recorded_at,
            latest.runtime_day_id,
        ):
            latest_by_date[record.signal_date] = record
    verified_days = tuple(latest_by_date[value] for value in sorted(latest_by_date))
    dates = tuple(day.signal_date for day in verified_days)
    if dates != tuple(sorted(dates)) or len(set(dates)) != len(dates):
        raise ValueError("shadow days must have unique ascending signal dates")
    valid_days = tuple(
        day
        for day in verified_days
        if _is_valid_shadow_day(model_id, day, calendar)
    )
    valid_dates = tuple(day.signal_date for day in valid_days)
    invalid_dates = tuple(day.signal_date for day in verified_days if day not in valid_days)
    maximum = 0
    current = 0
    for day in verified_days:
        if day in valid_days and day.has_psi_error:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    if maximum >= 3:
        status = ShadowQualificationStatus.REVIEW_REQUIRED
    elif len(valid_days) >= 60:
        status = ShadowQualificationStatus.READY_FOR_REVIEW
    else:
        status = ShadowQualificationStatus.INCOMPLETE
    return ShadowQualification(
        model_id=model_id,
        valid_day_count=len(valid_days),
        status=status,
        max_consecutive_psi_errors=maximum,
        valid_signal_dates=valid_dates,
        invalid_signal_dates=invalid_dates,
        valid_report_ids=tuple(day.report.report_id for day in valid_days),
        valid_runtime_day_ids=tuple(day.runtime_day_id for day in valid_days),
        invalid_runtime_day_ids=tuple(
            day.runtime_day_id for day in verified_days if day not in valid_days
        ),
    )


def build_shadow_evidence(
    model_id: str,
    *,
    runtime_root: Path,
    evidence_root: Path,
    trading_dates: Sequence[date],
) -> ShadowEvidence:
    """Create evidence for later review without automatically changing any model lifecycle state."""
    qualification = evaluate_shadow_qualification(
        model_id,
        runtime_root=runtime_root,
        trading_dates=trading_dates,
    )
    return ShadowEvidenceStore.from_environment(evidence_root).issue(qualification)


class ShadowEvidenceStore:
    """Sign qualification evidence with the evaluator-owned evidence trust root."""

    def __init__(self, root: Path, *, signing_key: bytes) -> None:
        if not isinstance(signing_key, bytes) or not signing_key:
            raise ValueError("shadow evidence signing_key must be non-empty bytes")
        self._root = root
        self._signing_key = signing_key

    @classmethod
    def from_environment(cls, root: Path) -> ShadowEvidenceStore:
        secret = os.environ.get(_SHADOW_EVIDENCE_KEY_ENV, "")
        if not secret:
            raise ValueError(f"shadow evidence requires {_SHADOW_EVIDENCE_KEY_ENV}")
        return cls(root, signing_key=secret.encode("utf-8"))

    def issue(self, qualification: ShadowQualification) -> ShadowEvidence:
        if qualification.status is not ShadowQualificationStatus.READY_FOR_REVIEW:
            raise ValueError("shadow evidence requires READY_FOR_REVIEW qualification")
        payload: dict[str, object] = {
            "invalid_signal_dates": [
                value.isoformat() for value in qualification.invalid_signal_dates
            ],
            "lifecycle_transition": "NONE",
            "max_consecutive_psi_errors": qualification.max_consecutive_psi_errors,
            "model_id": qualification.model_id,
            "observed_day_count": len(qualification.valid_signal_dates)
            + len(qualification.invalid_signal_dates),
            "qualification_status": qualification.status.value,
            "schema_version": 1,
            "successful_day_rate": len(qualification.valid_signal_dates)
            / (
                len(qualification.valid_signal_dates)
                + len(qualification.invalid_signal_dates)
            ),
            "valid_day_count": qualification.valid_day_count,
            "valid_report_ids": list(qualification.valid_report_ids),
            "valid_runtime_day_ids": list(qualification.valid_runtime_day_ids),
            "invalid_runtime_day_ids": list(qualification.invalid_runtime_day_ids),
            "valid_signal_dates": [value.isoformat() for value in qualification.valid_signal_dates],
        }
        _validate_shadow_evidence_payload(payload)
        evidence_id = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        evidence = ShadowEvidence(
            evidence_id,
            payload,
            _shadow_evidence_signature(self._signing_key, evidence_id, payload),
        )
        destination = self._root / ".issued" / f"{evidence.evidence_id}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            _write_exclusive(destination, evidence.to_json())
        except FileExistsError as error:
            raise FileExistsError(
                f"shadow evidence is immutable: {evidence.evidence_id}"
            ) from error
        return evidence


class ShadowEvidenceVerifier:
    """Verify signed qualification evidence before it can be consumed downstream."""

    def __init__(self, root: Path, *, verification_key: bytes) -> None:
        if not isinstance(verification_key, bytes) or not verification_key:
            raise ValueError("shadow evidence verification_key must be non-empty bytes")
        self._root = root
        self._verification_key = verification_key

    @classmethod
    def from_environment(cls, root: Path) -> ShadowEvidenceVerifier:
        secret = os.environ.get(_SHADOW_EVIDENCE_KEY_ENV, "")
        if not secret:
            raise ValueError(f"shadow evidence requires {_SHADOW_EVIDENCE_KEY_ENV}")
        return cls(root, verification_key=secret.encode("utf-8"))

    def verify(self, evidence: ShadowEvidence) -> None:
        if not isinstance(evidence, ShadowEvidence):
            raise ValueError("shadow evidence is invalid")
        evidence.verify()
        if not hmac.compare_digest(
            evidence.signature,
            _shadow_evidence_signature(
                self._verification_key, evidence.evidence_id, evidence.payload
            ),
        ):
            raise ValueError("shadow evidence signature verification failed")
        path = self._root / ".issued" / f"{evidence.evidence_id}.json"
        try:
            issued = path.read_text(encoding="utf-8")
        except OSError as error:
            raise ValueError("shadow evidence has not been issued") from error
        if issued != evidence.to_json():
            raise ValueError("shadow evidence does not match its issued record")


def write_shadow_evidence(evidence: ShadowEvidence, root: Path) -> Path:
    """Publish one immutable shadow_evidence.json under its model-scoped evidence directory."""
    ShadowEvidenceVerifier.from_environment(root).verify(evidence)
    model_id = evidence.payload["model_id"]
    if not isinstance(model_id, str):
        raise ValueError("shadow evidence model_id is invalid")
    destination = root / model_id.removeprefix("sha256:") / "shadow_evidence.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_exclusive(destination, evidence.to_json())
    except FileExistsError as error:
        raise FileExistsError(f"shadow evidence is immutable: {evidence.evidence_id}") from error
    return destination


def _is_valid_shadow_day(
    model_id: str,
    day: _ShadowRuntimeDayRecord,
    trading_dates: tuple[date, ...],
) -> bool:
    if day.model_id != model_id:
        return False
    if (
        day.model_status != "SHADOW"
        or day.shadow_pointer_model_id != model_id
        or day.shadow_pointer_model_version != day.model_version
    ):
        return False
    if day.signal_date not in trading_dates or day.has_blocker:
        return False
    try:
        day.report.verify()
    except ValueError:
        return False
    payload = day.report.payload
    return (
        payload["model_id"] == model_id
        and payload["model_version"] == day.model_version
        and payload["signal_date"] == day.signal_date.isoformat()
        and payload["pipeline_status"] == "SUCCESS"
        and payload["prediction_published"] is True
    )


def _has_psi_error(report: ShadowDailyReport) -> bool:
    features = report.payload["psi_error_features"]
    return isinstance(features, list) and bool(features)


def _shadow_runtime_day_id(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _shadow_runtime_day_signature(
    signing_key: bytes,
    runtime_day_id: str,
    payload: Mapping[str, object],
) -> str:
    signed = {"payload": payload, "runtime_day_id": runtime_day_id}
    return hmac.new(
        signing_key, _canonical_json(signed).encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _shadow_runtime_day_from_payload(
    runtime_day_id: str,
    payload: Mapping[str, object],
) -> _ShadowRuntimeDayRecord:
    expected = {
        "has_blocker",
        "has_psi_error",
        "model_id",
        "model_status",
        "model_version",
        "recorded_at",
        "report_id",
        "report_payload",
        "shadow_pointer_model_id",
        "shadow_pointer_model_version",
        "signal_date",
    }
    if set(payload) != expected:
        raise ValueError("shadow runtime day record is malformed")
    model_id = payload.get("model_id")
    model_version = payload.get("model_version")
    signal_date = payload.get("signal_date")
    recorded_at = payload.get("recorded_at")
    model_status = payload.get("model_status")
    pointer_model_id = payload.get("shadow_pointer_model_id")
    pointer_model_version = payload.get("shadow_pointer_model_version")
    report_id = payload.get("report_id")
    report_payload = payload.get("report_payload")
    boolean_values = ("has_blocker", "has_psi_error")
    if (
        not isinstance(model_id, str)
        or not _MODEL_ID.fullmatch(model_id)
        or isinstance(model_version, bool)
        or not isinstance(model_version, int)
        or model_version < 1
        or not isinstance(signal_date, str)
        or not isinstance(recorded_at, str)
        or not isinstance(model_status, str)
        or not model_status.strip()
        or (
            pointer_model_id is not None
            and (not isinstance(pointer_model_id, str) or not _MODEL_ID.fullmatch(pointer_model_id))
        )
        or (
            pointer_model_version is not None
            and (
                isinstance(pointer_model_version, bool)
                or not isinstance(pointer_model_version, int)
                or pointer_model_version < 1
            )
        )
        or not isinstance(report_id, str)
        or not _EVIDENCE_HASH.fullmatch(report_id)
        or not isinstance(report_payload, dict)
        or not all(isinstance(payload.get(name), bool) for name in boolean_values)
    ):
        raise ValueError("shadow runtime day record is malformed")
    try:
        parsed_date = date.fromisoformat(signal_date)
        parsed_recorded_at = _as_utc(datetime.fromisoformat(recorded_at))
    except ValueError as error:
        raise ValueError("shadow runtime day record is malformed") from error
    report = ShadowDailyReport(report_id, report_payload)
    report.verify()
    if (
        report.payload["model_id"] != model_id
        or report.payload["model_version"] != model_version
        or report.payload["signal_date"] != signal_date
        or _has_psi_error(report) is not payload["has_psi_error"]
    ):
        raise ValueError("shadow runtime day record is malformed")
    return _ShadowRuntimeDayRecord(
        runtime_day_id=runtime_day_id,
        model_id=model_id,
        model_version=model_version,
        signal_date=parsed_date,
        recorded_at=parsed_recorded_at,
        model_status=model_status,
        shadow_pointer_model_id=pointer_model_id,
        shadow_pointer_model_version=pointer_model_version,
        has_blocker=cast(bool, payload["has_blocker"]),
        has_psi_error=payload["has_psi_error"],
        report=report,
    )


def _shadow_evidence_signature(
    signing_key: bytes,
    evidence_id: str,
    payload: Mapping[str, object],
) -> str:
    signed = {"evidence_id": evidence_id, "payload": payload}
    return hmac.new(
        signing_key, _canonical_json(signed).encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _validate_shadow_evidence_payload(payload: Mapping[str, object]) -> None:
    expected = {
        "invalid_signal_dates",
        "invalid_runtime_day_ids",
        "lifecycle_transition",
        "max_consecutive_psi_errors",
        "model_id",
        "observed_day_count",
        "qualification_status",
        "schema_version",
        "successful_day_rate",
        "valid_day_count",
        "valid_report_ids",
        "valid_runtime_day_ids",
        "valid_signal_dates",
    }
    if set(payload) != expected:
        raise ValueError("shadow evidence payload is invalid")
    model_id = payload.get("model_id")
    if not isinstance(model_id, str) or not _MODEL_ID.fullmatch(model_id):
        raise ValueError("shadow evidence model_id is invalid")
    if payload.get("schema_version") != 1 or payload.get("lifecycle_transition") != "NONE":
        raise ValueError("shadow evidence schema or transition is invalid")
    if payload.get("qualification_status") != ShadowQualificationStatus.READY_FOR_REVIEW:
        raise ValueError("shadow evidence qualification status is invalid")
    count = payload.get("valid_day_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 60:
        raise ValueError("shadow evidence valid_day_count is invalid")
    observed_count = payload.get("observed_day_count")
    success_rate = payload.get("successful_day_rate")
    if (
        isinstance(observed_count, bool)
        or not isinstance(observed_count, int)
        or observed_count < count
        or isinstance(success_rate, bool)
        or not isinstance(success_rate, int | float)
        or not math.isfinite(float(success_rate))
        or not 0 <= success_rate <= 1
    ):
        raise ValueError("shadow evidence observed-day metrics are invalid")
    max_errors = payload.get("max_consecutive_psi_errors")
    if isinstance(max_errors, bool) or not isinstance(max_errors, int) or max_errors < 0:
        raise ValueError("shadow evidence max_consecutive_psi_errors is invalid")
    if max_errors >= 3:
        raise ValueError("shadow evidence cannot be generated while review is required")
    valid_dates = _date_list(payload.get("valid_signal_dates"), "valid_signal_dates")
    invalid_dates = _date_list(payload.get("invalid_signal_dates"), "invalid_signal_dates")
    report_ids = payload.get("valid_report_ids")
    valid_runtime_day_ids = payload.get("valid_runtime_day_ids")
    invalid_runtime_day_ids = payload.get("invalid_runtime_day_ids")
    if (
        len(valid_dates) != count
        or observed_count != count + len(invalid_dates)
        or success_rate != count / observed_count
        or len(set(valid_dates)) != len(valid_dates)
        or set(valid_dates).intersection(invalid_dates)
        or not isinstance(report_ids, list)
        or len(report_ids) != count
        or not all(
            isinstance(value, str) and _EVIDENCE_HASH.fullmatch(value)
            for value in report_ids
        )
        or not isinstance(valid_runtime_day_ids, list)
        or len(valid_runtime_day_ids) != count
        or not all(
            isinstance(value, str) and _EVIDENCE_HASH.fullmatch(value)
            for value in valid_runtime_day_ids
        )
        or not isinstance(invalid_runtime_day_ids, list)
        or len(invalid_runtime_day_ids) != len(invalid_dates)
        or not all(
            isinstance(value, str) and _EVIDENCE_HASH.fullmatch(value)
            for value in invalid_runtime_day_ids
        )
        or set(valid_runtime_day_ids).intersection(invalid_runtime_day_ids)
    ):
        raise ValueError("shadow evidence dates or report IDs are invalid")


def _date_list(value: object, name: str) -> tuple[date, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"shadow evidence {name} is invalid")
    try:
        parsed = tuple(date.fromisoformat(item) for item in value)
    except ValueError as error:
        raise ValueError(f"shadow evidence {name} is invalid") from error
    if parsed != tuple(sorted(parsed)) or len(set(parsed)) != len(parsed):
        raise ValueError(f"shadow evidence {name} is invalid")
    return parsed


def _trading_calendar(values: Sequence[date]) -> tuple[date, ...]:
    calendar = tuple(values)
    if not calendar or any(not isinstance(value, date) for value in calendar):
        raise ValueError("shadow qualification requires trading calendar dates")
    if calendar != tuple(sorted(calendar)) or len(set(calendar)) != len(calendar):
        raise ValueError("shadow trading calendar dates must be strictly increasing")
    return calendar


def _shadow_approval_id(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _shadow_approval_signature(
    signing_key: bytes,
    approval_id: str,
    payload: Mapping[str, object],
) -> str:
    signed = {"approval_id": approval_id, "payload": payload}
    return hmac.new(
        signing_key, _canonical_json(signed).encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _shadow_approval_from_payload(
    approval_id: str,
    payload: Mapping[str, object],
) -> _ShadowDeploymentApprovalRecord:
    expected = {
        "actor",
        "approved_at",
        "artifact_bundle_hash",
        "inference_contract_hash",
        "model_id",
        "rollback_plan_hash",
        "v2test_drill_hash",
        "walk_forward_evidence_hash",
    }
    if set(payload) != expected:
        raise ValueError("shadow deployment approval record is malformed")
    actor = payload.get("actor")
    approved_at = payload.get("approved_at")
    model_id = payload.get("model_id")
    walk_forward_evidence_hash = _evidence_hash_value(payload, "walk_forward_evidence_hash")
    artifact_bundle_hash = _evidence_hash_value(payload, "artifact_bundle_hash")
    inference_contract_hash = _evidence_hash_value(payload, "inference_contract_hash")
    v2test_drill_hash = _evidence_hash_value(payload, "v2test_drill_hash")
    rollback_plan_hash = _evidence_hash_value(payload, "rollback_plan_hash")
    if (
        not isinstance(actor, str)
        or not actor.strip()
        or not isinstance(approved_at, str)
        or not isinstance(model_id, str)
        or not _MODEL_ID.fullmatch(model_id)
    ):
        raise ValueError("shadow deployment approval record is malformed")
    try:
        _as_utc(datetime.fromisoformat(approved_at))
    except ValueError as error:
        raise ValueError("shadow deployment approval record is malformed") from error
    return _ShadowDeploymentApprovalRecord(
        approval_id=approval_id,
        model_id=model_id,
        actor=actor,
        approved_at=approved_at,
        walk_forward_evidence_hash=walk_forward_evidence_hash,
        artifact_bundle_hash=artifact_bundle_hash,
        inference_contract_hash=inference_contract_hash,
        v2test_drill_hash=v2test_drill_hash,
        rollback_plan_hash=rollback_plan_hash,
    )


def _evidence_hash_value(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not _EVIDENCE_HASH.fullmatch(value):
        raise ValueError("shadow deployment approval record is malformed")
    return value


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("shadow approval timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _timestamp(value: datetime) -> str:
    return _as_utc(value).isoformat()


def _write_exclusive(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
