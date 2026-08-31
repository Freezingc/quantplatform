"""Population stability index calculations for non-blocking inference drift monitoring."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path

from research_v2.monitoring.policy import DEFAULT_MONITORING_POLICY, MonitoringPolicy


class DriftSeverity(StrEnum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


class DriftReviewStatus(StrEnum):
    CLEAR = "CLEAR"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


@dataclass(frozen=True)
class PsiResult:
    value: float
    severity: DriftSeverity


@dataclass(frozen=True)
class DistributionDriftReport:
    """Daily and complete rolling-window PSI for one monitored distribution."""

    daily_psi: PsiResult
    rolling_20d_psi: PsiResult | None


@dataclass(frozen=True)
class DailyDriftStatus:
    signal_date: date
    has_psi_error: bool
    is_valid_day: bool = True


@dataclass(frozen=True)
class DriftReviewReport:
    status: DriftReviewStatus
    consecutive_error_days: int
    review_required_since: date | None

    @property
    def model_changes_allowed(self) -> bool:
        return self.status is DriftReviewStatus.CLEAR


@dataclass(frozen=True)
class DriftReviewApproval:
    approval_id: str
    review_required_since: date


@dataclass(frozen=True)
class _ApprovalRecord:
    approval_id: str
    review_required_since: date
    acknowledged_on: date
    actor: str
    evidence_hash: str


class DriftReviewApprovalStore:
    """Issue and verify immutable, HMAC-signed approvals for drift-review release."""

    def __init__(self, root: Path, *, signing_key: bytes) -> None:
        if not isinstance(signing_key, bytes) or not signing_key:
            raise ValueError("drift approval signing_key must be non-empty bytes")
        self._root = root
        self._signing_key = signing_key

    def issue(
        self,
        *,
        review_required_since: date,
        actor: str,
        evidence_hash: str,
        acknowledged_on: date,
    ) -> DriftReviewApproval:
        payload = {
            "acknowledged_on": acknowledged_on.isoformat(),
            "actor": actor,
            "evidence_hash": evidence_hash,
            "review_required_since": review_required_since.isoformat(),
        }
        record = _approval_record_from_payload(_approval_id(payload), payload)
        self._root.mkdir(parents=True, exist_ok=True)
        document = {
            "approval_id": record.approval_id,
            "payload": payload,
            "signature": _approval_signature(self._signing_key, record.approval_id, payload),
        }
        path = self._root / f"{record.approval_id}.json"
        try:
            _write_exclusive(path, _canonical_json(document) + "\n")
        except FileExistsError as error:
            raise FileExistsError(f"drift approval is immutable: {record.approval_id}") from error
        return DriftReviewApproval(record.approval_id, record.review_required_since)

    def verify(self, approval: DriftReviewApproval) -> _ApprovalRecord:
        if not isinstance(approval, DriftReviewApproval):
            raise ValueError("drift review approval is invalid")
        path = self._root / f"{approval.approval_id}.json"
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("drift review approval cannot be loaded") from error
        payload = document.get("payload") if isinstance(document, dict) else None
        signature = document.get("signature") if isinstance(document, dict) else None
        if (
            not isinstance(payload, dict)
            or document.get("approval_id") != approval.approval_id
            or not isinstance(signature, str)
            or _approval_id(payload) != approval.approval_id
            or not hmac.compare_digest(
                signature,
                _approval_signature(self._signing_key, approval.approval_id, payload),
            )
        ):
            raise ValueError("drift review approval signature verification failed")
        record = _approval_record_from_payload(approval.approval_id, payload)
        if record.review_required_since != approval.review_required_since:
            raise ValueError("drift review approval does not match its immutable record")
        return record


def population_stability_index(
    reference_bins: Sequence[float],
    current_bins: Sequence[float],
    *,
    policy: MonitoringPolicy = DEFAULT_MONITORING_POLICY,
) -> PsiResult:
    """Calculate PSI from bin masses with epsilon smoothing and normalized distributions."""
    if len(reference_bins) != len(current_bins) or not reference_bins:
        raise ValueError("PSI inputs must have equal non-empty bins")
    reference = _normalize(reference_bins, policy.psi_epsilon)
    current = _normalize(current_bins, policy.psi_epsilon)
    value = sum(
        (actual - expected) * math.log(actual / expected)
        for expected, actual in zip(reference, current, strict=True)
    )
    if value < policy.psi_info_upper_bound:
        severity = DriftSeverity.INFO
    elif value <= policy.psi_warn_upper_bound:
        severity = DriftSeverity.WARN
    else:
        severity = DriftSeverity.ERROR
    return PsiResult(value=float(value), severity=severity)


def evaluate_distribution_drift(
    *,
    reference_bin_edges: Sequence[float],
    reference_bin_masses: Sequence[float],
    daily_values: Sequence[float],
    rolling_values_by_date: Mapping[date, Sequence[float]] | None = None,
    policy: MonitoringPolicy = DEFAULT_MONITORING_POLICY,
) -> DistributionDriftReport:
    """Calculate daily and, once available, latest-20-trading-day distribution PSI."""
    edges = _validate_distribution_edges(reference_bin_edges, reference_bin_masses)
    reference = _validate_reference_masses(reference_bin_masses)
    daily = population_stability_index(
        reference,
        _distribution_masses(daily_values, edges),
        policy=policy,
    )
    rolling: PsiResult | None = None
    if rolling_values_by_date is not None:
        if any(not isinstance(signal_date, date) for signal_date in rolling_values_by_date):
            raise ValueError("rolling distribution requires trading dates")
        for values in rolling_values_by_date.values():
            if isinstance(values, (str, bytes, bytearray)):
                raise ValueError("rolling distribution values must be numeric sequences")
        ordered_dates = sorted(rolling_values_by_date)
        if len(ordered_dates) >= 20:
            recent_dates = ordered_dates[-20:]
            recent_values = [
                value
                for signal_date in recent_dates
                for value in rolling_values_by_date[signal_date]
            ]
            rolling = population_stability_index(
                reference,
                _distribution_masses(recent_values, edges),
                policy=policy,
            )
    return DistributionDriftReport(daily_psi=daily, rolling_20d_psi=rolling)


def evaluate_drift_review(
    days: Sequence[DailyDriftStatus],
    *,
    trading_dates: Sequence[date],
    approval: DriftReviewApproval | None = None,
    approval_store: DriftReviewApprovalStore | None = None,
    policy: MonitoringPolicy = DEFAULT_MONITORING_POLICY,
) -> DriftReviewReport:
    """Require acknowledgement after consecutive PSI errors on calendar-adjacent valid days."""
    if any(not isinstance(day, DailyDriftStatus) for day in days):
        raise ValueError("drift review requires DailyDriftStatus records")
    if approval is not None and approval_store is None:
        raise ValueError("drift review approval requires approval_store")
    if approval is None and approval_store is not None:
        raise ValueError("drift review approval_store requires approval")
    ordered = tuple(days)
    calendar = tuple(trading_dates)
    if not calendar or any(not isinstance(value, date) for value in calendar):
        raise ValueError("drift review requires trading calendar dates")
    if any(current <= previous for previous, current in zip(calendar, calendar[1:], strict=False)):
        raise ValueError("trading calendar dates must be strictly increasing")
    if any(
        current.signal_date <= previous.signal_date
        for previous, current in zip(ordered, ordered[1:], strict=False)
    ):
        raise ValueError("drift review dates must be strictly increasing")
    calendar_index = {value: index for index, value in enumerate(calendar)}
    if any(day.signal_date not in calendar_index for day in ordered):
        raise ValueError("drift review contains a non-trading day observation")
    if any(
        calendar_index[current.signal_date] != calendar_index[previous.signal_date] + 1
        for previous, current in zip(ordered, ordered[1:], strict=False)
    ):
        raise ValueError("drift review observations must be contiguous trading days")
    streak = 0
    review_required_since: date | None = None
    for day in ordered:
        if day.is_valid_day and day.has_psi_error:
            streak += 1
            if streak >= policy.consecutive_psi_error_days:
                review_required_since = day.signal_date
        else:
            streak = 0
    approved = False
    if review_required_since is not None and approval is not None and approval_store is not None:
        record = approval_store.verify(approval)
        approved = (
            record.review_required_since == review_required_since
            and review_required_since <= record.acknowledged_on <= ordered[-1].signal_date
            and record.acknowledged_on in calendar_index
        )
    blocked = review_required_since is not None and not approved
    return DriftReviewReport(
        status=DriftReviewStatus.REVIEW_REQUIRED if blocked else DriftReviewStatus.CLEAR,
        consecutive_error_days=streak,
        review_required_since=review_required_since if blocked else None,
    )


def _normalize(values: Sequence[float], epsilon: float) -> tuple[float, ...]:
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("PSI bin masses must be finite and non-negative")
    smoothed = [float(value) + epsilon for value in values]
    total = sum(smoothed)
    return tuple(value / total for value in smoothed)


def _validate_distribution_edges(
    edges: Sequence[float],
    reference_masses: Sequence[float],
) -> tuple[float, ...]:
    if len(edges) != len(reference_masses) + 1 or len(edges) < 2:
        raise ValueError("distribution bin edges must be one longer than reference bins")
    normalized = tuple(float(value) for value in edges)
    if any(not math.isfinite(value) for value in normalized) or any(
        right <= left for left, right in zip(normalized, normalized[1:], strict=False)
    ):
        raise ValueError("distribution bin edges must be finite and increasing")
    return normalized


def _validate_reference_masses(values: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if not normalized or any(not math.isfinite(value) or value < 0 for value in normalized):
        raise ValueError("reference distribution bins must be finite and non-negative")
    if sum(normalized) <= 0:
        raise ValueError("reference distribution bins must contain mass")
    return normalized


def _distribution_masses(
    values: Sequence[float],
    edges: Sequence[float],
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("distribution values must be a numeric sequence")
    try:
        numeric = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError("distribution values must be numeric") from error
    finite = tuple(value for value in numeric if math.isfinite(value))
    bounded = tuple(max(edges[0], min(value, edges[-1])) for value in finite)
    counts = [0] * (len(edges) - 1)
    for value in bounded:
        index = next(
            (position for position, (left, right) in enumerate(
                zip(edges, edges[1:], strict=False)
            ) if left <= value < right),
            len(counts) - 1,
        )
        counts[index] += 1
    total = sum(counts)
    if total == 0:
        return tuple(0.0 for _ in counts)
    return tuple(count / total for count in counts)


def _approval_id(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _approval_signature(
    signing_key: bytes,
    approval_id: str,
    payload: Mapping[str, object],
) -> str:
    signed = {"approval_id": approval_id, "payload": payload}
    return hmac.new(signing_key, _canonical_json(signed).encode(), hashlib.sha256).hexdigest()


def _approval_record_from_payload(
    approval_id: str,
    payload: Mapping[str, object],
) -> _ApprovalRecord:
    if set(payload) != {"acknowledged_on", "actor", "evidence_hash", "review_required_since"}:
        raise ValueError("drift review approval record is malformed")
    acknowledged_on = payload.get("acknowledged_on")
    actor = payload.get("actor")
    evidence_hash = payload.get("evidence_hash")
    review_required_since = payload.get("review_required_since")
    if (
        not isinstance(acknowledged_on, str)
        or not isinstance(actor, str)
        or not isinstance(evidence_hash, str)
        or not isinstance(review_required_since, str)
        or not actor.strip()
        or not _is_sha256(evidence_hash)
    ):
        raise ValueError("drift review approval record is malformed")
    try:
        return _ApprovalRecord(
            approval_id,
            date.fromisoformat(review_required_since),
            date.fromisoformat(acknowledged_on),
            actor,
            evidence_hash,
        )
    except ValueError as error:
        raise ValueError("drift review approval record is malformed") from error


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


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
