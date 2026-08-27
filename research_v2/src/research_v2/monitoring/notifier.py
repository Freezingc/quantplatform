"""Persistent, deduplicated notification audit independent of prediction publication."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path


class AlertSeverity(StrEnum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    RESOLVED = "RESOLVED"


class NotificationState(StrEnum):
    PENDING = "PENDING"
    DELIVERED = "DELIVERED"
    DEDUPLICATED = "DEDUPLICATED"
    DELIVERY_ERROR = "DELIVERY_ERROR"


@dataclass(frozen=True)
class NotificationEvent:
    delivery_id: str
    alert_key: str
    signal_date: date
    severity: AlertSeverity
    payload: Mapping[str, object]


@dataclass(frozen=True)
class NotificationRecord:
    alert_key: str
    signal_date: date
    severity: AlertSeverity
    state: NotificationState
    occurred_at: str
    error_type: str | None = None


@dataclass(frozen=True)
class _PendingDelivery:
    event: NotificationEvent
    occurred_at: str


def _utc_now() -> datetime:
    return datetime.now(UTC)


class AuditNotifier:
    """Persist successful deliveries as the single source of audit, dedupe, and active state."""

    def __init__(
        self,
        *,
        audit_root: Path,
        transport: Callable[[NotificationEvent], None] | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._root = audit_root
        self._transport = transport or (lambda _: None)
        self._clock = clock
        for path in (self._delivered_root, self._lock_root, self._pending_root, self._record_root):
            path.mkdir(parents=True, exist_ok=True)

    def notify(
        self,
        *,
        alert_key: str,
        signal_date: date,
        severity: AlertSeverity,
        payload: Mapping[str, object],
    ) -> NotificationRecord:
        if severity is AlertSeverity.RESOLVED:
            raise ValueError("use resolve for recovered alerts")
        return self._dispatch(alert_key, signal_date, severity, payload, requires_active=False)

    def resolve(
        self,
        *,
        alert_key: str,
        signal_date: date,
        payload: Mapping[str, object],
    ) -> NotificationRecord:
        return self._dispatch(
            alert_key,
            signal_date,
            AlertSeverity.RESOLVED,
            payload,
            requires_active=True,
        )

    def audit(self) -> tuple[NotificationRecord, ...]:
        paths = (
            *self._delivered_root.glob("*.json"),
            *self._record_root.glob("*.json"),
        )
        records = [_load_record(path) for path in paths]
        for path in self._pending_root.glob("*.json"):
            pending = _load_pending_delivery(path)
            if self._delivered_path(
                pending.event.alert_key,
                pending.event.signal_date,
                pending.event.severity,
            ).exists():
                continue
            records.append(
                NotificationRecord(
                    pending.event.alert_key,
                    pending.event.signal_date,
                    pending.event.severity,
                    NotificationState.PENDING,
                    pending.occurred_at,
                )
            )
        return tuple(sorted(records, key=lambda record: (record.occurred_at, record.alert_key)))

    def drain_pending(self) -> tuple[NotificationRecord, ...]:
        """Replay persistent unsent events with their original idempotency keys after restart."""
        results: list[NotificationRecord] = []
        for path in sorted(self._pending_root.glob("*.json")):
            pending = _load_pending_delivery(path)
            event = pending.event
            with _exclusive_lock(self._lock_path(event.alert_key)):
                delivered_path = self._delivered_path(
                    event.alert_key,
                    event.signal_date,
                    event.severity,
                )
                if delivered_path.exists():
                    self._validated_marker(
                        delivered_path,
                        event.alert_key,
                        event.signal_date,
                        event.severity,
                        NotificationState.DELIVERED,
                    )
                    path.unlink(missing_ok=True)
                    continue
                results.append(self._deliver_pending(pending, path))
        return tuple(results)

    @property
    def _delivered_root(self) -> Path:
        return self._root / "delivered"

    @property
    def _lock_root(self) -> Path:
        return self._root / "locks"

    @property
    def _pending_root(self) -> Path:
        return self._root / "pending"

    @property
    def _record_root(self) -> Path:
        return self._root / "records"

    def _dispatch(
        self,
        alert_key: str,
        signal_date: date,
        severity: AlertSeverity,
        payload: Mapping[str, object],
        *,
        requires_active: bool,
    ) -> NotificationRecord:
        _require_alert_identity(alert_key, signal_date, severity)
        _validate_safe_payload(payload)
        with _exclusive_lock(self._lock_path(alert_key)):
            if requires_active and not self._is_active(alert_key):
                return self._record(
                    alert_key,
                    signal_date,
                    severity,
                    NotificationState.DEDUPLICATED,
                )
            delivered_path = self._delivered_path(alert_key, signal_date, severity)
            if delivered_path.exists():
                self._validated_marker(
                    delivered_path,
                    alert_key,
                    signal_date,
                    severity,
                    NotificationState.DELIVERED,
                )
                return self._record(
                    alert_key,
                    signal_date,
                    severity,
                    NotificationState.DEDUPLICATED,
                )
            pending_path = self._pending_path(alert_key, signal_date, severity)
            if pending_path.exists():
                pending = _load_pending_delivery(pending_path)
                if pending.event != _notification_event(alert_key, signal_date, severity, payload):
                    raise ValueError(
                        "notification pending event does not match the requested payload"
                    )
            else:
                pending = _PendingDelivery(
                    _notification_event(alert_key, signal_date, severity, payload),
                    _timestamp(self._clock()),
                )
                _write_atomic(pending_path, _pending_json(pending))
            return self._deliver_pending(pending, pending_path)

    def _deliver_pending(
        self,
        pending: _PendingDelivery,
        pending_path: Path,
    ) -> NotificationRecord:
        event = pending.event
        try:
            self._transport(event)
        except Exception as error:
            return self._record(
                event.alert_key,
                event.signal_date,
                event.severity,
                NotificationState.DELIVERY_ERROR,
                error_type=type(error).__name__,
            )
        record = NotificationRecord(
            event.alert_key,
            event.signal_date,
            event.severity,
            NotificationState.DELIVERED,
            _timestamp(self._clock()),
        )
        _write_atomic(
            self._delivered_path(event.alert_key, event.signal_date, event.severity),
            _record_json(record),
        )
        pending_path.unlink(missing_ok=True)
        return record

    def _is_active(self, alert_key: str) -> bool:
        records: list[NotificationRecord] = []
        for path in self._delivered_root.glob("*.json"):
            record = _load_record(path)
            if record.alert_key == alert_key:
                records.append(record)
        if not records:
            return False
        latest = max(
            records,
            key=lambda record: (record.occurred_at, record.signal_date.isoformat()),
        )
        return latest.severity is not AlertSeverity.RESOLVED

    def _record(
        self,
        alert_key: str,
        signal_date: date,
        severity: AlertSeverity,
        state: NotificationState,
        *,
        error_type: str | None = None,
    ) -> NotificationRecord:
        record = NotificationRecord(
            alert_key,
            signal_date,
            severity,
            state,
            _timestamp(self._clock()),
            error_type,
        )
        _write_exclusive(
            self._record_root / f"{uuid.uuid4().hex}.json",
            _record_json(record),
        )
        return record

    def _lock_path(self, alert_key: str) -> Path:
        return self._lock_root / f"{_digest(alert_key)}.lock"

    def _delivered_path(
        self,
        alert_key: str,
        signal_date: date,
        severity: AlertSeverity,
    ) -> Path:
        digest = _digest(alert_key, signal_date.isoformat(), severity)
        return self._delivered_root / f"{digest}.json"

    def _pending_path(
        self,
        alert_key: str,
        signal_date: date,
        severity: AlertSeverity,
    ) -> Path:
        digest = _digest(alert_key, signal_date.isoformat(), severity)
        return self._pending_root / f"{digest}.json"

    @staticmethod
    def _validated_marker(
        path: Path,
        alert_key: str,
        signal_date: date,
        severity: AlertSeverity,
        expected_state: NotificationState,
    ) -> NotificationRecord:
        record = _load_record(path)
        if (
            record.alert_key != alert_key
            or record.signal_date != signal_date
            or record.severity is not severity
            or record.state is not expected_state
        ):
            raise ValueError("notification delivery marker is malformed")
        return record


def _record_payload(record: NotificationRecord) -> dict[str, str | None]:
    return {
        "alert_key": record.alert_key,
        "error_type": record.error_type,
        "occurred_at": record.occurred_at,
        "severity": record.severity,
        "signal_date": record.signal_date.isoformat(),
        "state": record.state,
    }


def _record_json(record: NotificationRecord) -> str:
    return json.dumps(_record_payload(record), separators=(",", ":"), sort_keys=True) + "\n"


def _notification_event(
    alert_key: str,
    signal_date: date,
    severity: AlertSeverity,
    payload: Mapping[str, object],
) -> NotificationEvent:
    return NotificationEvent(
        _digest(alert_key, signal_date.isoformat(), severity),
        alert_key,
        signal_date,
        severity,
        _json_safe_mapping(payload),
    )


def _pending_json(pending: _PendingDelivery) -> str:
    event = pending.event
    document = {
        "delivery_id": event.delivery_id,
        "event": {
            "alert_key": event.alert_key,
            "payload": event.payload,
            "severity": event.severity,
            "signal_date": event.signal_date.isoformat(),
        },
        "occurred_at": pending.occurred_at,
    }
    return json.dumps(document, separators=(",", ":"), sort_keys=True) + "\n"


def _load_pending_delivery(path: Path) -> _PendingDelivery:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("notification pending record is malformed") from error
    if not isinstance(document, dict) or set(document) != {"delivery_id", "event", "occurred_at"}:
        raise ValueError("notification pending record is malformed")
    delivery_id = document.get("delivery_id")
    event = document.get("event")
    occurred_at = document.get("occurred_at")
    if (
        not isinstance(delivery_id, str)
        or not isinstance(event, dict)
        or not isinstance(occurred_at, str)
    ):
        raise ValueError("notification pending record is malformed")
    if set(event) != {"alert_key", "payload", "severity", "signal_date"}:
        raise ValueError("notification pending record is malformed")
    alert_key = event.get("alert_key")
    payload = event.get("payload")
    severity = event.get("severity")
    signal_date = event.get("signal_date")
    if (
        not isinstance(alert_key, str)
        or not isinstance(payload, dict)
        or severity not in set(AlertSeverity)
        or not isinstance(signal_date, str)
    ):
        raise ValueError("notification pending record is malformed")
    try:
        parsed_date = date.fromisoformat(signal_date)
        _as_utc(datetime.fromisoformat(occurred_at))
    except ValueError as error:
        raise ValueError("notification pending record is malformed") from error
    typed_severity = AlertSeverity(severity)
    _require_alert_identity(alert_key, parsed_date, typed_severity)
    _validate_safe_payload(payload)
    expected_delivery_id = _digest(alert_key, parsed_date.isoformat(), typed_severity)
    if delivery_id != expected_delivery_id:
        raise ValueError("notification pending record is malformed")
    return _PendingDelivery(
        NotificationEvent(
            delivery_id,
            alert_key,
            parsed_date,
            typed_severity,
            _json_safe_mapping(payload),
        ),
        occurred_at,
    )


def _load_record(path: Path) -> NotificationRecord:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("notification audit record is malformed") from error
    if not isinstance(payload, dict) or set(payload) != {
        "alert_key",
        "error_type",
        "occurred_at",
        "severity",
        "signal_date",
        "state",
    }:
        raise ValueError("notification audit record is malformed")
    alert_key = payload.get("alert_key")
    signal_date = payload.get("signal_date")
    severity = payload.get("severity")
    state = payload.get("state")
    occurred_at = payload.get("occurred_at")
    error_type = payload.get("error_type")
    if (
        not isinstance(alert_key, str)
        or not isinstance(signal_date, str)
        or severity not in set(AlertSeverity)
        or state not in set(NotificationState)
        or not isinstance(occurred_at, str)
        or (error_type is not None and not isinstance(error_type, str))
    ):
        raise ValueError("notification audit record is malformed")
    try:
        parsed_date = date.fromisoformat(signal_date)
        _as_utc(datetime.fromisoformat(occurred_at))
    except ValueError as error:
        raise ValueError("notification audit record is malformed") from error
    _require_alert_identity(alert_key, parsed_date, AlertSeverity(severity))
    return NotificationRecord(
        alert_key,
        parsed_date,
        AlertSeverity(severity),
        NotificationState(state),
        occurred_at,
        error_type,
    )


def _require_alert_identity(alert_key: str, signal_date: date, severity: AlertSeverity) -> None:
    if (
        not isinstance(alert_key, str)
        or not _ALERT_KEY_PATTERN.fullmatch(alert_key)
        or any(component not in _ALERT_KEY_COMPONENTS for component in alert_key.split(":"))
        or not isinstance(signal_date, date)
    ):
        raise ValueError("notification alert identity is invalid")
    if not isinstance(severity, AlertSeverity):
        raise ValueError("notification severity is invalid")


def _validate_safe_payload(payload: Mapping[str, object]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("notification payload must be a mapping")
    _validate_safe_value(payload)


def _json_safe_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return {str(key): _json_safe_value(nested) for key, nested in value.items()}


def _json_safe_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_safe_value(nested) for nested in value]
    return value


def _validate_safe_value(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str) or any(
                forbidden in key.lower()
                for forbidden in _SENSITIVE_KEY_PARTS
            ):
                raise ValueError("notification payload contains a sensitive field")
            _validate_safe_value(nested)
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for nested in value:
            _validate_safe_value(nested)
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise ValueError("notification payload must contain finite structured values")


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


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


def _write_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_exclusive(temporary, content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _digest(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("notification timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _timestamp(value: datetime) -> str:
    return _as_utc(value).isoformat()


_ALERT_KEY_PATTERN = re.compile(r"[a-z][a-z0-9_]*(?::[a-z][a-z0-9_]*){0,3}")
_ALERT_KEY_COMPONENTS = frozenset(
    {
        "coverage",
        "data_health",
        "delivery",
        "drift",
        "factor",
        "feature",
        "finite",
        "health",
        "missing",
        "notification",
        "pipeline",
        "psi",
        "publish",
        "quality",
        "score",
        "shadow",
        "universe",
        "watermark",
    }
)
_SENSITIVE_KEY_PARTS = (
    "account",
    "access_key",
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "email",
    "password",
    "passwd",
    "private_key",
    "secret",
    "token",
    "user",
)
