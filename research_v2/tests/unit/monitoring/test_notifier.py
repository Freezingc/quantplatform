from __future__ import annotations

import json
from datetime import date

import pytest

from research_v2.monitoring.notifier import (
    AlertSeverity,
    AuditNotifier,
    NotificationEvent,
    NotificationState,
)


def test_notifier_deduplicates_an_active_alert_and_records_a_single_recovery(
    tmp_path: object,
) -> None:
    delivered: list[tuple[str, AlertSeverity]] = []
    notifier = AuditNotifier(
        audit_root=tmp_path / "alerts",  # type: ignore[operator]
        transport=lambda event: delivered.append((event.alert_key, event.severity))
    )

    first = notifier.notify(
        alert_key="psi:quality",
        signal_date=date(2026, 8, 27),
        severity=AlertSeverity.ERROR,
        payload={"feature": "quality", "psi": 0.31},
    )
    duplicate = notifier.notify(
        alert_key="psi:quality",
        signal_date=date(2026, 8, 27),
        severity=AlertSeverity.ERROR,
        payload={"feature": "quality", "psi": 0.31},
    )
    resolved = notifier.resolve(
        alert_key="psi:quality",
        signal_date=date(2026, 8, 28),
        payload={"feature": "quality"},
    )
    resolved_again = notifier.resolve(
        alert_key="psi:quality",
        signal_date=date(2026, 8, 28),
        payload={"feature": "quality"},
    )

    assert first.state is NotificationState.DELIVERED
    assert duplicate.state is NotificationState.DEDUPLICATED
    assert resolved.state is NotificationState.DELIVERED
    assert resolved_again.state is NotificationState.DEDUPLICATED
    assert delivered == [
        ("psi:quality", AlertSeverity.ERROR),
        ("psi:quality", AlertSeverity.RESOLVED),
    ]


def test_notifier_records_delivery_failures_without_raising_to_the_pipeline(
    tmp_path: object,
) -> None:
    def fail(_: NotificationEvent) -> None:
        raise ConnectionError("transport unavailable")

    notifier = AuditNotifier(audit_root=tmp_path / "alerts", transport=fail)  # type: ignore[operator]

    result = notifier.notify(
        alert_key="psi:quality",
        signal_date=date(2026, 8, 27),
        severity=AlertSeverity.ERROR,
        payload={"feature": "quality"},
    )

    assert result.state is NotificationState.DELIVERY_ERROR
    assert result.error_type == "ConnectionError"
    assert {record.state for record in notifier.audit()} == {
        NotificationState.PENDING,
        NotificationState.DELIVERY_ERROR,
    }


def test_notifier_rejects_sensitive_payload_even_for_an_already_resolved_alert(
    tmp_path: object,
) -> None:
    notifier = AuditNotifier(audit_root=tmp_path / "alerts")  # type: ignore[operator]

    with pytest.raises(ValueError, match="sensitive"):
        notifier.resolve(
            alert_key="psi:quality",
            signal_date=date(2026, 8, 27),
            payload={"token": "must-not-be-emitted"},
        )


def test_notifier_retries_failures_persists_deduplication_and_avoids_false_recovery(
    tmp_path: object,
) -> None:
    delivered: list[AlertSeverity] = []
    delivery_ids: list[str] = []
    attempts = 0

    def flaky(event: NotificationEvent) -> None:
        nonlocal attempts
        attempts += 1
        delivery_ids.append(event.delivery_id)
        if attempts == 1:
            raise ConnectionError("temporary outage")
        delivered.append(event.severity)

    root = tmp_path / "alerts"  # type: ignore[operator]
    notifier = AuditNotifier(audit_root=root, transport=flaky)
    failed = notifier.notify(
        alert_key="psi:quality",
        signal_date=date(2026, 8, 27),
        severity=AlertSeverity.ERROR,
        payload={"feature": "quality"},
    )
    suppressed_resolution = notifier.resolve(
        alert_key="psi:quality",
        signal_date=date(2026, 8, 28),
        payload={"feature": "quality"},
    )
    retried = notifier.notify(
        alert_key="psi:quality",
        signal_date=date(2026, 8, 27),
        severity=AlertSeverity.ERROR,
        payload={"feature": "quality"},
    )
    restarted = AuditNotifier(audit_root=root, transport=flaky)
    duplicate = restarted.notify(
        alert_key="psi:quality",
        signal_date=date(2026, 8, 27),
        severity=AlertSeverity.ERROR,
        payload={"feature": "quality"},
    )

    assert failed.state is NotificationState.DELIVERY_ERROR
    assert suppressed_resolution.state is NotificationState.DEDUPLICATED
    assert retried.state is NotificationState.DELIVERED
    assert duplicate.state is NotificationState.DEDUPLICATED
    assert delivered == [AlertSeverity.ERROR]
    assert len(set(delivery_ids)) == 1


def test_notifier_drains_a_persisted_pending_outbox_after_restart(tmp_path: object) -> None:
    root = tmp_path / "alerts"  # type: ignore[operator]
    first = AuditNotifier(
        audit_root=root,
        transport=lambda _: (_ for _ in ()).throw(ConnectionError("outage")),
    )
    failed = first.notify(
        alert_key="psi:quality",
        signal_date=date(2026, 8, 27),
        severity=AlertSeverity.ERROR,
        payload={"feature": "quality"},
    )
    delivered: list[NotificationEvent] = []
    restarted = AuditNotifier(audit_root=root, transport=delivered.append)

    drained = restarted.drain_pending()

    assert failed.state is NotificationState.DELIVERY_ERROR
    assert [record.state for record in drained] == [NotificationState.DELIVERED]
    assert len(delivered) == 1
    assert delivered[0].payload == {"feature": "quality"}
    assert all(record.state is not NotificationState.PENDING for record in restarted.audit())


def test_notifier_rejects_sensitive_fields_nested_in_sequences(tmp_path: object) -> None:
    notifier = AuditNotifier(audit_root=tmp_path / "alerts")  # type: ignore[operator]

    with pytest.raises(ValueError, match="sensitive"):
        notifier.notify(
            alert_key="psi:quality",
            signal_date=date(2026, 8, 27),
            severity=AlertSeverity.ERROR,
            payload={"details": [{"api_token": "must-not-be-emitted"}]},
        )


def test_notifier_rejects_sensitive_information_in_the_alert_key(tmp_path: object) -> None:
    notifier = AuditNotifier(audit_root=tmp_path / "alerts")  # type: ignore[operator]

    with pytest.raises(ValueError, match="identity"):
        notifier.notify(
            alert_key="psi:access_key:abc123",
            signal_date=date(2026, 8, 27),
            severity=AlertSeverity.ERROR,
            payload={"feature": "quality"},
        )


def test_notifier_rejects_non_allowlisted_alert_key_components(tmp_path: object) -> None:
    notifier = AuditNotifier(audit_root=tmp_path / "alerts")  # type: ignore[operator]

    with pytest.raises(ValueError, match="identity"):
        notifier.notify(
            alert_key="psi:ssh_key:abc123",
            signal_date=date(2026, 8, 27),
            severity=AlertSeverity.ERROR,
            payload={"feature": "quality"},
        )


def test_notifier_derives_active_alert_state_from_persistent_success_records(
    tmp_path: object,
) -> None:
    root = tmp_path / "alerts"  # type: ignore[operator]
    first = AuditNotifier(audit_root=root)
    first.notify(
        alert_key="psi:quality",
        signal_date=date(2026, 8, 27),
        severity=AlertSeverity.ERROR,
        payload={"feature": "quality"},
    )
    delivered: list[AlertSeverity] = []
    restarted = AuditNotifier(
        audit_root=root,
        transport=lambda event: delivered.append(event.severity),
    )

    resolved = restarted.resolve(
        alert_key="psi:quality",
        signal_date=date(2026, 8, 28),
        payload={"feature": "quality"},
    )

    assert resolved.state is NotificationState.DELIVERED
    assert delivered == [AlertSeverity.RESOLVED]


def test_notifier_rejects_a_corrupt_persistent_delivery_marker(tmp_path: object) -> None:
    root = tmp_path / "alerts"  # type: ignore[operator]
    notifier = AuditNotifier(audit_root=root)
    notifier.notify(
        alert_key="psi:quality",
        signal_date=date(2026, 8, 27),
        severity=AlertSeverity.ERROR,
        payload={"feature": "quality"},
    )
    marker = next((root / "delivered").glob("*.json"))
    marker.write_text(json.dumps({"unexpected": "record"}), encoding="utf-8")

    with pytest.raises(ValueError, match="audit record"):
        notifier.notify(
            alert_key="psi:quality",
            signal_date=date(2026, 8, 27),
            severity=AlertSeverity.ERROR,
            payload={"feature": "quality"},
        )
