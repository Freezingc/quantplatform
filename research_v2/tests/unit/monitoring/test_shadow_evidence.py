from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from research_v2.monitoring.shadow import (
    ShadowEvidenceVerifier,
    ShadowQualificationStatus,
    ShadowRuntimeDay,
    ShadowRuntimeDayStore,
    ShadowTradingCalendar,
    ShadowTradingCalendarStore,
    build_shadow_evidence,
    evaluate_shadow_qualification,
    write_shadow_evidence,
)
from research_v2.reporting.shadow import ShadowDailyReport, build_shadow_daily_report

_MODEL_ID = "sha256:" + "a" * 64
_RUNTIME_KEY = "trusted-runtime-key"
_EVIDENCE_KEY = "trusted-evidence-key"
_CALENDAR_KEY = "trusted-calendar-key"


@pytest.fixture(autouse=True)
def _trusted_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QRV2_SHADOW_RUNTIME_HMAC_KEY", _RUNTIME_KEY)
    monkeypatch.setenv("QRV2_SHADOW_EVIDENCE_HMAC_KEY", _EVIDENCE_KEY)
    monkeypatch.setenv("QRV2_SHADOW_CALENDAR_HMAC_KEY", _CALENDAR_KEY)


def _trading_dates(start: date, count: int) -> tuple[date, ...]:
    dates: list[date] = []
    current = start
    while len(dates) < count:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return tuple(dates)


def _calendar(root: Path, dates: tuple[date, ...]) -> ShadowTradingCalendar:
    return ShadowTradingCalendarStore(root, signing_key=_CALENDAR_KEY.encode()).issue(dates)


def _report(signal_date: date, *, psi_error: bool = False) -> ShadowDailyReport:
    return build_shadow_daily_report(
        signal_date=signal_date,
        model_id=_MODEL_ID,
        model_version=1,
        pipeline_status="SUCCESS",
        prediction_published=True,
        inference_manifest_id="sha256:" + "b" * 64,
        artifact_bundle_hash="c" * 64,
        source_watermark_id="d" * 64,
        latency_seconds=120.0,
        universe_size=1_200,
        feature_coverage={"quality": 0.98},
        score_health={"finite_coverage": 0.99, "unique_score_count": 800, "score_std": 0.15},
        seed_agreement={"minimum_spearman": 0.42},
        drift_alert_keys=(),
        event_ids=("e" * 64,),
        psi_error_features=("quality",) if psi_error else (),
    )


def _day(
    store: ShadowRuntimeDayStore,
    signal_date: date,
    *,
    model_status: str = "SHADOW",
    shadow_pointer_model_id: str | None = _MODEL_ID,
    shadow_pointer_model_version: int | None = 1,
    has_blocker: bool = False,
    psi_error: bool = False,
) -> ShadowRuntimeDay:
    return store.issue(
        _report(signal_date, psi_error=psi_error),
        model_status=model_status,
        shadow_pointer_model_id=shadow_pointer_model_id,
        shadow_pointer_model_version=shadow_pointer_model_version,
        has_blocker=has_blocker,
    )


def test_shadow_qualification_counts_only_matching_shadow_pointer_days(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    calendar_root = tmp_path / "calendar"
    store = ShadowRuntimeDayStore(root, signing_key=_RUNTIME_KEY.encode())
    dates = _trading_dates(date(2026, 1, 2), 60)
    calendar = _calendar(calendar_root, dates)
    for signal_date in dates:
        _day(store, signal_date)
    _day(store, date(2026, 3, 28))

    incomplete_root = tmp_path / "incomplete-runtime"
    incomplete_store = ShadowRuntimeDayStore(incomplete_root, signing_key=_RUNTIME_KEY.encode())
    tuple(_day(incomplete_store, signal_date) for signal_date in dates[:-1])
    _day(
        incomplete_store,
        dates[-1],
        shadow_pointer_model_id="sha256:" + "f" * 64,
    )

    ready = evaluate_shadow_qualification(
        _MODEL_ID, runtime_root=root, calendar=calendar, calendar_root=calendar_root
    )
    incomplete = evaluate_shadow_qualification(
        _MODEL_ID,
        runtime_root=incomplete_root,
        calendar=calendar,
        calendar_root=calendar_root,
    )

    assert ready.valid_day_count == 60
    assert ready.invalid_signal_dates == (date(2026, 3, 28),)
    assert ready.status is ShadowQualificationStatus.READY_FOR_REVIEW
    assert incomplete.valid_day_count == 59
    assert incomplete.status is ShadowQualificationStatus.INCOMPLETE


def test_shadow_evidence_requires_sixty_healthy_days_and_never_advances_lifecycle(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    evidence_root = tmp_path / "evidence"
    calendar_root = tmp_path / "calendar"
    store = ShadowRuntimeDayStore(runtime_root, signing_key=_RUNTIME_KEY.encode())
    dates = _trading_dates(date(2026, 1, 2), 60)
    calendar = _calendar(calendar_root, dates)
    for signal_date in dates:
        _day(store, signal_date)

    evidence = build_shadow_evidence(
        _MODEL_ID,
        runtime_root=runtime_root,
        evidence_root=evidence_root,
        calendar=calendar,
        calendar_root=calendar_root,
    )

    ShadowEvidenceVerifier.from_environment(evidence_root).verify(evidence)
    published = write_shadow_evidence(evidence, evidence_root)
    assert published.name == "shadow_evidence.json"
    assert evidence.evidence_id in published.read_text(encoding="utf-8")
    assert evidence.payload["qualification_status"] == "READY_FOR_REVIEW"
    assert evidence.payload["valid_day_count"] == 60
    assert evidence.payload["observed_day_count"] == 60
    assert evidence.payload["successful_day_rate"] == 1.0
    assert evidence.payload["lifecycle_transition"] == "NONE"
    with pytest.raises(ValueError, match="READY_FOR_REVIEW"):
        build_shadow_evidence(
            _MODEL_ID,
            runtime_root=tmp_path / "incomplete-runtime",
            evidence_root=evidence_root,
            calendar=calendar,
            calendar_root=calendar_root,
        )


def test_three_consecutive_valid_shadow_psi_errors_require_review_even_after_sixty_days(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime"
    calendar_root = tmp_path / "calendar"
    store = ShadowRuntimeDayStore(root, signing_key=_RUNTIME_KEY.encode())
    for index, signal_date in enumerate(_trading_dates(date(2026, 1, 2), 60)):
        _day(store, signal_date, psi_error=index in {10, 11, 12})

    dates = _trading_dates(date(2026, 1, 2), 60)
    qualification = evaluate_shadow_qualification(
        _MODEL_ID,
        runtime_root=root,
        calendar=_calendar(calendar_root, dates),
        calendar_root=calendar_root,
    )

    assert qualification.status is ShadowQualificationStatus.REVIEW_REQUIRED
    assert qualification.max_consecutive_psi_errors == 3


def test_shadow_qualification_uses_the_latest_signed_attempt_for_each_date(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime"
    calendar_root = tmp_path / "calendar"
    dates = _trading_dates(date(2026, 1, 2), 60)
    store = ShadowRuntimeDayStore(root, signing_key=_RUNTIME_KEY.encode())
    for signal_date in dates:
        _day(store, signal_date)
    later_attempt = ShadowRuntimeDayStore(
        root,
        signing_key=_RUNTIME_KEY.encode(),
        clock=lambda: datetime(2027, 1, 1, tzinfo=UTC),
    )
    _day(later_attempt, dates[10], has_blocker=True)

    qualification = evaluate_shadow_qualification(
        _MODEL_ID,
        runtime_root=root,
        calendar=_calendar(calendar_root, dates),
        calendar_root=calendar_root,
    )

    assert qualification.valid_day_count == 59
    assert qualification.invalid_signal_dates == (dates[10],)
    assert qualification.status is ShadowQualificationStatus.INCOMPLETE


def test_shadow_qualification_rejects_runtime_facts_signed_by_an_untrusted_writer(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime"
    calendar_root = tmp_path / "calendar"
    ShadowRuntimeDayStore(root, signing_key=b"attacker-key").issue(
        _report(date(2026, 1, 2)),
        model_status="SHADOW",
        shadow_pointer_model_id=_MODEL_ID,
        shadow_pointer_model_version=1,
        has_blocker=False,
    )

    with pytest.raises(ValueError, match="signature"):
        evaluate_shadow_qualification(
            _MODEL_ID,
            runtime_root=root,
            calendar=_calendar(calendar_root, (date(2026, 1, 2),)),
            calendar_root=calendar_root,
        )


def test_shadow_qualification_rejects_a_calendar_signed_by_an_untrusted_writer(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    calendar_root = tmp_path / "calendar"
    signal_date = date(2026, 1, 2)
    _day(
        ShadowRuntimeDayStore(runtime_root, signing_key=_RUNTIME_KEY.encode()), signal_date
    )
    forged = ShadowTradingCalendarStore(
        calendar_root, signing_key=b"attacker-key"
    ).issue((signal_date,))

    with pytest.raises(ValueError, match="signature"):
        evaluate_shadow_qualification(
            _MODEL_ID,
            runtime_root=runtime_root,
            calendar=forged,
            calendar_root=calendar_root,
        )


def test_non_trading_day_records_do_not_reset_a_consecutive_psi_error_streak(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime"
    calendar_root = tmp_path / "calendar"
    dates = _trading_dates(date(2026, 1, 2), 60)
    store = ShadowRuntimeDayStore(root, signing_key=_RUNTIME_KEY.encode())
    for index, signal_date in enumerate(dates):
        _day(store, signal_date, psi_error=index in {5, 6, 7})
    _day(store, date(2026, 1, 10))

    qualification = evaluate_shadow_qualification(
        _MODEL_ID,
        runtime_root=root,
        calendar=_calendar(calendar_root, dates),
        calendar_root=calendar_root,
    )

    assert qualification.status is ShadowQualificationStatus.REVIEW_REQUIRED
    assert qualification.max_consecutive_psi_errors == 3


def test_shadow_qualification_rejects_ambiguous_same_timestamp_retries(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    calendar_root = tmp_path / "calendar"
    signal_date = date(2026, 1, 2)

    def fixed_clock() -> datetime:
        return datetime(2026, 1, 3, tzinfo=UTC)

    first = ShadowRuntimeDayStore(root, signing_key=_RUNTIME_KEY.encode(), clock=fixed_clock)
    second = ShadowRuntimeDayStore(root, signing_key=_RUNTIME_KEY.encode(), clock=fixed_clock)
    _day(first, signal_date)
    _day(second, signal_date, has_blocker=True)

    with pytest.raises(ValueError, match="ambiguous"):
        evaluate_shadow_qualification(
            _MODEL_ID,
            runtime_root=root,
            calendar=_calendar(calendar_root, (signal_date,)),
            calendar_root=calendar_root,
        )


def test_rebuilding_identical_shadow_evidence_is_idempotent(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    evidence_root = tmp_path / "evidence"
    calendar_root = tmp_path / "calendar"
    dates = _trading_dates(date(2026, 1, 2), 60)
    store = ShadowRuntimeDayStore(runtime_root, signing_key=_RUNTIME_KEY.encode())
    for signal_date in dates:
        _day(store, signal_date)
    calendar = _calendar(calendar_root, dates)

    first = build_shadow_evidence(
        _MODEL_ID,
        runtime_root=runtime_root,
        evidence_root=evidence_root,
        calendar=calendar,
        calendar_root=calendar_root,
    )
    second = build_shadow_evidence(
        _MODEL_ID,
        runtime_root=runtime_root,
        evidence_root=evidence_root,
        calendar=calendar,
        calendar_root=calendar_root,
    )

    assert second == first
    assert write_shadow_evidence(second, evidence_root) == write_shadow_evidence(
        first, evidence_root
    )
