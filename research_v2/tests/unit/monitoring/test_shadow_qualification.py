from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from research_v2.monitoring.shadow import (
    ShadowQualificationStatus,
    ShadowRuntimeDay,
    ShadowRuntimeDayStore,
    ShadowTradingCalendar,
    ShadowTradingCalendarStore,
    evaluate_shadow_qualification,
)
from research_v2.reporting.shadow import build_shadow_daily_report

_MODEL_ID = "sha256:" + "a" * 64
_RUNTIME_KEY = "trusted-runtime-key"
_CALENDAR_KEY = "trusted-calendar-key"


@pytest.fixture(autouse=True)
def _trusted_runtime_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QRV2_SHADOW_RUNTIME_HMAC_KEY", _RUNTIME_KEY)
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


def _day(
    store: ShadowRuntimeDayStore, signal_date: date, *, psi_error: bool = False
) -> ShadowRuntimeDay:
    report = build_shadow_daily_report(
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
        score_health={
            "finite_coverage": 0.99,
            "unique_score_count": 800,
            "score_std": 0.15,
        },
        seed_agreement={"minimum_spearman": 0.42},
        drift_alert_keys=(),
        event_ids=("e" * 64,),
        psi_error_features=("quality",) if psi_error else (),
    )
    return store.issue(
        report,
        model_status="SHADOW",
        shadow_pointer_model_id=_MODEL_ID,
        shadow_pointer_model_version=1,
        has_blocker=False,
    )


def test_shadow_qualification_requires_sixty_distinct_healthy_trading_days(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime"
    calendar_root = tmp_path / "calendar"
    store = ShadowRuntimeDayStore(root, signing_key=_RUNTIME_KEY.encode())
    dates = _trading_dates(date(2025, 1, 2), 60)
    calendar = _calendar(calendar_root, dates)
    for signal_date in dates:
        _day(store, signal_date)
    ready = evaluate_shadow_qualification(
        _MODEL_ID, runtime_root=root, calendar=calendar, calendar_root=calendar_root
    )
    assert ready.valid_day_count == 60
    assert ready.status is ShadowQualificationStatus.READY_FOR_REVIEW
    incomplete_root = tmp_path / "incomplete-runtime"
    incomplete_store = ShadowRuntimeDayStore(incomplete_root, signing_key=_RUNTIME_KEY.encode())
    tuple(_day(incomplete_store, signal_date) for signal_date in dates[:-1])
    incomplete = evaluate_shadow_qualification(
        _MODEL_ID,
        runtime_root=incomplete_root,
        calendar=calendar,
        calendar_root=calendar_root,
    )
    assert incomplete.status is ShadowQualificationStatus.INCOMPLETE


def test_three_consecutive_psi_errors_require_review_even_at_sixty_days(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime"
    calendar_root = tmp_path / "calendar"
    store = ShadowRuntimeDayStore(root, signing_key=_RUNTIME_KEY.encode())
    for index, signal_date in enumerate(_trading_dates(date(2025, 1, 2), 60)):
        _day(store, signal_date, psi_error=index in {10, 11, 12})

    assert (
        evaluate_shadow_qualification(
            _MODEL_ID,
            runtime_root=root,
            calendar=_calendar(calendar_root, _trading_dates(date(2025, 1, 2), 60)),
            calendar_root=calendar_root,
        ).status
        is ShadowQualificationStatus.REVIEW_REQUIRED
    )
