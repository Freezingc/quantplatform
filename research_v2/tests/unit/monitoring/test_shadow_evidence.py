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


def _report(
    signal_date: date,
    *,
    psi_error: bool = False,
    pipeline_status: str = "SUCCESS",
    prediction_published: bool = True,
) -> ShadowDailyReport:
    return build_shadow_daily_report(
        signal_date=signal_date,
        model_id=_MODEL_ID,
        model_version=1,
        pipeline_status=pipeline_status,
        prediction_published=prediction_published,
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
    pipeline_status: str = "SUCCESS",
    prediction_published: bool = True,
) -> ShadowRuntimeDay:
    return store.issue(
        _report(
            signal_date,
            psi_error=psi_error,
            pipeline_status=pipeline_status,
            prediction_published=prediction_published,
        ),
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


# --- which days count, and what "consecutive" means --------------------------
#
# _is_valid_shadow_day is a six-condition filter and evaluate_shadow_qualification
# turns its output into the status that the pointer switch consumes. Eight of sixteen
# mutations of these two functions survived, and four of the eight are conditions nothing
# upstream enforces: ShadowRuntimeDayStore.issue verifies the report's own signature and
# takes model_status and the pointer version straight from the caller, so a day that was
# not a shadow day, or that failed, or that published nothing, is issuable and only this
# filter excludes it.


def test_a_day_counts_only_if_it_was_a_real_shadow_day_for_this_model(tmp_path: Path) -> None:
    """The filter's conditions that nothing upstream enforces, each broken on one of sixty days.

    Each case must cost exactly one valid day, which drops the model below the sixty-day
    floor and out of READY_FOR_REVIEW, and must name that day as invalid.

    Three of the six conditions are redundant and are recorded as redundant rather than
    pinned. ``records_for_model`` skips any record whose payload names a different model
    and ``ShadowRuntimeDayVerifier.verify`` re-checks the same thing, so the filter's model
    match is a third copy. ``_shadow_runtime_day_from_payload`` calls ``report.verify()``
    and refuses a malformed record before it can reach the filter, so the filter's own
    try/except cannot take its except branch at all. And ``build_shadow_daily_report``
    enforces ``prediction_published is (pipeline_status == "SUCCESS")`` on every legal
    report, which makes the filter's pipeline-status condition and its
    prediction-published condition the SAME predicate written twice: deleting either one
    alone changes nothing, and only deleting both lets a failed day count -- which is what
    the third case below is for.
    """
    cases = (
        ("not-in-shadow", {"model_status": "PRODUCTION"}),
        ("pointer-version-drifted", {"shadow_pointer_model_version": 2}),
        # FAILED forces prediction_published=False, so this one case is the only legal way
        # to present a day that did not succeed, and it catches the pair of clauses.
        (
            "pipeline-failed",
            {"pipeline_status": "FAILED", "prediction_published": False},
        ),
    )
    for name, overrides in cases:
        root = tmp_path / name
        calendar_root = root / "calendar"
        runtime_root = root / "runtime"
        store = ShadowRuntimeDayStore(runtime_root, signing_key=_RUNTIME_KEY.encode())
        dates = _trading_dates(date(2026, 1, 2), 60)
        calendar = _calendar(calendar_root, dates)
        for signal_date in dates[:-1]:
            _day(store, signal_date)
        _day(store, dates[-1], **overrides)  # type: ignore[arg-type]

        qualification = evaluate_shadow_qualification(
            _MODEL_ID,
            runtime_root=runtime_root,
            calendar=calendar,
            calendar_root=calendar_root,
        )

        assert qualification.valid_day_count == 59, name
        assert qualification.status is ShadowQualificationStatus.INCOMPLETE, name
        assert qualification.invalid_signal_dates == (dates[-1],), name


def test_the_psi_run_is_consecutive_and_not_a_total(tmp_path: Path) -> None:
    """Three PSI errors spread across the calendar are not three in a row.

    Measured: with the run's reset removed, ``max_consecutive_psi_errors`` becomes a count
    of every PSI-error day, so three scattered errors -- each isolated, each recovered from
    by the next trading day -- read as a streak and push a fully qualified sixty-day model
    into REVIEW_REQUIRED. Every existing streak test uses consecutive days, on which the
    two readings agree, which is why this survived.
    """
    root = tmp_path / "runtime"
    calendar_root = tmp_path / "calendar"
    store = ShadowRuntimeDayStore(root, signing_key=_RUNTIME_KEY.encode())
    dates = _trading_dates(date(2026, 1, 2), 60)
    calendar = _calendar(calendar_root, dates)
    scattered = {dates[10], dates[30], dates[50]}
    for signal_date in dates:
        _day(store, signal_date, psi_error=signal_date in scattered)

    qualification = evaluate_shadow_qualification(
        _MODEL_ID, runtime_root=root, calendar=calendar, calendar_root=calendar_root
    )

    assert qualification.max_consecutive_psi_errors == 1
    assert qualification.valid_day_count == 60
    assert qualification.status is ShadowQualificationStatus.READY_FOR_REVIEW
