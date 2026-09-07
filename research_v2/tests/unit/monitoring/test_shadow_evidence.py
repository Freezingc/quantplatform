from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from research_v2.monitoring.shadow import (
    ShadowEvidence,
    ShadowEvidenceVerifier,
    ShadowQualificationStatus,
    ShadowRuntimeDay,
    ShadowRuntimeDayStore,
    ShadowRuntimeDayVerifier,
    ShadowTradingCalendar,
    ShadowTradingCalendarStore,
    ShadowTradingCalendarVerifier,
    _canonical_json,
    _shadow_calendar_signature,
    _shadow_evidence_signature,
    _shadow_runtime_day_signature,
    build_shadow_evidence,
    evaluate_shadow_qualification,
    write_shadow_evidence,
)
from research_v2.reporting.shadow import ShadowDailyReport, build_shadow_daily_report

_MODEL_ID = "sha256:" + "a" * 64
_RUNTIME_KEY = "trusted-runtime-key"
_EVIDENCE_KEY = "trusted-evidence-key"
_CALENDAR_KEY = "trusted-calendar-key"


def _content_hash_of(payload: object) -> str:
    """The content address every shadow document id is supposed to be."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


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
    model_id: str = _MODEL_ID,
    psi_error: bool = False,
    pipeline_status: str = "SUCCESS",
    prediction_published: bool = True,
) -> ShadowDailyReport:
    return build_shadow_daily_report(
        signal_date=signal_date,
        model_id=model_id,
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
    model_id: str = _MODEL_ID,
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
            model_id=model_id,
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


# --- what the other three signed documents bind ------------------------------
#
# The signature check on each of these is pinned: an untrusted writer is refused for the
# calendar and the runtime day, and removing any compare_digest kills a test. What was not
# pinned is what each signature BINDS -- the payload's own content hash, the id the document
# is stored under, the exact key set, the issued record, and the exclusivity and mode of the
# file the issuer writes. Each test below forges a document that is correctly signed by the
# trusted key and therefore passes every signature check, so only the clause under test can
# refuse it.


def _stored(root: Path, name: str) -> dict:
    return json.loads((root / f"{name}.json").read_text(encoding="utf-8"))


def _write_forgery(root: Path, document_id: str, document: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{document_id}.json").write_text(
        _canonical_json(document) + "\n", encoding="utf-8"
    )


def test_a_calendar_stored_under_a_foreign_id_or_with_extra_fields_is_refused(
    tmp_path: Path,
) -> None:
    """The calendar id must be its payload's hash, and the payload must hold nothing else.

    The calendar is what decides which days a shadow qualification is measured over, so a
    document that is correctly signed but stored under an id that is not its content hash
    -- or that carries an extra field nobody validated -- is the case the signature check
    cannot see.
    """
    root = tmp_path / "calendar"
    dates = _trading_dates(date(2026, 1, 2), 5)
    calendar = _calendar(root, dates)
    payload = _stored(root, calendar.calendar_id)["payload"]
    # Guard the guard: the genuine document really is content-addressed.
    assert calendar.calendar_id == _content_hash_of(payload)

    verifier = ShadowTradingCalendarVerifier.from_environment(root)
    bogus = "0" * 64
    _write_forgery(
        root,
        bogus,
        {
            "calendar_id": bogus,
            "payload": payload,
            "signature": _shadow_calendar_signature(_CALENDAR_KEY.encode(), bogus, payload),
        },
    )
    with pytest.raises(ValueError, match="signature verification failed"):
        verifier.verify(ShadowTradingCalendar(bogus))

    extra_root = tmp_path / "calendar-extra"
    extra_payload = {**payload, "surprise": 1}
    extra_id = _content_hash_of(extra_payload)
    _write_forgery(
        extra_root,
        extra_id,
        {
            "calendar_id": extra_id,
            "payload": extra_payload,
            "signature": _shadow_calendar_signature(
                _CALENDAR_KEY.encode(), extra_id, extra_payload
            ),
        },
    )
    with pytest.raises(ValueError, match="calendar is invalid"):
        ShadowTradingCalendarVerifier.from_environment(extra_root).verify(
            ShadowTradingCalendar(extra_id)
        )


def test_a_runtime_day_stored_under_a_foreign_id_is_refused(tmp_path: Path) -> None:
    """A runtime day's payload must hash to the id it is stored and presented under.

    Forged with the trusted runtime key, so the signature check passes and the payload is
    otherwise a genuine day; only the content-hash comparison can refuse it.
    """
    root = tmp_path / "runtime"
    store = ShadowRuntimeDayStore(root, signing_key=_RUNTIME_KEY.encode())
    day = _day(store, date(2026, 1, 2))
    payload = _stored(root, day.runtime_day_id)["payload"]
    assert day.runtime_day_id == _content_hash_of(payload)

    bogus = "0" * 64
    _write_forgery(
        root,
        bogus,
        {
            "runtime_day_id": bogus,
            "payload": payload,
            "signature": _shadow_runtime_day_signature(_RUNTIME_KEY.encode(), bogus, payload),
        },
    )

    with pytest.raises(ValueError, match="signature verification failed"):
        ShadowRuntimeDayVerifier.from_environment(root).verify(
            ShadowRuntimeDay(bogus, _MODEL_ID)
        )


def _issued_evidence(base: Path) -> tuple[ShadowEvidence, Path]:
    """Sixty clean shadow days, qualified and issued as signed readiness evidence."""
    runtime_root = base / "runtime"
    calendar_root = base / "calendar"
    evidence_root = base / "evidence"
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
    return evidence, evidence_root


def test_readiness_evidence_must_hash_to_its_id_and_match_its_issued_record(
    tmp_path: Path,
) -> None:
    """Two refusals on the document the pointer switch actually consumes.

    Tampering with the issued record after the fact is caught by comparing the presented
    bytes against it; presenting a payload re-signed under an id that is not its content
    hash is caught before the signature is even considered. Both were unpinned, and this is
    the evidence that authorises a shadow pointer switch.
    """
    genuine, root = _issued_evidence(tmp_path / "genuine")
    verifier = ShadowEvidenceVerifier.from_environment(root)
    assert verifier.verify(genuine) is None  # control: the genuine article verifies

    record = root / ".issued" / f"{genuine.evidence_id}.json"
    record.write_text(genuine.to_json() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its issued record"):
        verifier.verify(genuine)

    source, forged_root = _issued_evidence(tmp_path / "forged")
    bogus = "0" * 64
    assert bogus != source.evidence_id
    forged = ShadowEvidence(
        bogus,
        source.payload,
        _shadow_evidence_signature(_EVIDENCE_KEY.encode(), bogus, source.payload),
    )
    (forged_root / ".issued").mkdir(parents=True, exist_ok=True)
    (forged_root / ".issued" / f"{bogus}.json").write_text(
        forged.to_json(), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="does not match evidence_id"):
        ShadowEvidenceVerifier.from_environment(forged_root).verify(forged)


def test_issued_shadow_documents_are_exclusive_and_owner_only(tmp_path: Path) -> None:
    """Issuance is write-once at 0o600, and both are invisible to every workflow test.

    Re-issuing the same calendar produces the same id and therefore the same path; without
    O_EXCL that silently overwrites a document other approvals may already cite, and no
    behaviour in the workflow changes either way, which is why only an explicit assertion
    holds it.
    """
    root = tmp_path / "calendar"
    dates = _trading_dates(date(2026, 1, 2), 5)
    calendar = _calendar(root, dates)
    path = root / f"{calendar.calendar_id}.json"

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError, match="immutable"):
        _calendar(root, dates)


def test_a_runtime_day_presented_under_another_model_is_refused(tmp_path: Path) -> None:
    """The record's model must equal the model the day is presented for.

    ``records_for_model`` filters by model before verifying, so this clause never fires on
    that path -- which is exactly why it needs its own test: the verifier is public, and
    without the clause a caller who presents another model's day id under their own model
    gets that day's facts back as though they were theirs.
    """
    root = tmp_path / "runtime"
    store = ShadowRuntimeDayStore(root, signing_key=_RUNTIME_KEY.encode())
    day = _day(store, date(2026, 1, 2))
    verifier = ShadowRuntimeDayVerifier.from_environment(root)
    assert verifier.verify(day).model_id == _MODEL_ID  # control

    with pytest.raises(ValueError, match="does not match its immutable record"):
        verifier.verify(ShadowRuntimeDay(day.runtime_day_id, "sha256:" + "b" * 64))


def test_qualification_ignores_other_models_days_in_a_shared_runtime_directory(
    tmp_path: Path,
) -> None:
    """``records_for_model`` skips other models' records instead of choking on them.

    Measured, and the difference is not silent: with the filter removed the verifier is
    handed another model's record and refuses it, so a runtime directory holding two
    models' days -- which is what the store's flat layout produces -- would fail
    qualification outright instead of counting only the days belonging to the model being
    qualified. This is also the clause that made chain 11's SH-A redundant, so pinning it
    is what keeps that classification true.
    """
    runtime_root = tmp_path / "runtime"
    calendar_root = tmp_path / "calendar"
    store = ShadowRuntimeDayStore(runtime_root, signing_key=_RUNTIME_KEY.encode())
    dates = _trading_dates(date(2026, 1, 2), 60)
    calendar = _calendar(calendar_root, dates)
    other = "sha256:" + "b" * 64
    for signal_date in dates:
        _day(store, signal_date)
        _day(store, signal_date, model_id=other)

    qualification = evaluate_shadow_qualification(
        _MODEL_ID,
        runtime_root=runtime_root,
        calendar=calendar,
        calendar_root=calendar_root,
    )

    assert qualification.valid_day_count == 60
    assert qualification.status is ShadowQualificationStatus.READY_FOR_REVIEW


def test_readiness_evidence_with_a_forged_signature_is_refused(tmp_path: Path) -> None:
    """The HMAC is checked even when the issued record has been rewritten to match it.

    Forging the signature alone would be refused by the record comparison, so a test that
    stopped there would pin the wrong clause. Rewriting the record to match the forgery
    leaves the HMAC as the only thing standing between a made-up signature and accepted
    readiness evidence -- which is the entire reason it is an HMAC rather than a hash.
    """
    genuine, root = _issued_evidence(tmp_path / "evidence")
    verifier = ShadowEvidenceVerifier.from_environment(root)
    assert verifier.verify(genuine) is None  # control

    forged = replace(genuine, signature="0" * 64)
    (root / ".issued" / f"{genuine.evidence_id}.json").write_text(
        forged.to_json(), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="signature verification failed"):
        verifier.verify(forged)


# --- the evidence payload's rulebook -----------------------------------------
#
# Fifteen of seventeen mutations of the issuance path survived. What was pinned is the
# status check in issue() and the content-addressed evidence id; what was not is the whole
# rulebook that ties the published numbers to each other -- the sixty-day floor, the
# observed-day and success-rate domains, and six arithmetic cross-checks binding the date
# lists and the three id lists to the day count. Those cross-checks are the only thing that
# makes the published evidence self-consistent rather than merely signed.


def test_the_evidence_payload_rulebook_refuses_each_inconsistency(tmp_path: Path) -> None:
    """One case per clause of _validate_shadow_evidence_payload, each refused by name.

    Every payload below is a real issued payload made inconsistent in exactly one way, so
    each case isolates one clause -- which takes more care than it looks, because the
    cross-checks interlock: violating the observed-day sum also violates the rate quotient
    unless the rate is corrected too, and putting a valid date into the invalid list also
    breaks the observed-day count unless that is corrected as well. Three of these cases
    were rebuilt for exactly that reason after a first version tripped an earlier clause and
    so proved nothing about the clause it named. ``ShadowEvidence.verify`` validates the
    payload before it compares the id, which is what makes the reason survive instead of
    collapsing into "the hash does not match".
    """
    genuine, _root = _issued_evidence(tmp_path / "rulebook")
    base = dict(genuine.payload)
    count = int(base["valid_day_count"])
    # Guard the guard: the genuine payload passes its own rulebook.
    assert ShadowEvidence(genuine.evidence_id, base, genuine.signature).verify() is None

    fifty_nine = {
        **base,
        "valid_day_count": 59,
        "valid_signal_dates": list(base["valid_signal_dates"])[:59],
        "valid_report_ids": list(base["valid_report_ids"])[:59],
        "valid_runtime_day_ids": list(base["valid_runtime_day_ids"])[:59],
        "observed_day_count": 59,
        "successful_day_rate": 1.0,
    }
    cases = (
        ("extra field", {**base, "surprise": 1}, "payload is invalid"),
        ("fifty-nine consistent days", fifty_nine, "valid_day_count is invalid"),
        ("observed below valid", {**base, "observed_day_count": count - 1}, "observed-day metrics"),
        ("rate out of domain", {**base, "successful_day_rate": 1.5}, "observed-day metrics"),
        (
            "valid dates short",
            {**base, "valid_signal_dates": list(base["valid_signal_dates"])[:-1]},
            "dates or report IDs",
        ),
        # Observed disagrees with valid plus invalid, while the rate still equals its own
        # quotient, so only the sum check can refuse it.
        (
            "observed is not the sum",
            {
                **base,
                "observed_day_count": count + 5,
                "successful_day_rate": count / (count + 5),
            },
            "dates or report IDs",
        ),
        ("rate is not the quotient", {**base, "successful_day_rate": 0.5}, "dates or report IDs"),
        # A date in both lists, with every count and the rate kept consistent and a runtime
        # id that does NOT overlap, so only the date-overlap check can refuse it.
        (
            "a date both valid and invalid",
            {
                **base,
                "invalid_signal_dates": [base["valid_signal_dates"][0]],
                "invalid_runtime_day_ids": ["f" * 64],
                "observed_day_count": count + 1,
                "successful_day_rate": count / (count + 1),
            },
            "dates or report IDs",
        ),
        (
            "report ids short",
            {**base, "valid_report_ids": list(base["valid_report_ids"])[:-1]},
            "dates or report IDs",
        ),
        # Counts, dates and list lengths all consistent, and the invalid runtime id IS one of
        # the valid ones, so only the runtime-id overlap check can refuse it.
        (
            "runtime day ids overlap",
            {
                **base,
                "invalid_signal_dates": ["2026-03-28"],
                "invalid_runtime_day_ids": [base["valid_runtime_day_ids"][0]],
                "observed_day_count": count + 1,
                "successful_day_rate": count / (count + 1),
            },
            "dates or report IDs",
        ),
    )
    for name, payload, match in cases:
        try:
            ShadowEvidence("0" * 64, payload, genuine.signature).verify()
        except ValueError as error:
            assert match in str(error), f"{name}: refused for the wrong reason: {error}"
        else:
            pytest.fail(f"payload accepted: {name}")


def test_issued_evidence_reports_the_true_success_rate_not_a_perfect_one(
    tmp_path: Path,
) -> None:
    """A day that was observed but not valid must show up in the published rate.

    Sixty valid days plus one invalid day is the shape the qualification produces whenever
    any day fails the filter, and the payload carries both the counts and their quotient.
    Hardcoding the rate to 1.0 leaves every count correct and only the quotient wrong, so
    it is invisible to any test that reads the counts -- and the rate is what a reviewer
    reads first.
    """
    runtime_root = tmp_path / "runtime"
    calendar_root = tmp_path / "calendar"
    evidence_root = tmp_path / "evidence"
    store = ShadowRuntimeDayStore(runtime_root, signing_key=_RUNTIME_KEY.encode())
    dates = _trading_dates(date(2026, 1, 2), 60)
    calendar = _calendar(calendar_root, dates)
    for signal_date in dates:
        _day(store, signal_date)
    _day(store, date(2026, 3, 28))  # observed, but not a trading date: invalid

    evidence = build_shadow_evidence(
        _MODEL_ID,
        runtime_root=runtime_root,
        evidence_root=evidence_root,
        calendar=calendar,
        calendar_root=calendar_root,
    )

    assert evidence.payload["valid_day_count"] == 60
    assert evidence.payload["observed_day_count"] == 61
    assert evidence.payload["successful_day_rate"] == pytest.approx(60 / 61)
    assert evidence.payload["invalid_signal_dates"] == ["2026-03-28"]


def test_publishing_shadow_evidence_verifies_it_and_scopes_it_to_its_model(
    tmp_path: Path,
) -> None:
    """The publisher is a gate, not a write: it verifies first, and files by model.

    Two properties, both unpinned. Skipping the verification call publishes whatever it is
    handed, and the file it writes is the artifact downstream consumers read. And the
    destination is model-scoped by design -- one directory per model id -- so collapsing it
    to a single shared path would let a second model's evidence overwrite the first, which
    the existing assertion on the file NAME cannot see.
    """
    genuine, root = _issued_evidence(tmp_path / "publish")
    published = write_shadow_evidence(genuine, root)

    assert published.name == "shadow_evidence.json"
    assert published.parent.name == _MODEL_ID.removeprefix("sha256:")

    # Tamper with the id and leave the payload valid, so the refusal is the publisher's
    # verification step and not the payload rulebook exercised above.
    tampered = ShadowEvidence("1" * 64, genuine.payload, genuine.signature)
    other_root = tmp_path / "publish-tampered"
    with pytest.raises(ValueError, match="does not match evidence_id"):
        write_shadow_evidence(tampered, other_root)
    assert not (other_root / _MODEL_ID.removeprefix("sha256:")).exists()
