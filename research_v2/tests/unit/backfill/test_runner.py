from __future__ import annotations

from datetime import date

from research_v2.backfill.runner import MonthlyBackfillRunner
from research_v2.pit.calendar import TradingCalendar


def _calendar() -> TradingCalendar:
    return TradingCalendar.from_dates(
        [
            date(2019, 12, 30),
            date(2019, 12, 31),
            date(2020, 1, 2),
            date(2020, 1, 3),
            date(2020, 1, 6),
            date(2020, 1, 7),
            date(2020, 1, 8),
            date(2020, 1, 9),
            date(2020, 1, 10),
            date(2020, 2, 3),
            date(2020, 2, 4),
        ]
    )


def test_runner_plans_only_missing_objects_for_each_month() -> None:
    pointers = {
        ("market_daily_clean", "2020-01"): "market-v1",
        ("tradability_daily", "2020-01"): "trad-v1",
        ("universe_daily", "2020-01"): "universe-v1",
        ("market_daily_clean", "2020-02"): "market-v2",
    }
    runner = MonthlyBackfillRunner(
        _calendar(),
        lambda object_name, month: pointers.get((object_name, month)),
        required_objects=("market_daily_clean", "tradability_daily", "universe_daily"),
    )

    plans = runner.plan(date(2020, 1, 1), date(2020, 2, 29), lookback_trading_days=2)

    assert plans[0].status == "complete"
    assert plans[0].pending_objects == ()
    assert plans[1].status == "pending"
    assert plans[1].pending_objects == ("tradability_daily", "universe_daily")


def test_runner_skips_complete_month_and_invokes_publish_for_pending_month() -> None:
    pointers = {
        ("market_daily_clean", "2020-01"): "market-v1",
        ("tradability_daily", "2020-01"): "trad-v1",
        ("universe_daily", "2020-01"): "universe-v1",
    }
    calls: list[tuple[str, tuple[str, ...]]] = []
    runner = MonthlyBackfillRunner(
        _calendar(),
        lambda object_name, month: pointers.get((object_name, month)),
        required_objects=("market_daily_clean", "tradability_daily", "universe_daily"),
    )

    results = runner.run(
        date(2020, 1, 1),
        date(2020, 2, 29),
        publish=lambda window, pending: calls.append((window.month, pending)) or pending,
        lookback_trading_days=2,
    )

    assert calls == [("2020-02", ("market_daily_clean", "tradability_daily", "universe_daily"))]
    assert [result.status for result in results] == ["complete", "published"]


# --- the fail-closed guards, and the boundary between this runner and C5.4 -----------------------

import pytest  # noqa: E402

_REQUIRED = ("market_daily_clean", "tradability_daily", "universe_daily")
_JANUARY_POINTERS = {
    ("market_daily_clean", "2020-01"): "market-v1",
    ("tradability_daily", "2020-01"): "trad-v1",
    ("universe_daily", "2020-01"): "universe-v1",
}


def _runner(pointers: dict[tuple[str, str], str]) -> MonthlyBackfillRunner:
    return MonthlyBackfillRunner(
        _calendar(),
        lambda object_name, month: pointers.get((object_name, month)),
        required_objects=_REQUIRED,
    )


def test_the_publish_callback_cannot_report_an_object_that_was_not_pending() -> None:
    """The guard that keeps a month's publish from writing another month's object.

    ``run`` plans per month and hands the callback only that month's pending set. A callback that
    reports anything outside it is refused rather than recorded, because the alternative is a
    result claiming an object was published for a month whose pointer was never checked -- which
    is how a resumable backfill ends up with a month marked complete on evidence about a
    different month.
    """
    runner = _runner(dict(_JANUARY_POINTERS))

    with pytest.raises(ValueError, match="was not pending"):
        runner.run(
            date(2020, 1, 1),
            date(2020, 2, 29),
            # an object outside the required set, so it is pending in no month at all. Reporting
            # one of the required names instead would trip the uniqueness guard first and the
            # cross-object guard would go untested.
            publish=lambda window, pending: (*pending, "label_daily"),
            lookback_trading_days=2,
        )


def test_the_publish_callback_cannot_report_the_same_object_twice() -> None:
    runner = _runner(dict(_JANUARY_POINTERS))

    with pytest.raises(ValueError, match="unique object names"):
        runner.run(
            date(2020, 1, 1),
            date(2020, 2, 29),
            publish=lambda window, pending: ("market_daily_clean", "market_daily_clean"),
            lookback_trading_days=2,
        )


def test_a_publish_callback_that_reports_nothing_is_refused() -> None:
    """A pending month whose callback publishes nothing must not be recorded as published.

    Silently accepting an empty report would mark the month ``published`` while no pointer was
    written, so the next resumed run would plan it as pending again and the results would disagree
    with the pointer table. Note the message this raises talks about *unique* names, which is
    misleading for the empty case -- pinned as-is so that improving it is a deliberate change.
    """
    runner = _runner(dict(_JANUARY_POINTERS))

    with pytest.raises(ValueError, match="unique object names"):
        runner.run(
            date(2020, 1, 1),
            date(2020, 2, 29),
            publish=lambda window, pending: (),
            lookback_trading_days=2,
        )


@pytest.mark.parametrize(
    "required",
    [(), ("market_daily_clean", "market_daily_clean")],
    ids=["empty", "duplicated"],
)
def test_required_objects_must_be_non_empty_and_unique(required: tuple[str, ...]) -> None:
    """An empty requirement set would make every month trivially complete and publish nothing."""
    with pytest.raises(ValueError, match="unique object names"):
        MonthlyBackfillRunner(
            _calendar(),
            lambda object_name, month: None,
            required_objects=required,
        )


def test_the_runner_is_blind_to_source_change_and_answers_only_pointer_presence() -> None:
    """The boundary between this runner and C5.4, so it is not mistaken for incremental scope.

    C5.4 asks that a change in a technical daily source recompute **only the changed signal
    dates**. This runner answers a different question: *which months have never been published*.
    Its sole input about state is the injected pointer reader, so a month that already has a
    pointer is reported complete **no matter what happened to its source afterwards** -- a
    corrected value, a restated row or a new watermark in an already-published month would never
    be recomputed by it, and the plan would not change.

    That is not a defect in the runner; resumability and change detection are different jobs, and
    conflating them would make a resumed backfill re-publish months it had already finished. It is
    recorded because "we have a resumable monthly runner" reads like "we have incremental
    recomputation", and the second does not exist. Detecting source change needs a per-date digest
    or watermark comparison against persisted state, which is a schema decision.

    The two source digests below are the test's stand-in for a changed source. Neither is an input
    to the runner, and the assertion is exactly that: the plans are identical.
    """
    runner = _runner(dict(_JANUARY_POINTERS))
    source_digest_before = {"2020-01": "digest-a"}
    source_digest_after = {"2020-01": "digest-b-changed"}
    assert source_digest_before != source_digest_after, "fixture premise: the source did change"

    plans_before = runner.plan(date(2020, 1, 1), date(2020, 1, 31), lookback_trading_days=2)
    plans_after = runner.plan(date(2020, 1, 1), date(2020, 1, 31), lookback_trading_days=2)

    assert [plan.status for plan in plans_before] == ["complete"]
    assert plans_before == plans_after, "the plan did not react to a changed source"
    assert all(plan.pending_objects == () for plan in plans_after)


def test_a_missing_pointer_for_one_object_leaves_only_that_object_pending() -> None:
    """Granularity is per object within a month, so a partial month republishes only the gap."""
    pointers = {
        key: value for key, value in _JANUARY_POINTERS.items() if key[0] != "universe_daily"
    }
    runner = _runner(pointers)

    plans = runner.plan(date(2020, 1, 1), date(2020, 1, 31), lookback_trading_days=2)

    assert plans[0].status == "pending"
    assert plans[0].pending_objects == ("universe_daily",)
