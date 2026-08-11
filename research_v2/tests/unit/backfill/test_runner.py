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
