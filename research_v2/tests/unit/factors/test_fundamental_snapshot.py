from __future__ import annotations

from datetime import date

import pandas as pd

from research_v2.factors.fundamental_snapshot import build_fundamental_snapshot
from research_v2.pit.calendar import TradingCalendar


def test_snapshot_materializes_metrics_independently_with_pit_lineage() -> None:
    events = pd.DataFrame(
        {
            "metric_id": ["roa", "roa", "roe"],
            "ts_code": ["000001.SZ"] * 3,
            "report_period": [date(2024, 9, 30), date(2024, 12, 31), date(2024, 12, 31)],
            "ann_date": [date(2025, 3, 28)] * 3,
            "source_revision_id": ["z-old", "a-new", "first"],
            "raw_value": [1.0, 2.0, 3.0],
        }
    )
    universe = pd.DataFrame(
        {
            "signal_date": [date(2025, 3, 31)],
            "ts_code": ["000001.SZ"],
        }
    )
    calendar = TradingCalendar.from_dates([date(2025, 3, 28), date(2025, 3, 31)])

    result = build_fundamental_snapshot(events, universe, calendar)

    assert result[["metric_id", "value"]].to_dict("records") == [
        {"metric_id": "roa", "value": 2.0},
        {"metric_id": "roe", "value": 3.0},
    ]
    assert result["available_date"].tolist() == [date(2025, 3, 31)] * 2
    assert result["report_period"].tolist() == [date(2024, 12, 31)] * 2


def test_snapshot_rejects_duplicate_metric_stock_day_keys() -> None:
    events = pd.DataFrame(
        {
            "metric_id": ["roa"],
            "ts_code": ["000001.SZ"],
            "report_period": [date(2024, 12, 31)],
            "ann_date": [date(2025, 3, 28)],
            "source_revision_id": ["first"],
            "raw_value": [2.0],
        }
    )
    universe = pd.DataFrame(
        {
            "signal_date": [date(2025, 3, 31), date(2025, 3, 31)],
            "ts_code": ["000001.SZ", "000001.SZ"],
        }
    )
    calendar = TradingCalendar.from_dates([date(2025, 3, 28), date(2025, 3, 31)])

    try:
        build_fundamental_snapshot(events, universe, calendar)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate universe keys must be rejected")


def test_snapshot_handles_nontrading_announcements_with_event_lineage() -> None:
    events = pd.DataFrame(
        {
            "metric_id": ["roa"],
            "ts_code": ["000001.SZ"],
            "report_period": [date(2024, 12, 31)],
            "ann_date": [date(2025, 3, 29)],
            "source_revision_id": ["first"],
            "raw_value": [2.0],
        }
    )
    universe = pd.DataFrame(
        {
            "signal_date": [date(2025, 3, 31), date(2025, 4, 1)],
            "ts_code": ["000001.SZ", "000001.SZ"],
        }
    )
    calendar = TradingCalendar.from_dates(
        [date(2025, 3, 28), date(2025, 3, 31), date(2025, 4, 1)]
    )

    result = build_fundamental_snapshot(events, universe, calendar)

    assert result["available_date"].tolist() == [date(2025, 3, 31)] * 2
    assert result["stale_days"].tolist() == [0, 1]


def test_snapshot_is_order_invariant_when_announcement_dates_collapse() -> None:
    """Two announcements rolling to one available date must not be settled by row order.

    ``available_date`` is the first trading day strictly after ``ann_date``, so a Friday and the
    weekend behind it all map to the same Monday. Two events for one metric, security, report
    period and revision id announced on 2025-08-29 and 2025-08-30 therefore tie on
    ``available_date``, ``report_period`` and ``source_revision_id``, and before ``ann_date``
    became the fourth sort key the stable sort plus ``keep="last"`` resolved that tie by input
    row order -- publishing a different ``ann_date`` lineage column depending on how the extract
    arrived, with the value itself never in dispute.

    This is the same shape hypothesis found in
    ``availability.expand_fundamental_events`` on 2026-09-06, in the function whose docstring
    calls itself that one's vectorized per-security equivalent. Both are fixed the same way so
    the equivalence survives permutation.

    A census of the real sources found this shape does not occur today -- across 1,377,959
    (ts_code, end_date, update_flag) groups in ``financial_metrics_table``, ``balance_sheet`` and
    ``company_income_table``, 505 hold more than one ``ann_date`` and none of the 505 collapse to
    a single available date -- so the published ``fundamental_pit_snapshot`` rows are unaffected
    and this fix is a no-op on current data. It is pinned because a vendor restatement or a new
    metric table would introduce the shape silently, and because an equivalence claimed in a
    docstring should be one a test can fail.
    """
    events = pd.DataFrame(
        {
            "metric_id": ["roa", "roa"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "report_period": [date(2024, 12, 31), date(2024, 12, 31)],
            "ann_date": [date(2025, 8, 29), date(2025, 8, 30)],
            "source_revision_id": ["r1", "r1"],
            "raw_value": [2.0, 2.0],
        }
    )
    universe = pd.DataFrame(
        {"signal_date": [date(2025, 9, 1)], "ts_code": ["000001.SZ"]}
    )
    calendar = TradingCalendar.from_dates([date(2025, 9, 1), date(2025, 9, 2)])

    forward = build_fundamental_snapshot(events, universe, calendar)
    backward = build_fundamental_snapshot(
        events.iloc[::-1].reset_index(drop=True), universe, calendar
    )

    pd.testing.assert_frame_equal(forward, backward)
    assert len(forward) == 1
    assert forward["available_date"].tolist() == [date(2025, 9, 1)]
    assert forward["stale_days"].tolist() == [0]
    # The fourth key is ascending with keep="last", so the later announcement wins the lineage
    # column -- the same outcome expand_fundamental_events reaches descending and taking first.
    assert forward["ann_date"].tolist() == [date(2025, 8, 30)]
    assert forward["value"].tolist() == [2.0]


def test_snapshot_uses_later_effective_event_after_earlier_report_expires() -> None:
    events = pd.DataFrame(
        {
            "metric_id": ["roa", "roa"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "report_period": [date(2024, 9, 30), date(2024, 12, 31)],
            "ann_date": [date(2024, 9, 27), date(2025, 3, 28)],
            "source_revision_id": ["old", "new"],
            "raw_value": [1.0, 2.0],
        }
    )
    universe = pd.DataFrame(
        {
            "signal_date": [date(2025, 4, 1)],
            "ts_code": ["000001.SZ"],
        }
    )
    calendar = TradingCalendar.from_dates(
        [
            date(2024, 9, 27),
            date(2024, 9, 30),
            date(2025, 3, 28),
            date(2025, 3, 31),
            date(2025, 4, 1),
        ]
    )

    result = build_fundamental_snapshot(events, universe, calendar, max_staleness_calendar_days=180)

    assert result[["signal_date", "value"]].to_dict("records") == [
        {"signal_date": date(2025, 4, 1), "value": 2.0}
    ]
