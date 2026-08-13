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
