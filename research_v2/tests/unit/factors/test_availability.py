from __future__ import annotations

from datetime import date

import pandas as pd

from research_v2.factors.availability import (
    assign_fundamental_available_date,
    expand_fundamental_events,
    materialize_technical_availability,
)
from research_v2.pit.calendar import TradingCalendar


def _calendar() -> TradingCalendar:
    return TradingCalendar.from_dates(
        [
            date(2025, 3, 28),
            date(2025, 3, 31),
            date(2025, 4, 1),
            date(2025, 9, 26),
            date(2025, 9, 29),
        ]
    )


def test_technical_factor_is_available_only_on_its_observation_date() -> None:
    observations = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "observation_date": [date(2025, 3, 28)],
            "raw_value": [1.0],
        }
    )

    result = materialize_technical_availability(observations)

    assert result.loc[0, "available_date"] == date(2025, 3, 28)
    assert result.loc[0, "signal_date"] == date(2025, 3, 28)
    assert result.loc[0, "stale_days"] == 0


def test_fundamental_factor_becomes_available_on_next_trading_day_and_expires() -> None:
    calendar = _calendar()
    assert assign_fundamental_available_date(date(2025, 3, 28), calendar) == date(2025, 3, 31)
    events = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "report_period": [date(2024, 12, 31)],
            "ann_date": [date(2025, 3, 28)],
            "raw_value": [2.0],
            "source_revision_id": ["first"],
        }
    )
    universe = pd.DataFrame(
        {
            "signal_date": [
                date(2025, 3, 28),
                date(2025, 3, 31),
                date(2025, 9, 26),
                date(2025, 9, 29),
            ],
            "ts_code": ["000001.SZ"] * 4,
        }
    )

    result = expand_fundamental_events(events, universe, calendar, max_staleness_calendar_days=180)

    assert result["signal_date"].tolist() == [date(2025, 3, 31), date(2025, 9, 26)]
    assert result["stale_days"].tolist() == [0, 179]


def test_fundamental_event_prefers_latest_report_period_before_revision() -> None:
    calendar = _calendar()
    events = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "report_period": [date(2024, 9, 30), date(2024, 12, 31)],
            "ann_date": [date(2025, 3, 28), date(2025, 3, 28)],
            "raw_value": [1.0, 2.0],
            "source_revision_id": ["z-earlier-period", "a-later-period"],
        }
    )
    universe = pd.DataFrame(
        {
            "signal_date": [date(2025, 3, 31)],
            "ts_code": ["000001.SZ"],
        }
    )

    result = expand_fundamental_events(events, universe, calendar)

    assert result.loc[0, "report_period"] == date(2024, 12, 31)
    assert result.loc[0, "raw_value"] == 2.0
    assert result.loc[0, "source_revision_id"] == "a-later-period"
