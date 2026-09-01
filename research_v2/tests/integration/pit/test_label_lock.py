"""Locked-year label materialization integration evidence (B7.5)."""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import pytest

from research_v2.labels.forward_return import LockedLabelError, build_forward_open_labels
from research_v2.pit.calendar import TradingCalendar

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("QRV2_INTEGRATION") != "1",
        reason="set QRV2_INTEGRATION=1 to run PIT integration tests",
    ),
]

# Calendar spanning the 2025/2026 boundary.
_CALENDAR = TradingCalendar.from_dates(
    [
        date(2025, 12, 22),
        date(2025, 12, 23),
        date(2025, 12, 24),
        date(2025, 12, 25),
        date(2025, 12, 26),
        date(2025, 12, 29),
        date(2025, 12, 30),
        date(2025, 12, 31),
        date(2026, 1, 2),
        date(2026, 1, 5),
        date(2026, 1, 6),
    ]
)
_MARKET = pd.DataFrame(
    [
        {"trade_date": date(2025, 12, 26), "ts_code": "X01", "adjusted_open": 20.0},
        {"trade_date": date(2025, 12, 29), "ts_code": "X01", "adjusted_open": 20.5},
        {"trade_date": date(2026, 1, 5), "ts_code": "X01", "adjusted_open": 21.0},
    ]
)


def _members(signal_dates) -> pd.DataFrame:
    return pd.DataFrame(
        [{"signal_date": day, "ts_code": "X01"} for day in signal_dates]
    )


def test_ordinary_identity_rejects_2026_labels_before_computation() -> None:
    members = _members([date(2025, 12, 25), date(2026, 1, 5)])

    with pytest.raises(LockedLabelError, match="locked year"):
        build_forward_open_labels(members, _MARKET, _CALENDAR)


def test_ordinary_identity_rejects_a_pure_2026_frame() -> None:
    members = _members([date(2026, 1, 5)])

    with pytest.raises(LockedLabelError, match="locked year"):
        build_forward_open_labels(members, _MARKET, _CALENDAR)


def test_2025_signal_with_2026_exit_remains_historical_and_records_cross_year() -> None:
    members = _members([date(2025, 12, 25)])

    labels = build_forward_open_labels(members, _MARKET, _CALENDAR)

    row = labels.iloc[0]
    assert row["entry_date"] == date(2025, 12, 26)
    assert row["exit_date"] == date(2026, 1, 5)
    assert bool(row["label_valid"]) is True
    assert row["raw_return_5d"] == pytest.approx(0.05, abs=1e-12)
    # Maturity is expressed by the exit date itself falling in the locked year.
    assert row["label_available_date"] == date(2026, 1, 5)
    assert row["exit_date"].year != row["signal_date"].year


def test_2025_signal_waits_for_unmatured_exit_price() -> None:
    # Signal 2025-12-26 exits at 2026-01-06; that exit price is not observed yet,
    # so the label stays invalid until maturity instead of leaking an estimate.
    members = _members([date(2025, 12, 26)])

    labels = build_forward_open_labels(members, _MARKET, _CALENDAR)

    row = labels.iloc[0]
    assert row["exit_date"] == date(2026, 1, 6)
    assert bool(row["label_valid"]) is False
    assert row["invalid_reason"] == "MISSING_EXIT_PRICE"
    assert bool(row["exit_price_observed"]) is False
