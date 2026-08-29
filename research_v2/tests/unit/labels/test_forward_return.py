from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import research_v2.labels.forward_return as forward_return
from research_v2.labels.forward_return import (
    LockedLabelError,
    build_forward_open_labels,
)
from research_v2.pit.calendar import TradingCalendar


@pytest.fixture(autouse=True)
def _locked_access_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QRV2_LOCKED_TEST_ACCESS_HMAC_KEY", "label-test-access-key")


def test_primary_label_is_t_plus_1_to_t_plus_6_adjusted_open_return() -> None:
    calendar = TradingCalendar.from_dates(
        [
            date(2025, 1, 3),
            date(2025, 1, 6),
            date(2025, 1, 7),
            date(2025, 1, 8),
            date(2025, 1, 9),
            date(2025, 1, 10),
            date(2025, 1, 13),
        ]
    )
    members = pd.DataFrame({"signal_date": [date(2025, 1, 3)], "ts_code": ["000001.SZ"]})
    market = pd.DataFrame(
        {
            "trade_date": [date(2025, 1, 6), date(2025, 1, 13)],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "adjusted_open": [10.0, 11.0],
        }
    )

    labels = build_forward_open_labels(members, market, calendar)

    assert labels.loc[0, "entry_date"] == date(2025, 1, 6)
    assert labels.loc[0, "exit_date"] == date(2025, 1, 13)
    assert labels.loc[0, "raw_return_5d"] == pytest.approx(0.1)
    assert labels.loc[0, "label_valid"]
    assert labels.loc[0, "label_available_date"] == date(2025, 1, 13)
    assert labels.loc[0, "entry_realized_tradable"]
    assert labels.loc[0, "exit_price_observed"]


def test_missing_entry_or_exit_price_is_an_invalid_label() -> None:
    calendar = TradingCalendar.from_dates(
        [
            date(2025, 1, 3),
            date(2025, 1, 6),
            date(2025, 1, 7),
            date(2025, 1, 8),
            date(2025, 1, 9),
            date(2025, 1, 10),
            date(2025, 1, 13),
        ]
    )
    members = pd.DataFrame({"signal_date": [date(2025, 1, 3)], "ts_code": ["000001.SZ"]})
    market = pd.DataFrame(
        {"trade_date": [date(2025, 1, 6)], "ts_code": ["000001.SZ"], "adjusted_open": [10.0]}
    )

    labels = build_forward_open_labels(members, market, calendar)

    assert not labels.loc[0, "label_valid"]
    assert pd.isna(labels.loc[0, "raw_return_5d"])
    assert labels.loc[0, "invalid_reason"] == "MISSING_EXIT_PRICE"
    assert labels.loc[0, "entry_realized_tradable"]
    assert not labels.loc[0, "exit_price_observed"]


def test_ordinary_label_build_rejects_2026_locked_signals() -> None:
    calendar = TradingCalendar.from_dates(
        [
            date(2026, 1, 2),
            date(2026, 1, 5),
            date(2026, 1, 6),
            date(2026, 1, 7),
            date(2026, 1, 8),
            date(2026, 1, 9),
            date(2026, 1, 12),
        ]
    )
    members = pd.DataFrame({"signal_date": [date(2026, 1, 2)], "ts_code": ["000001.SZ"]})
    market = pd.DataFrame(
        {
            "trade_date": [date(2026, 1, 5), date(2026, 1, 12)],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "adjusted_open": [10.0, 11.0],
        }
    )

    with pytest.raises(LockedLabelError, match="locked year"):
        build_forward_open_labels(members, market, calendar)


def test_authorized_locked_access_builds_labels_only_for_its_locked_year() -> None:
    calendar = TradingCalendar.from_dates(
        [
            date(2026, 1, 2),
            date(2026, 1, 5),
            date(2026, 1, 6),
            date(2026, 1, 7),
            date(2026, 1, 8),
            date(2026, 1, 9),
            date(2026, 1, 12),
        ]
    )
    members = pd.DataFrame({"signal_date": [date(2026, 1, 2)], "ts_code": ["000001.SZ"]})
    market = pd.DataFrame(
        {
            "trade_date": [date(2026, 1, 5), date(2026, 1, 12)],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "adjusted_open": [10.0, 11.0],
        }
    )

    labels = build_forward_open_labels(
        members,
        market,
        calendar,
        locked_access=forward_return.LockedLabelAccess.issue(
            locked_year=2026,
            authorization_id="user-approved-final-evaluation",
            signing_key=b"label-test-access-key",
        ),
    )

    assert labels.loc[0, "label_valid"]
    assert labels.loc[0, "raw_return_5d"] == pytest.approx(0.1)


def test_empty_members_return_an_empty_label_frame_with_the_output_schema() -> None:
    labels = build_forward_open_labels(
        pd.DataFrame(columns=["signal_date", "ts_code"]),
        pd.DataFrame(columns=["trade_date", "ts_code", "adjusted_open"]),
        TradingCalendar.from_dates([date(2025, 1, 2)]),
    )

    assert labels.empty
    assert {"signal_date", "ts_code", "label_valid"}.issubset(labels.columns)
