"""Golden boundary samples for the T+1..T+6 adjusted-open label builder (B8)."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from research_v2.labels.forward_return import build_forward_open_labels
from research_v2.pit.calendar import TradingCalendar

_DATA = Path(__file__).parent / "data"


def _load_calendar() -> TradingCalendar:
    frame = pd.read_csv(_DATA / "pit_label_calendar.csv", dtype=str)
    return TradingCalendar.from_dates(
        [date.fromisoformat(value) for value in frame["trade_date"]]
    )


def test_golden_pit_labels_match_independent_reference() -> None:
    calendar = _load_calendar()
    members = pd.read_csv(_DATA / "pit_label_members.csv", dtype=str)
    market = pd.read_csv(
        _DATA / "pit_label_market.csv",
        dtype={"trade_date": str, "ts_code": str},
    )
    market["adjusted_open"] = pd.to_numeric(market["adjusted_open"], errors="coerce")
    expected = json.loads((_DATA / "expected_pit_labels.json").read_text(encoding="utf-8"))

    labels = build_forward_open_labels(members, market, calendar)

    assert labels.shape[0] == len(expected["rows"])
    for index, expected_row in enumerate(expected["rows"]):
        actual = labels.iloc[index]
        assert str(actual["signal_date"]) == expected_row["signal_date"]
        assert str(actual["ts_code"]) == expected_row["ts_code"]
        assert str(actual["entry_date"]) == str(expected_row["entry_date"])
        assert str(actual["exit_date"]) == str(expected_row["exit_date"])
        assert str(actual["label_available_date"]) == str(
            expected_row["label_available_date"]
        )
        expected_return = expected_row["raw_return_5d"]
        if expected_return is None:
            assert isinstance(actual["raw_return_5d"], float)
            assert math.isnan(actual["raw_return_5d"])
        else:
            assert actual["raw_return_5d"] == pytest.approx(expected_return, abs=1e-12)
        assert bool(actual["label_valid"]) is expected_row["label_valid"]
        reason = actual["invalid_reason"]
        assert (None if pd.isna(reason) else str(reason)) == expected_row["invalid_reason"]
        assert bool(actual["entry_realized_tradable"]) is expected_row[
            "entry_realized_tradable"
        ]
        assert bool(actual["exit_price_observed"]) is expected_row["exit_price_observed"]


def test_golden_fixture_covers_every_invalid_boundary_class() -> None:
    expected = json.loads((_DATA / "expected_pit_labels.json").read_text(encoding="utf-8"))
    reasons = {row["invalid_reason"] for row in expected["rows"]}
    assert reasons == {
        None,
        "MISSING_ENTRY_PRICE",
        "MISSING_EXIT_PRICE",
        "INVALID_ENTRY_PRICE",
        "INVALID_EXIT_PRICE",
        "INSUFFICIENT_CALENDAR_COVERAGE",
    }
