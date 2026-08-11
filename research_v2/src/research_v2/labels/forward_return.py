"""Five-trading-day adjusted-open return labels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from research_v2.pit.calendar import TradingCalendar, TradingCalendarBoundaryError


class LockedLabelError(ValueError):
    """Raised when ordinary materialization attempts to build locked labels."""


@dataclass(frozen=True)
class LockedLabelAccess:
    """Explicit audit capability for the one isolated locked-label workflow."""

    locked_year: int
    authorization_id: str

    def __post_init__(self) -> None:
        if self.locked_year < _LOCKED_YEAR:
            raise LockedLabelError("locked access cannot target a non-locked year")
        if not self.authorization_id.strip():
            raise LockedLabelError("locked access requires a non-empty authorization_id")


_LOCKED_YEAR = 2026
_OUTPUT_COLUMNS = [
    "signal_date",
    "ts_code",
    "entry_date",
    "exit_date",
    "label_available_date",
    "raw_return_5d",
    "label_valid",
    "invalid_reason",
    "entry_realized_tradable",
    "exit_price_observed",
]


def _is_valid_price(value: Any) -> bool:
    return (
        value is not None
        and pd.notna(value)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def build_forward_open_labels(
    members: pd.DataFrame,
    market: pd.DataFrame,
    calendar: TradingCalendar,
    *,
    locked_year: int = _LOCKED_YEAR,
    locked_access: LockedLabelAccess | None = None,
) -> pd.DataFrame:
    """Label each signal T with the raw return from T+1 open to T+6 open."""
    required_members = {"signal_date", "ts_code"}
    required_market = {"trade_date", "ts_code", "adjusted_open"}
    if missing := sorted(required_members.difference(members.columns)):
        raise ValueError(f"members are missing columns: {missing}")
    if missing := sorted(required_market.difference(market.columns)):
        raise ValueError(f"market is missing columns: {missing}")
    if market.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("market contains duplicate trade_date/ts_code keys")
    if locked_access is not None and locked_access.locked_year != locked_year:
        raise LockedLabelError("locked access year must match the configured lock year")
    prices = {
        (pd.Timestamp(row["trade_date"]).date(), str(row["ts_code"])): row["adjusted_open"]
        for row in cast(list[dict[str, Any]], market.to_dict(orient="records"))
    }
    rows: list[dict[str, object]] = []
    for member in cast(list[dict[str, Any]], members.to_dict(orient="records")):
        signal_date = pd.Timestamp(member["signal_date"]).date()
        if locked_access is not None and signal_date.year != locked_access.locked_year:
            raise LockedLabelError("locked access can only materialize its locked year")
        if signal_date.year >= locked_year and locked_access is None:
            raise LockedLabelError(
                f"ordinary label materialization rejects locked year {locked_year} and later"
            )
        ts_code = str(member["ts_code"])
        try:
            entry_date = calendar.shift(signal_date, 1)
            exit_date = calendar.shift(signal_date, 6)
        except TradingCalendarBoundaryError:
            rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": ts_code,
                    "entry_date": None,
                    "exit_date": None,
                    "label_available_date": None,
                    "raw_return_5d": float("nan"),
                    "label_valid": False,
                    "invalid_reason": "INSUFFICIENT_CALENDAR_COVERAGE",
                    "entry_realized_tradable": False,
                    "exit_price_observed": False,
                }
            )
            continue
        entry_price = prices.get((entry_date, ts_code))
        exit_price = prices.get((exit_date, ts_code))
        entry_realized_tradable = _is_valid_price(entry_price)
        exit_price_observed = _is_valid_price(exit_price)
        invalid_reason: str | None = None
        if entry_price is None or pd.isna(entry_price):
            invalid_reason = "MISSING_ENTRY_PRICE"
        elif exit_price is None or pd.isna(exit_price):
            invalid_reason = "MISSING_EXIT_PRICE"
        elif not math.isfinite(float(entry_price)) or float(entry_price) <= 0:
            invalid_reason = "INVALID_ENTRY_PRICE"
        elif not math.isfinite(float(exit_price)) or float(exit_price) <= 0:
            invalid_reason = "INVALID_EXIT_PRICE"
        raw_return = float("nan")
        if invalid_reason is None:
            assert entry_price is not None
            assert exit_price is not None
            raw_return = float(exit_price) / float(entry_price) - 1
        rows.append(
            {
                "signal_date": signal_date,
                "ts_code": ts_code,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "label_available_date": exit_date,
                "raw_return_5d": raw_return,
                "label_valid": invalid_reason is None,
                "invalid_reason": invalid_reason,
                "entry_realized_tradable": entry_realized_tradable,
                "exit_price_observed": exit_price_observed,
            }
        )
    if not rows:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    return pd.DataFrame(rows).sort_values(["signal_date", "ts_code"], ignore_index=True)
