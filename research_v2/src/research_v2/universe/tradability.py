"""Signal-date tradability derived only from information available on that date."""

from __future__ import annotations

from datetime import date
from typing import Any, Final, cast

import pandas as pd

from research_v2.pit.calendar import TradingCalendar
from research_v2.pit.security_master import SecurityMasterConflict

_MARKET_COLUMNS: Final = frozenset({"trade_date", "ts_code", "valid_days_20", "adv20"})
_SCD_COLUMNS: Final = frozenset(
    {"ts_code", "list_date", "delist_date", "is_st", "valid_from", "valid_to"}
)
_SUSPENSION_COLUMNS: Final = frozenset({"trade_date", "ts_code", "suspend_type"})
_NOT_LISTED: Final = 1
_ST: Final = 2
_SUSPENDED: Final = 4
_YOUNG_LISTING: Final = 8
_INSUFFICIENT_HISTORY: Final = 16


def _as_date(value: Any) -> date | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def _validate_columns(frame: pd.DataFrame, required: frozenset[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _active_security_states(
    security_scd: pd.DataFrame,
    trade_date: date,
) -> dict[str, dict[str, Any]]:
    normalized = security_scd.copy(deep=True)
    normalized["ts_code"] = normalized["ts_code"].astype(str)
    normalized["valid_from"] = normalized["valid_from"].map(_as_date)
    normalized["valid_to"] = normalized["valid_to"].map(_as_date)
    active = normalized.loc[
        normalized["valid_from"].notna()
        & (normalized["valid_from"] <= trade_date)
        & (normalized["valid_to"].isna() | (trade_date < normalized["valid_to"]))
    ]
    duplicate_codes = active.loc[active["ts_code"].duplicated(), "ts_code"]
    if not duplicate_codes.empty:
        raise SecurityMasterConflict(
            f"multiple SCD states on {trade_date.isoformat()}: {sorted(duplicate_codes.unique())}"
        )
    return {
        str(record["ts_code"]): record
        for record in cast(list[dict[str, Any]], active.to_dict(orient="records"))
    }


def build_tradability_daily(
    market: pd.DataFrame,
    security_scd: pd.DataFrame,
    suspensions: pd.DataFrame,
    calendar: TradingCalendar,
    *,
    minimum_listing_trading_days: int = 252,
    minimum_valid_days_20: int = 15,
) -> pd.DataFrame:
    """Build daily tradability without consulting any future market observations."""
    _validate_columns(market, _MARKET_COLUMNS, "market")
    _validate_columns(security_scd, _SCD_COLUMNS, "security_scd")
    _validate_columns(suspensions, _SUSPENSION_COLUMNS, "suspensions")
    if market.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("market contains duplicate trade_date/ts_code keys")
    if minimum_listing_trading_days < 1 or minimum_valid_days_20 < 1:
        raise ValueError("minimum history thresholds must be positive")

    suspended_keys = {
        (_as_date(row["trade_date"]), str(row["ts_code"]))
        for row in cast(list[dict[str, Any]], suspensions.to_dict(orient="records"))
        if _as_date(row["trade_date"]) is not None
        and "S" in str(row["suspend_type"]).upper()
    }
    rows: list[dict[str, object]] = []
    sorted_market = market.sort_values(["trade_date", "ts_code"])
    trade_dates: set[date] = set()
    for value in sorted_market["trade_date"]:
        trade_date = _as_date(value)
        if trade_date is None:
            raise ValueError("market contains a null trade_date")
        if not calendar.window(trade_date, trade_date):
            raise ValueError(f"market trade_date is not in calendar: {trade_date.isoformat()}")
        trade_dates.add(trade_date)
    states_by_trade_date = {
        trade_date: _active_security_states(security_scd, trade_date)
        for trade_date in trade_dates
    }
    for market_row in cast(list[dict[str, Any]], sorted_market.to_dict(orient="records")):
        trade_date = _as_date(market_row["trade_date"])
        if trade_date is None:
            raise ValueError("market contains a null trade_date")
        ts_code = str(market_row["ts_code"])
        state = states_by_trade_date[trade_date].get(ts_code)
        is_listed = state is not None
        is_st = bool(state["is_st"]) if state is not None else False
        list_date = _as_date(state["list_date"]) if state is not None else None
        listing_age = calendar.count(list_date, trade_date) if list_date is not None else 0
        is_suspended = (trade_date, ts_code) in suspended_keys
        valid_days_20 = (
            0 if pd.isna(market_row["valid_days_20"]) else int(market_row["valid_days_20"])
        )
        adv20 = float(market_row["adv20"]) if pd.notna(market_row["adv20"]) else float("nan")
        reason_bitmap = 0
        if not is_listed:
            reason_bitmap |= _NOT_LISTED
        else:
            if is_st:
                reason_bitmap |= _ST
            if is_suspended:
                reason_bitmap |= _SUSPENDED
            if listing_age < minimum_listing_trading_days:
                reason_bitmap |= _YOUNG_LISTING
            if valid_days_20 < minimum_valid_days_20:
                reason_bitmap |= _INSUFFICIENT_HISTORY
        rows.append(
            {
                "trade_date": trade_date,
                "ts_code": ts_code,
                "is_st": is_st,
                "is_suspended": is_suspended,
                "is_listed": is_listed,
                "listing_age_trading_days": listing_age,
                "valid_days_20": valid_days_20,
                "adv20": adv20,
                "is_eligible_pre_liquidity": reason_bitmap == 0,
                "reason_bitmap": reason_bitmap,
            }
        )
    return pd.DataFrame(rows).sort_values(["trade_date", "ts_code"], ignore_index=True)
