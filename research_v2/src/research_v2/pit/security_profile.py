"""Conservative security-profile fallbacks derived from observable market history."""

from __future__ import annotations

import hashlib
from datetime import date

import pandas as pd

from research_v2.contracts.source import SourceBatch

_REQUIRED_MARKET_COLUMNS = frozenset({"ts_code", "trade_date"})


def _exchange_and_market(ts_code: str) -> tuple[str, str]:
    symbol, separator, exchange = ts_code.partition(".")
    if separator != ".":
        raise ValueError(f"invalid ts_code for derived security profile: {ts_code}")
    if exchange == "SZ" and symbol.startswith("300"):
        return "SZSE", "创业板"
    if exchange == "SZ" and symbol.startswith(("000", "001", "002", "003")):
        return "SZSE", "主板"
    if exchange == "SH" and symbol.startswith("688"):
        return "SSE", "科创板"
    if exchange == "SH" and symbol.startswith("6"):
        return "SSE", "主板"
    if exchange == "BJ":
        return "BSE", "北交所"
    raise ValueError(f"unsupported board for derived security profile: {ts_code}")


def build_market_derived_security_profile(
    market_history: SourceBatch,
    *,
    as_of_date: date,
    ts_codes: set[str],
) -> SourceBatch:
    """Build profiles only for observed mainland A-share codes missing from both masters.

    The derived ``list_date`` is deliberately the earliest source observation, so
    listing-age based filters remain conservative. This is a narrow completeness
    fallback, not a replacement for an authoritative security master.
    """
    if market_history.table_name != "stock_factor_daily":
        raise ValueError("market-derived profiles require a stock_factor_daily batch")
    missing_columns = sorted(_REQUIRED_MARKET_COLUMNS.difference(market_history.frame.columns))
    if missing_columns:
        raise ValueError(f"market history is missing columns: {missing_columns}")
    requested_codes = {str(code) for code in ts_codes}
    if not requested_codes:
        raise ValueError("at least one ts_code is required for a derived security profile")
    history = market_history.frame.loc[:, ["ts_code", "trade_date"]].copy()
    history["ts_code"] = history["ts_code"].astype(str)
    history["trade_date"] = pd.to_datetime(history["trade_date"], errors="raise").dt.date
    history = history.loc[
        history["ts_code"].isin(requested_codes) & (history["trade_date"] <= as_of_date)
    ]
    observed_codes = set(history["ts_code"])
    missing_codes = sorted(requested_codes.difference(observed_codes))
    if missing_codes:
        raise ValueError(f"requested codes lack market observations: {missing_codes}")

    rows: list[dict[str, object]] = []
    for ts_code, records in history.groupby("ts_code", sort=True):
        exchange, market = _exchange_and_market(str(ts_code))
        rows.append(
            {
                "ts_code": str(ts_code),
                "exchange": exchange,
                "market": market,
                "list_date": min(records["trade_date"]),
                "delist_date": None,
                "name": None,
                "list_status": "L",
            }
        )
    profile = pd.DataFrame(rows)
    schema_payload = "|".join(f"{column}:{profile[column].dtype}" for column in profile.columns)
    schema_fingerprint = hashlib.sha256(schema_payload.encode()).hexdigest()
    watermark_payload = "|".join(
        [
            "DERIVED_MARKET_PROFILE_V1",
            market_history.watermark_id,
            as_of_date.isoformat(),
            ",".join(sorted(requested_codes)),
            schema_fingerprint,
        ]
    )
    return SourceBatch(
        frame=profile,
        source_name="DerivedFromYesLabMarket",
        table_name="derived_market_security_profile",
        watermark_id=hashlib.sha256(watermark_payload.encode()).hexdigest(),
        max_observation_date=as_of_date,
        source_row_count=len(profile),
        schema_fingerprint=schema_fingerprint,
        extracted_at=market_history.extracted_at,
    )
