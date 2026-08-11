"""Secondary-source reconciliation that never overwrites the primary source."""

from __future__ import annotations

import hashlib
import math
from datetime import date
from typing import Any, Final, cast

import pandas as pd

_KEYS: Final = ("trade_date", "ts_code")
_PRICE_FIELDS: Final = frozenset({"open", "high", "low", "close"})
_VOLUME_FIELDS: Final = frozenset({"volume", "amount"})
_STATUS_FIELDS: Final = frozenset({"listing_status", "is_st", "is_suspended"})


def should_sample_history(trade_date: date, ts_code: str) -> bool:
    """Select a stable approximately-five-percent historical audit sample."""
    payload = f"{trade_date.isoformat()}|{ts_code}".encode()
    return int(hashlib.sha256(payload).hexdigest(), 16) % 100 < 5


def _numeric_matches(field: str, primary: Any, secondary: Any) -> bool:
    if pd.isna(primary) or pd.isna(secondary):
        return bool(pd.isna(primary)) and bool(pd.isna(secondary))
    primary_float = float(primary)
    secondary_float = float(secondary)
    if not math.isfinite(primary_float) or not math.isfinite(secondary_float):
        return primary_float == secondary_float
    absolute_difference = abs(primary_float - secondary_float)
    relative_difference = absolute_difference / max(abs(primary_float), 1e-12)
    if field in _PRICE_FIELDS:
        return absolute_difference <= 0.01 or relative_difference <= 5e-4
    return relative_difference <= 1e-3


def reconcile_market(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    """Return field-level source discrepancies; neither input is modified."""
    for label, frame in (("primary", primary), ("secondary", secondary)):
        missing = sorted(set(_KEYS).difference(frame.columns))
        if missing:
            raise ValueError(f"{label} is missing columns: {missing}")
        if frame.duplicated(list(_KEYS)).any():
            raise ValueError(f"{label} has duplicate reconciliation keys")
    secondary_rows = {
        (pd.Timestamp(row["trade_date"]).date(), str(row["ts_code"])): row
        for row in cast(list[dict[str, Any]], secondary.to_dict(orient="records"))
    }
    comparable_fields = sorted(
        (_PRICE_FIELDS | _VOLUME_FIELDS | _STATUS_FIELDS)
        .intersection(primary.columns)
        .intersection(secondary.columns)
    )
    findings: list[dict[str, object]] = []
    for primary_row in cast(list[dict[str, Any]], primary.to_dict(orient="records")):
        key = (pd.Timestamp(primary_row["trade_date"]).date(), str(primary_row["ts_code"]))
        secondary_row = secondary_rows.get(key)
        if secondary_row is None:
            findings.append(
                {
                    "trade_date": key[0],
                    "ts_code": key[1],
                    "field": "ROW",
                    "reason": "MISSING_SECONDARY",
                }
            )
            continue
        for field in comparable_fields:
            primary_value = primary_row[field]
            secondary_value = secondary_row[field]
            matches = (
                _numeric_matches(field, primary_value, secondary_value)
                if field in _PRICE_FIELDS | _VOLUME_FIELDS
                else primary_value == secondary_value
            )
            if not matches:
                findings.append(
                    {
                        "trade_date": key[0],
                        "ts_code": key[1],
                        "field": field,
                        "reason": "VALUE_MISMATCH",
                        "primary_value": primary_value,
                        "secondary_value": secondary_value,
                    }
                )
    return pd.DataFrame(findings)
