"""Point-in-time security master (SCD2) construction."""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Final

import pandas as pd


class SecurityMasterConflict(ValueError):
    """Raised when source security attributes imply two states on one date."""


_SECURITY_COLUMNS: Final = frozenset(
    {
        "ts_code",
        "exchange",
        "market",
        "list_date",
        "delist_date",
        "name",
        "list_status",
    }
)
_NAME_COLUMNS: Final = frozenset({"ts_code", "name", "start_date", "end_date"})
_BOARD_MAP: Final = {
    ("SSE", "主板"): "SH_MAIN",
    ("SSE", "科创板"): "STAR",
    ("SZSE", "主板"): "SZ_MAIN",
    ("SZSE", "创业板"): "CHINEXT",
    ("BSE", "北交所"): "BEIJING",
}


def _as_date(value: Any) -> date | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def _next_day(value: date) -> date:
    return value + timedelta(days=1)


def _board(exchange: object, market: object) -> str:
    return _BOARD_MAP.get((str(exchange), str(market)), "OTHER")


_ST_DESIGNATION = re.compile(r"^(\*ST|S\*ST|SST|ST)(?![A-Z0-9])")


def _is_st_name(name: str | None) -> bool:
    return name is not None and _ST_DESIGNATION.match(name.strip().upper()) is not None


def _validate_columns(frame: pd.DataFrame, required: frozenset[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _name_at(
    records: pd.DataFrame,
    ts_code: str,
    effective_date: date,
) -> str | None:
    active_names: set[str] = set()
    for record in records.itertuples(index=False):
        start = _as_date(record.start_date)
        end = _as_date(record.end_date)
        if start is None:
            raise SecurityMasterConflict(f"{ts_code} has a name record without start_date")
        if start <= effective_date and (end is None or effective_date <= end):
            active_names.add(str(record.name))
    if len(active_names) > 1:
        raise SecurityMasterConflict(
            f"{ts_code} has conflicting active names on {effective_date.isoformat()}"
        )
    return next(iter(active_names), None)


def _boundaries(
    records: pd.DataFrame,
    list_date: date,
    terminal_date: date | None,
) -> list[date]:
    boundaries = {list_date}
    if terminal_date is not None:
        boundaries.add(terminal_date)
    for record in records.itertuples(index=False):
        start = _as_date(record.start_date)
        end = _as_date(record.end_date)
        if start is None:
            continue
        for boundary in (start, _next_day(end) if end is not None else None):
            if boundary is None or boundary < list_date:
                continue
            if terminal_date is None or boundary <= terminal_date:
                boundaries.add(boundary)
    return sorted(boundaries)


def build_security_master_scd(
    securities: pd.DataFrame,
    name_history: pd.DataFrame,
) -> pd.DataFrame:
    """Build non-overlapping SCD2 intervals from source master and name history.

    ``stock_info_table.name`` is deliberately not used to populate historical
    name gaps: that field is a current snapshot and would leak future changes.
    A historical name only appears during its explicit ``start_date/end_date``
    interval. Intervals are closed-open; source ``end_date`` is inclusive.
    """
    _validate_columns(securities, _SECURITY_COLUMNS, "securities")
    _validate_columns(name_history, _NAME_COLUMNS, "name_history")
    if securities["ts_code"].duplicated().any():
        raise SecurityMasterConflict("securities contains duplicate ts_code values")

    normalized_names = name_history.loc[:, list(_NAME_COLUMNS)].copy()
    rows: list[dict[str, object]] = []
    for security in securities.sort_values("ts_code").itertuples(index=False):
        ts_code = str(security.ts_code)
        list_date = _as_date(security.list_date)
        delist_date = _as_date(security.delist_date)
        if list_date is None:
            raise SecurityMasterConflict(f"{ts_code} has no list_date")
        if delist_date is not None and delist_date < list_date:
            raise SecurityMasterConflict(f"{ts_code} delists before listing")
        terminal_date = _next_day(delist_date) if delist_date is not None else None
        records = normalized_names.loc[normalized_names["ts_code"].astype(str) == ts_code]
        boundaries = _boundaries(records, list_date, terminal_date)
        for index, valid_from in enumerate(boundaries):
            if terminal_date is not None and valid_from == terminal_date:
                continue
            valid_to = boundaries[index + 1] if index + 1 < len(boundaries) else terminal_date
            security_name = _name_at(records, ts_code, valid_from)
            rows.append(
                {
                    "ts_code": ts_code,
                    "exchange": str(security.exchange),
                    "board": _board(security.exchange, security.market),
                    "list_date": list_date,
                    "delist_date": delist_date,
                    "security_name": security_name,
                    "is_st": _is_st_name(security_name),
                    "listing_status": str(security.list_status),
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=[
                "ts_code",
                "exchange",
                "board",
                "list_date",
                "delist_date",
                "security_name",
                "is_st",
                "listing_status",
                "valid_from",
                "valid_to",
            ]
        )
    return result.sort_values(["ts_code", "valid_from"], ignore_index=True)
