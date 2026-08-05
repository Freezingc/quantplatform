"""Audit historical BJ coverage across Tushare market endpoints.

This is a read-only availability check.  It records endpoint row counts and
date ranges only; it never uses YesLab values as a market-data fallback.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ENDPOINTS = ("daily", "daily_basic", "stk_factor")
LOCKED_TEST_START = date(2026, 1, 1)


def _parse_date(value: str) -> date:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc
    if parsed >= LOCKED_TEST_START:
        raise argparse.ArgumentTypeError("2026 locked-test data is not allowed")
    return parsed


def summarize_frame(frame: pd.DataFrame | None) -> dict[str, object]:
    if frame is None or frame.empty:
        return {
            "status": "NO_ROWS",
            "rows": 0,
            "finite_trade_date_rows": 0,
            "min_trade_date": None,
            "max_trade_date": None,
        }
    if "trade_date" not in frame.columns:
        raise ValueError("Tushare endpoint response is missing trade_date")
    dates = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    finite = dates.notna()
    finite_dates = dates.loc[finite]
    return {
        "status": "ROWS_PRESENT",
        "rows": int(len(frame)),
        "finite_trade_date_rows": int(finite.sum()),
        "min_trade_date": finite_dates.min().date().isoformat() if finite.any() else None,
        "max_trade_date": finite_dates.max().date().isoformat() if finite.any() else None,
    }


def _fetch_endpoint(pro: object, endpoint: str, code: str, start: date, end: date) -> pd.DataFrame:
    method = getattr(pro, endpoint)
    result = method(
        ts_code=code,
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
    )
    if result is None:
        return pd.DataFrame()
    if not isinstance(result, pd.DataFrame):
        raise TypeError(f"{endpoint} response is not a DataFrame")
    return result


def audit_code(pro: object, code: str, start: date, end: date) -> dict[str, object]:
    endpoints: dict[str, object] = {}
    for endpoint in ENDPOINTS:
        try:
            endpoints[endpoint] = summarize_frame(
                _fetch_endpoint(pro, endpoint, code, start, end)
            )
        except Exception as exc:  # endpoint availability is the audit result
            endpoints[endpoint] = {
                "status": "ERROR",
                "rows": 0,
                "finite_trade_date_rows": 0,
                "min_trade_date": None,
                "max_trade_date": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return {"code": str(code), "endpoints": endpoints}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", action="append", required=True)
    parser.add_argument("--start", type=_parse_date, required=True)
    parser.add_argument("--end", type=_parse_date, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("--start must be on or before --end")
    if any(not str(code).endswith(".BJ") for code in args.code):
        parser.error("--code values must be .BJ security codes")
    args.code = tuple(sorted(set(map(str, args.code))))
    return args


def run_audit(args: argparse.Namespace, *, pro: object) -> dict[str, object]:
    return {
        "scope": "TUSHARE_BJ_LEGACY_ENDPOINT_COVERAGE_V1",
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "codes": list(args.code),
        "database_writes": False,
        "locked_test_read": False,
        "endpoints": list(ENDPOINTS),
        "rows": [audit_code(pro, code, args.start, args.end) for code in args.code],
    }


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    import tushare as ts  # type: ignore[import-untyped]

    token = os.getenv("QRV2_TUSHARE_TOKEN") or os.getenv("TUSHARE_TOKEN", "")
    if not token:
        raise RuntimeError("Tushare token is required")
    payload = run_audit(args, pro=ts.pro_api(token))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    return payload


if __name__ == "__main__":
    main()
