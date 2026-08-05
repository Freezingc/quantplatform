"""Audit a ``pre_close``-corrected jump-deviation factor variant.

The existing ``SkipDemeanAbsCalculator`` derives the previous close with
``close.shift(1)``.  This isolated audit uses Tushare's point-in-time
``pre_close`` field instead and compares the resulting candidate with stored
YesLab observations. It never overwrites the source factor or writes V2.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import audit_formula_replay_consistency as replay
from audit_formula_replay_tushare import (
    _fetch_tushare_market,
    _query_open_dates,
    _query_v2_universe_codes,
)

LOCKED_TEST_START = date(2026, 1, 1)
FACTOR_ID = "yeslab_155"


def compute_preclose_skip_variant(market: pd.DataFrame) -> pd.DataFrame:
    """Compute ``abs(open/pre_close - cross_section_mean)`` in long form."""

    required = {"trade_date", "ts_code", "open", "pre_close"}
    if missing := sorted(required.difference(market.columns)):
        raise ValueError(f"market is missing columns: {missing}")
    frame = market.loc[:, sorted(required)].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.normalize()
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
    frame["pre_close"] = pd.to_numeric(frame["pre_close"], errors="coerce")
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("market contains duplicate trade-date/security keys")
    opens = frame.pivot(index="trade_date", columns="ts_code", values="open").sort_index()
    pre_close = frame.pivot(
        index="trade_date", columns="ts_code", values="pre_close"
    ).reindex(columns=opens.columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        skip = opens / pre_close.replace(0, np.nan)
    result = skip.sub(skip.mean(axis=1), axis=0).abs().replace([np.inf, -np.inf], np.nan)
    long = (
        result.rename_axis("trade_date")
        .stack(future_stack=True)
        .rename("replay_value")
        .reset_index()
    )
    return long.loc[:, ["trade_date", "ts_code", "replay_value"]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2025, 1, 2))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 1, 31))
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/qrv2-tushare-replay-cache"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.start > args.end:
        raise ValueError("start date must not be after end date")
    if args.end >= LOCKED_TEST_START:
        raise ValueError("pre-close audit cannot read 2026 locked-test data")
    return args


def _connect() -> object:
    import dolphindb as ddb  # type: ignore[import-untyped]

    client = ddb.session()
    client.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    return client


def run_audit(args: argparse.Namespace, *, client: object, tushare_token: str) -> dict[str, object]:
    definition = replay.REPLAY_DEFINITIONS[FACTOR_ID]
    codes = tuple(_query_v2_universe_codes(client, args.start, args.end))
    if not codes:
        raise RuntimeError("V2 universe query returned no securities")
    market_start = args.start - timedelta(days=30)
    open_dates = _query_open_dates(client, market_start, args.end)
    market = _fetch_tushare_market(
        codes,
        market_start,
        args.end,
        token=tushare_token,
        open_dates=open_dates,
        cache_dir=args.cache_dir,
        include_daily_basic=False,
    )
    source = replay._query_source(client, definition, args.start, args.end)
    candidate = compute_preclose_skip_variant(market)
    candidate = candidate.loc[
        candidate["trade_date"].between(pd.Timestamp(args.start), pd.Timestamp(args.end))
    ].copy()
    comparison = replay.compare_replay_frames(source, candidate)
    payload = {
        "scope": "FORMULA_REPAIR_PRECLOSE_VARIANT_AUDIT_V1",
        "factor_id": FACTOR_ID,
        "factor_name": definition.factor_name,
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "market_start": market_start.isoformat(),
        "market_source": "Tushare.daily.pre_close",
        "code_count": len(codes),
        "market_rows": len(market),
        "database_writes": False,
        "locked_test_read": False,
        "candidate_formula": "abs(open / pre_close - cross_section_mean(open / pre_close))",
        **comparison,
    }
    return payload


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    payload = run_audit(
        args,
        client=_connect(),
        tushare_token=os.getenv("QRV2_TUSHARE_TOKEN") or os.getenv("TUSHARE_TOKEN", ""),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
