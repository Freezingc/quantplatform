"""Replay selected YesLab factors with Tushare as the canonical market source.

This is a bounded, read-only reconciliation command.  It deliberately limits
the stock universe so nested rolling formulas can be checked without loading
the entire A-share panel into memory.  The stored factor observations remain
the YesLab comparison target; only the market inputs used by the calculator
are replaced with Tushare ``daily`` observations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import audit_formula_replay_consistency as replay
import numpy as np
import pandas as pd

LOCKED_TEST_START = date(2026, 1, 1)
SUPPORTED_FACTOR_IDS = ("yeslab_398", "yeslab_438", "yeslab_451", "yeslab_464")
REPO_ROOT = Path(__file__).resolve().parents[2]
CALCULATOR_ROOT = REPO_ROOT / "数据库"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-id", action="append")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 9, 14))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2020, 9, 25))
    parser.add_argument("--max-ts-codes", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.factor_id = tuple(args.factor_id or SUPPORTED_FACTOR_IDS)
    unsupported = sorted(set(args.factor_id).difference(SUPPORTED_FACTOR_IDS))
    if unsupported:
        raise ValueError(f"unsupported Tushare replay factor IDs: {unsupported}")
    if not args.factor_id or len(set(args.factor_id)) != len(args.factor_id):
        raise ValueError("factor IDs must be non-empty and unique")
    if args.start > args.end:
        raise ValueError("start date must not be after end date")
    if args.end >= LOCKED_TEST_START:
        raise ValueError("Tushare formula replay cannot read 2026 locked-test data")
    if args.max_ts_codes < 1:
        raise ValueError("max-ts-codes must be positive")
    return args


def normalize_tushare_market(raw: pd.DataFrame, *, codes: Sequence[str]) -> pd.DataFrame:
    required = {"ts_code", "trade_date", "open", "low", "amount", "pct_chg"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Tushare daily response is missing columns: {missing}")
    frame = raw.copy()
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"].astype(str), format="%Y%m%d", errors="coerce"
    ).dt.normalize()
    for column in ("open", "low", "amount", "pct_chg", "close"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[frame["ts_code"].isin(set(map(str, codes)))].copy()
    frame = frame.dropna(subset=["trade_date", "ts_code"])
    frame = frame.drop_duplicates(["trade_date", "ts_code"], keep="last")
    columns = ["trade_date", "ts_code"] + [
        column for column in ("open", "low", "close", "amount", "pct_chg") if column in frame
    ]
    return frame.loc[:, columns].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


class TushareMarketSession:
    """DolphinDB session wrapper that substitutes only stock market queries."""

    def __init__(self, base: object, market: pd.DataFrame, codes: Sequence[str]) -> None:
        self._base = base
        self._market = market
        self._codes = set(map(str, codes))

    def run(self, script: str) -> object:
        if 'loadTable("dfs://YesLab", "stock_factor_daily")' in script or (
            "loadTable('dfs://YesLab','stock_factor_daily')" in script
        ):
            return self._market.copy(deep=True)
        result = self._base.run(script)
        if isinstance(result, pd.DataFrame) and "ts_code" in result.columns:
            result = result.loc[result["ts_code"].astype(str).isin(self._codes)].copy()
        return result


def _query_codes(
    client: object, definition: replay.ReplayDefinition, start: date, end: date, limit: int
) -> tuple[str, ...]:
    result = client.run(
        "select distinct ts_code from "
        f"loadTable('dfs://YesLab','{definition.store_factor}') "
        f"where factor_name == '{definition.factor_name}' "
        f"and trade_date between date('{start:%Y.%m.%d}') "
        f"and date('{end:%Y.%m.%d}') order by ts_code"
    )
    if not isinstance(result, pd.DataFrame) or "ts_code" not in result.columns:
        raise RuntimeError("factor universe query did not return ts_code")
    return tuple(sorted(result["ts_code"].astype(str).dropna().unique())[:limit])


def _calculator_market_start(
    client: object, definition: replay.ReplayDefinition, start: date, end: date
) -> date:
    if str(CALCULATOR_ROOT) not in sys.path:
        sys.path.insert(0, str(CALCULATOR_ROOT))
    module = importlib.import_module(f"factor_calculators.{definition.calculator_module}")
    calculator_class = getattr(module, definition.calculator_class)
    calculator = calculator_class(client, definition.effective_replay_lookback_days)
    processed = calculator.preprocess_time_range(
        start.strftime("%Y.%m.%d"), end.strftime("%Y.%m.%d")
    )
    return date.fromisoformat(processed.replace(".", "-")) - timedelta(days=30)


def _fetch_tushare_market(
    codes: Sequence[str], start: date, end: date, *, token: str
) -> pd.DataFrame:
    import tushare as ts  # type: ignore[import-untyped]

    if not token.strip():
        raise ValueError("QRV2_TUSHARE_TOKEN is required for canonical market replay")
    pro = ts.pro_api(token)
    frames: list[pd.DataFrame] = []
    for code in codes:
        result = pro.daily(
            ts_code=code,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )
        if isinstance(result, pd.DataFrame) and not result.empty:
            frames.append(result)
    if not frames:
        raise RuntimeError("Tushare daily returned no market rows")
    return normalize_tushare_market(pd.concat(frames, ignore_index=True), codes=codes)


def run_audit(args: argparse.Namespace, *, client: object, tushare_token: str) -> dict[str, object]:
    definitions = [replay.REPLAY_DEFINITIONS[factor_id] for factor_id in args.factor_id]
    code_sets = [
        _query_codes(client, definition, args.start, args.end, args.max_ts_codes)
        for definition in definitions
    ]
    codes = tuple(sorted(set().union(*map(set, code_sets))))[: args.max_ts_codes]
    market_start = min(
        _calculator_market_start(client, definition, args.start, args.end)
        for definition in definitions
    )
    market = _fetch_tushare_market(codes, market_start, args.end, token=tushare_token)
    wrapped = TushareMarketSession(client, market, codes)
    rows: list[dict[str, object]] = []
    for factor_id in args.factor_id:
        definition = replay.REPLAY_DEFINITIONS[factor_id]
        comparison = replay.compare_replay_frames(
            replay._query_source(wrapped, definition, args.start, args.end),
            replay._replay_factor(wrapped, definition, args.start, args.end),
        )
        rows.append(
            {
                "factor_id": factor_id,
                "factor_name": definition.factor_name,
                "lookback_days": definition.lookback_days,
                "replay_lookback_days": definition.effective_replay_lookback_days,
                "max_ts_codes": args.max_ts_codes,
                **comparison,
            }
        )
    market_hash = hashlib.sha256(
        pd.util.hash_pandas_object(market, index=False).to_numpy(dtype="uint64").tobytes()
    ).hexdigest()
    return {
        "scope": "FORMULA_REPLAY_CONSISTENCY_TUSHARE_MARKET_V1",
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "market_start": market_start.isoformat(),
        "factor_ids": list(args.factor_id),
        "max_ts_codes": args.max_ts_codes,
        "market_source": "Tushare.daily",
        "market_rows": len(market),
        "market_finite_rows": int(
            np.isfinite(market.select_dtypes(include=[np.number])).to_numpy().sum()
        ),
        "market_hash": market_hash,
        "database_writes": False,
        "locked_test_read": False,
        "rows": rows,
    }


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


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    payload = run_audit(
        args,
        client=_connect(),
        tushare_token=os.getenv("QRV2_TUSHARE_TOKEN") or os.getenv("TUSHARE_TOKEN", ""),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, flush=True)
    return payload


if __name__ == "__main__":
    main()
