"""Audit finite factor observations that have no raw market business key."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import audit_formula_replay_consistency as replay

LOCKED_TEST_START = date(2026, 1, 1)


def compare_factor_market_coverage(
    source: pd.DataFrame, raw_market: pd.DataFrame
) -> dict[str, object]:
    required_source = {"trade_date", "ts_code", "value"}
    required_market = {"trade_date", "ts_code"}
    if missing := sorted(required_source.difference(source.columns)):
        raise ValueError(f"factor source is missing columns: {missing}")
    if missing := sorted(required_market.difference(raw_market.columns)):
        raise ValueError(f"raw market is missing columns: {missing}")
    source = source.loc[:, ["trade_date", "ts_code", "value"]].copy()
    raw_market = raw_market.loc[:, ["trade_date", "ts_code"]].copy()
    for frame in (source, raw_market):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.normalize()
        frame["ts_code"] = frame["ts_code"].astype(str)
    source["value"] = pd.to_numeric(source["value"], errors="coerce")
    source = source.drop_duplicates(["trade_date", "ts_code"], keep="last")
    raw_keys = raw_market.drop_duplicates(["trade_date", "ts_code"])
    merged = source.merge(
        raw_keys.assign(_raw_present=True),
        on=["trade_date", "ts_code"],
        how="left",
    )
    finite = np.isfinite(source["value"].to_numpy(float))
    orphan = finite & merged["_raw_present"].isna().to_numpy(bool)
    orphan_rows = merged.loc[orphan]
    return {
        "source_rows": len(source),
        "source_finite_rows": int(finite.sum()),
        "raw_key_rows": len(raw_keys),
        "finite_orphan_rows": int(orphan.sum()),
        "finite_orphan_codes": int(orphan_rows["ts_code"].nunique()),
        "finite_orphan_dates": int(orphan_rows["trade_date"].nunique()),
        "orphan_start": (
            orphan_rows["trade_date"].min().date().isoformat() if len(orphan_rows) else None
        ),
        "orphan_end": (
            orphan_rows["trade_date"].max().date().isoformat() if len(orphan_rows) else None
        ),
        "decision_status": (
            "MARKET_COVERAGE_PASS" if not orphan.any() else "MARKET_COVERAGE_BLOCKED"
        ),
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


def run_audit(
    *,
    client: object,
    factor_ids: Sequence[str],
    start: date,
    end: date,
    raw_start: date,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    raw_market = client.run(
        "select trade_date,ts_code from loadTable('dfs://YesLab','stock_factor_daily') "
        f"where trade_date between date('{raw_start:%Y.%m.%d}') "
        f"and date('{end:%Y.%m.%d}')"
    )
    if not isinstance(raw_market, pd.DataFrame):
        raise RuntimeError("raw market coverage query did not return a DataFrame")
    for factor_id in factor_ids:
        definition = replay.REPLAY_DEFINITIONS[factor_id]
        source = client.run(
            f"select trade_date,ts_code,value from loadTable('dfs://YesLab','{definition.store_factor}') "
            f"where factor_name == '{definition.factor_name}' "
            f"and trade_date between date('{start:%Y.%m.%d}') "
            f"and date('{end:%Y.%m.%d}')"
        )
        if not isinstance(source, pd.DataFrame):
            raise RuntimeError(f"factor coverage query failed for {factor_id}")
        rows.append(
            {
                "factor_id": factor_id,
                "factor_name": definition.factor_name,
                **compare_factor_market_coverage(source, raw_market),
            }
        )
    return {
        "scope": "FACTOR_RAW_MARKET_COVERAGE_AUDIT_V1",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "raw_market_start": raw_start.isoformat(),
        "factor_ids": list(factor_ids),
        "rows": rows,
        "database_writes": False,
        "locked_test_read": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-id", action="append", required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--raw-start", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    unsupported = sorted(set(args.factor_id).difference(replay.REPLAY_DEFINITIONS))
    if unsupported:
        raise ValueError(f"unsupported factor IDs: {unsupported}")
    if args.start > args.end or args.raw_start > args.end:
        raise ValueError("invalid audit date range")
    if args.end >= LOCKED_TEST_START:
        raise ValueError("coverage audit cannot read 2026 locked-test data")
    return args


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    payload = run_audit(
        client=_connect(),
        factor_ids=args.factor_id,
        start=args.start,
        end=args.end,
        raw_start=args.raw_start,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


if __name__ == "__main__":
    main()
