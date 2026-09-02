"""Replicate forward_open_5d:v1 labels for pre-2020 research windows (read-only).

Replicates the exact build_forward_open_labels spec (entry = adjusted open at
T+1, exit = adjusted open at T+6 on the trading calendar) from the backfilled
market_daily_clean and universe_daily tables. Validates the replication against
the official label_daily for 2020 first (must match at machine precision);
only if validation passes does it emit research labels for 2019.

Read-only research artifact; no database writes, no locked-test reads.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import dolphindb as ddb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG = REPO_ROOT / "research_v2" / "artifacts" / "diagnostics"


def _connect():
    import os
    session = ddb.session()
    session.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    return session


def load_market_prices(client, start: date, end: date) -> pd.DataFrame:
    frame = client.run(
        f'''
        db = database("dfs://QuantResearchV2Test")
        t = loadTable(db, "market_daily_clean")
        select trade_date, ts_code, adjusted_open from t
        where trade_date >= {start:%Y.%m.%d} and trade_date <= {end:%Y.%m.%d}
        '''
    )
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["ts_code"] = frame["ts_code"].astype(str)
    return frame


def load_universe_members(client, start: date, end: date) -> pd.DataFrame:
    frame = client.run(
        f'''
        db = database("dfs://QuantResearchV2Test")
        t = loadTable(db, "universe_daily")
        select signal_date, ts_code from t
        where signal_date >= {start:%Y.%m.%d} and signal_date <= {end:%Y.%m.%d} and is_member = 1
        '''
    )
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame["ts_code"] = frame["ts_code"].astype(str)
    return frame.drop_duplicates(["signal_date", "ts_code"])


def load_full_market_members(client, start: date, end: date) -> pd.DataFrame:
    """Research-only membership: every SH/SZ code present in market_daily_clean."""
    frame = client.run(
        f'''
        db = database("dfs://QuantResearchV2Test")
        t = loadTable(db, "market_daily_clean")
        select trade_date as signal_date, ts_code from t
        where trade_date >= {start:%Y.%m.%d} and trade_date <= {end:%Y.%m.%d}
          and (ts_code like "%.SH" or ts_code like "%.SZ")
        '''
    )
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame["ts_code"] = frame["ts_code"].astype(str)
    return frame.drop_duplicates(["signal_date", "ts_code"])


def load_official_labels(client, start: date, end: date) -> pd.DataFrame:
    frame = client.run(
        f'''
        db = database("dfs://QuantResearchV2Test")
        t = loadTable(db, "label_daily")
        select signal_date, ts_code, raw_return, label_valid from t
        where signal_date >= {start:%Y.%m.%d} and signal_date <= {end:%Y.%m.%d}
          and label_id = "forward_open_5d" and label_version = 1
        '''
    )
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame["ts_code"] = frame["ts_code"].astype(str)
    return frame


def replicate_labels(members: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Vectorized replication of build_forward_open_labels."""
    prices = market.pivot_table(index="trade_date", columns="ts_code", values="adjusted_open", aggfunc="last").sort_index()
    trading_dates = prices.index
    pos = {ts: i for i, ts in enumerate(trading_dates)}
    members = members.sort_values(["signal_date", "ts_code"]).copy()
    sig_dates = pd.Series(pd.DatetimeIndex(members["signal_date"]), index=members.index)
    # map signal dates to positions (signals must be trading dates)
    sig_pos = sig_dates.map(pos)
    entry_pos = sig_pos + 1
    exit_pos = sig_pos + 6
    max_pos = len(trading_dates) - 1
    members["entry_date"] = trading_dates[entry_pos.clip(upper=max_pos)].where(sig_pos.notna() & (entry_pos <= max_pos))
    members["exit_date"] = trading_dates[exit_pos.clip(upper=max_pos)].where(sig_pos.notna() & (exit_pos <= max_pos))
    codes = prices.columns
    code_pos = {c: i for i, c in enumerate(codes)}
    code_idx = members["ts_code"].map(code_pos).to_numpy()
    mat = prices.to_numpy()
    def take(pos_arr):
        out = np.full(len(pos_arr), np.nan)
        ok = ~np.isnan(pos_arr) & ~np.isnan(code_idx)
        pi = np.where(ok, pos_arr.astype(int), 0)
        ci = np.where(ok, code_idx, 0).astype(int)
        out[ok] = mat[pi[ok], ci[ok]]
        return out
    entry_price = take(entry_pos.to_numpy())
    exit_price = take(exit_pos.to_numpy())
    valid_entry = np.isfinite(entry_price) & (entry_price > 0) & ~np.isnan(entry_pos.to_numpy())
    valid_exit = np.isfinite(exit_price) & (exit_price > 0) & ~np.isnan(exit_pos.to_numpy())
    raw = np.full(len(members), np.nan)
    ok = valid_entry & valid_exit
    raw[ok] = exit_price[ok] / entry_price[ok] - 1.0
    members["raw_return"] = raw
    members["label_valid"] = ok
    return members[["signal_date", "ts_code", "raw_return", "label_valid"]].reset_index(drop=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-start", type=date.fromisoformat, default=date(2020, 1, 2))
    parser.add_argument("--validate-end", type=date.fromisoformat, default=date(2020, 2, 28))
    parser.add_argument("--research-start", type=date.fromisoformat, default=date(2019, 1, 2))
    parser.add_argument("--research-end", type=date.fromisoformat, default=date(2019, 12, 31))
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--members", choices=["universe", "full_market"], default="universe")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    client = _connect()

    # --- validation against official 2020 labels ---
    market_val = load_market_prices(client, args.validate_start - pd.Timedelta(days=20), args.validate_end + pd.Timedelta(days=20))
    members_val = load_universe_members(client, args.validate_start, args.validate_end)
    replicated = replicate_labels(members_val, market_val)
    official = load_official_labels(client, args.validate_start, args.validate_end)
    merged = replicated.merge(official, on=["signal_date", "ts_code"], how="inner", suffixes=("_rep", "_off"))
    both_valid = merged["label_valid_rep"] & merged["label_valid_off"]
    n_both = int(both_valid.sum())
    diff = (merged.loc[both_valid, "raw_return_rep"] - merged.loc[both_valid, "raw_return_off"]).abs()
    max_diff = float(diff.max()) if n_both else float("nan")
    valid_match = float((merged["label_valid_rep"] == merged["label_valid_off"]).mean())
    validation = {
        "window": f"{args.validate_start}/{args.validate_end}",
        "merged_rows": int(len(merged)),
        "both_valid_rows": n_both,
        "max_abs_return_diff": max_diff,
        "label_valid_agreement": valid_match,
        "passed": bool(n_both > 0 and max_diff <= args.tolerance and valid_match >= 0.9999),
    }
    print("validation:", json.dumps(validation))
    if not validation["passed"]:
        (args.output / "validation_result.json").write_text(json.dumps(validation, indent=2) + "\n")
        print("VALIDATION FAILED - not emitting research labels")
        return 1

    # --- research labels for 2019 ---
    market_res = load_market_prices(client, args.research_start - pd.Timedelta(days=10), args.research_end + pd.Timedelta(days=30))
    if args.members == "full_market":
        members_res = load_full_market_members(client, args.research_start, args.research_end)
    else:
        members_res = load_universe_members(client, args.research_start, args.research_end)
    research_labels = replicate_labels(members_res, market_res)
    research_labels.to_parquet(args.output / "research_labels_2019.parquet", index=False)
    summary = {
        "validation": validation,
        "research_window": f"{args.research_start}/{args.research_end}",
        "research_label_rows": int(len(research_labels)),
        "research_valid_rows": int(research_labels["label_valid"].sum()),
        "research_signal_dates": int(research_labels["signal_date"].nunique()),
        "tolerance": args.tolerance,
        "note": "Research-only labels replicated from market_daily_clean for historical-window RankIC evidence; not published to the V2 label pipeline.",
        "database_writes": False,
        "locked_test_read": False,
    }
    (args.output / "validation_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("research labels:", json.dumps({k: summary[k] for k in ("research_label_rows", "research_valid_rows", "research_signal_dates")}))
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
