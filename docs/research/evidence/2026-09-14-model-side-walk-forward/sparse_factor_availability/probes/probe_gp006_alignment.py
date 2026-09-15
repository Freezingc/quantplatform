#!/usr/bin/env python3
"""Read-only: is GP_006's store series a date-shifted copy of the base-positive series?

`probe_store_valued_share_vs_base.py` found that for five of the six sparse factors the YesLab store's
own daily valued share tracks the share of the cross-section whose power base is positive at Pearson
0.981-0.989 -- so the producer's `pow` also yields NaN on a negative base, and the day-level regime is a
property of the factors themselves rather than of our translation. GP_006 (yeslab_375) is the single
exception at 0.3895, and it disagrees with our replay in BOTH directions on different days (on
2024-10-09 the store valued 31% of its rows while our frozen panel covered 94% of the universe).

A date-convention mismatch would produce exactly that: high volatility, low same-day correlation, and
disagreement in both directions. This scans leads and lags of the base series against the store series
for GP_006, with GP_015 and GP_037 as controls that should peak at lag 0, and reports the best offset.

SAFETY: SELECT-only on dfs://YesLab; §BE's daily parquet read in place; no calculator executed; no
database writes; no artifact mutation.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
START, END = "2022.01.01", "2025.12.31"
TABLES = {
    "factor_data8": ["GP_003", "GP_006", "GP_012", "GP_015"],
    "factor_data9": ["GP_020", "GP_021", "GP_037", "GP_056"],
}
BASE_OF = {"GP_003": "share_pct_chg_up", "GP_020": "share_pct_chg_up", "GP_021": "share_pct_chg_up",
           "GP_006": "share_vol_up_20", "GP_015": "share_vol_up_20", "GP_037": "share_turnover_up_20"}
MAX_SHIFT = 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ART / "replay_vs_store")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    daily = pd.read_parquet(ART / "regime_mechanism" / "regime_mechanism_daily.parquet")
    daily["signal_date"] = pd.to_datetime(daily["signal_date"]).dt.date
    daily = daily.set_index("signal_date").sort_index()

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import new_holder, query
    holder = new_holder()

    store: dict[str, pd.Series] = {}
    for table, names in TABLES.items():
        quoted = ", ".join(f"'{n}'" for n in names)
        sql = (f"select trade_date, factor_name, count(*) as n_all, count(value) as n_val "
               f"from loadTable('dfs://YesLab', '{table}') "
               f"where factor_name in ({quoted}) and trade_date between date('{START}') and date('{END}') "
               f"group by trade_date, factor_name")
        ok, frame = query(holder, sql)
        if not ok or not isinstance(frame, pd.DataFrame) or frame.empty:
            raise SystemExit(f"query failed for {table}")
        frame["d"] = pd.to_datetime(frame["trade_date"]).dt.date
        for name, g in frame.groupby("factor_name"):
            s = g.groupby("d")[["n_all", "n_val"]].sum()
            store[str(name)] = (s["n_val"] / s["n_all"]).reindex(daily.index)
        print(f"  {table}: {sorted(frame['factor_name'].unique())}", flush=True)

    result = {
        "question": (
            "Is GP_006's store availability series a date-shifted copy of the base-positive series, or "
            "genuinely driven by something else?"
        ),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "safety": "SELECT-only; no calculator executed; nothing written to any database or artifact",
        "method": (
            "for each factor, correlate the store's daily valued share with the base-positive share "
            f"shifted by k trading days for k in -{MAX_SHIFT}..{MAX_SHIFT}; a date-convention mismatch "
            "shows up as a sharp peak away from k=0"
        ),
        "factors": {},
    }
    for name, series in store.items():
        base_col = BASE_OF.get(name)
        entry = {"store_share_median": float(series.median()),
                 "store_share_std": float(series.std(ddof=1))}
        if base_col:
            b = daily[base_col]
            scan = {}
            for k in range(-MAX_SHIFT, MAX_SHIFT + 1):
                shifted = b.shift(k)          # positive k compares the store with an EARLIER base day
                ok = series.notna() & shifted.notna()
                scan[k] = float(np.corrcoef(series[ok], shifted[ok])[0, 1]) if ok.sum() > 10 else None
            valid = {k: v for k, v in scan.items() if v is not None}
            best_k = max(valid, key=lambda k: valid[k])
            entry["base"] = base_col
            entry["lag_scan"] = {str(k): v for k, v in scan.items()}
            entry["correlation_at_lag_0"] = valid.get(0)
            entry["best_lag"] = best_k
            entry["correlation_at_best_lag"] = valid[best_k]
            entry["improvement_over_lag_0"] = valid[best_k] - (valid.get(0) or 0.0)
            entry["verdict"] = (
                "DATE_SHIFT_EXPLAINS_IT" if best_k != 0 and valid[best_k] > 0.90
                else "NO_SHIFT_FOUND" if best_k == 0
                else "SHIFT_HELPS_BUT_DOES_NOT_EXPLAIN"
            )
        result["factors"][name] = entry
        if base_col:
            print(f"  {name:8} r@0={entry['correlation_at_lag_0']:.4f} best k={entry['best_lag']:+d} "
                  f"r={entry['correlation_at_best_lag']:.4f} -> {entry['verdict']}", flush=True)
        else:
            print(f"  {name:8} (control, no power) store_share med={entry['store_share_median']:.4f} "
                  f"std={entry['store_share_std']:.4f}", flush=True)

    out = args.out / "gp006_alignment_scan.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
