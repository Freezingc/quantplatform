"""Redundancy pre-check for the alpha443 reg30_max20 repair variant.

Computes canonical reg30_max5 and repair reg30_max20 factor values over the
requested window and reports the daily cross-sectional Spearman correlation
between them (median/mean). If the variant is near-duplicate of canonical
(>= ~0.95), the canonical candidate's existing nine-edge audit transfers;
otherwise a full nine-edge audit of the variant is required. Read-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "research_v2" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "research_v2" / "src"))

from audit_formula_replay_tushare import (  # noqa: E402
    _connect,
    _query_open_dates,
    fetch_full_tushare_shsz_market,
)
from repair_alpha443_window_scan import compute_gp074_variant  # noqa: E402

CACHE = REPO_ROOT / "research_v2" / "artifacts" / "tushare_cache"


def daily_spearman(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    common_cols = a.columns.intersection(b.columns)
    ar, br = a.loc[:, common_cols].rank(axis=1), b.loc[:, common_cols].rank(axis=1)
    idx = ar.index.intersection(br.index)
    out = {}
    for dt in idx:
        x, y = ar.loc[dt], br.loc[dt]
        both = x.notna() & y.notna()
        if int(both.sum()) < 100:
            continue
        xv, yv = x[both].to_numpy(float), y[both].to_numpy(float)
        if np.std(xv) == 0 or np.std(yv) == 0:
            continue
        out[dt] = float(np.corrcoef(xv, yv)[0, 1])
    return pd.Series(out).sort_index()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    client = _connect()
    market_start = args.start - timedelta(days=150)
    market_end = args.end + timedelta(days=5)
    open_dates = _query_open_dates(client, market_start, market_end)
    market = fetch_full_tushare_shsz_market(
        open_dates, token=os.getenv("QRV2_TUSHARE_TOKEN", ""), cache_dir=CACHE, include_daily_basic=False
    )
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    market["ts_code"] = market["ts_code"].astype(str)
    amount = market.pivot_table(index="trade_date", columns="ts_code", values="amount", aggfunc="last").sort_index()
    pct_chg = market.pivot_table(index="trade_date", columns="ts_code", values="pct_chg", aggfunc="last").sort_index()
    wide = {"amount": amount, "pct_chg": pct_chg}

    canonical = compute_gp074_variant(wide, 30, 5)
    repair = compute_gp074_variant(wide, 30, 20)
    window_idx = [d for d in canonical.index if pd.Timestamp(args.start) <= d <= pd.Timestamp(args.end)]
    canonical = canonical.loc[canonical.index.intersection(window_idx)]
    repair = repair.loc[repair.index.intersection(window_idx)]

    corr = daily_spearman(canonical, repair)
    payload = {
        "scope": "ALPHA443_REPAIR_REDUNDANCY_PRECHECK_V1",
        "window": f"{args.start}/{args.end}",
        "pairs": {
            "canonical_reg30_max5_vs_repair_reg30_max20": {
                "days": int(len(corr)),
                "median_daily_spearman": round(float(corr.median()), 4) if len(corr) else None,
                "mean_daily_spearman": round(float(corr.mean()), 4) if len(corr) else None,
                "min_daily_spearman": round(float(corr.min()), 4) if len(corr) else None,
            }
        },
        "transfer_rule": "median >= 0.95 -> canonical nine-edge audit transfers; otherwise full nine-edge audit required",
        "database_writes": False,
        "locked_test_read": False,
    }
    (args.output / "alpha443_repair_redundancy_precheck.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["pairs"], ensure_ascii=False))
    print(f"wrote {args.output / 'alpha443_repair_redundancy_precheck.json'}")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
