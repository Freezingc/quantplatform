#!/usr/bin/env python3
"""Read-only: does the YesLab store's own daily valued share match the power base's positive share?

§BE proved that OUR frozen availability equals the share of the cross-section whose power base is
positive (Pearson 0.9988-1.0000 for all six sparse factors). §BD left one thing open: yeslab_375
(GP_006) is the one factor whose frozen/store row ratio is unstable across the window (std 0.1105 vs
0.0082-0.0366 for the other seven). §BB measured one day where the store held 3,740 valued codes
inside that day's universe while our frozen table held 536, and the base-positive share that day was
about 0.14 -- which cannot both be true of the same quantity.

This separates the two sides cleanly. For each of the six sparse factors and each of the 969 OOS days it
compares THREE series:

  store_share   the YesLab store's own valued share, count(value)/count(*) for that name that day
  base_share    the share of the frozen panel's cross-section whose power base is positive (§BE's series)
  frozen_share  the share of the frozen panel's cross-section that has a value

store_share is computed on the store's own (wider) code set, so it is not identical to base_share even
when the mechanism is the same; what matters is whether the two SERIES move together and sit at the
same level. If they do for five factors and not for GP_006, the discrepancy is localized to GP_006 and
the candidate explanations narrow to "our translation of pow differs from the producer's" or "the
producer's vol is not Tushare's vol". If they fail for all six, §BE's mechanism describes our replay but
not the producer's factor, which would be a much larger statement.

SAFETY: SELECT-only on dfs://YesLab; market data from the canonical Tushare cache, read in place;
§BE's daily series is re-read from its parquet rather than recomputed. No calculator is executed, no
database is written, no artifact is mutated.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
START, END = "2022.01.01", "2025.12.31"
FACTORS = (
    ("factor_data8", "GP_003", "yeslab_372", "pct_chg_up"),
    ("factor_data9", "GP_020", "yeslab_389", "pct_chg_up"),
    ("factor_data9", "GP_021", "yeslab_390", "pct_chg_up"),
    ("factor_data8", "GP_006", "yeslab_375", "vol_up_20"),
    ("factor_data8", "GP_015", "yeslab_384", "vol_up_20"),
    ("factor_data9", "GP_037", "yeslab_406", "turnover_up_20"),
    ("factor_data8", "GP_012", "yeslab_381", None),
    ("factor_data9", "GP_056", "yeslab_425", None),
)
BASE_COL = {"pct_chg_up": "share_pct_chg_up", "vol_up_20": "share_vol_up_20",
            "turnover_up_20": "share_turnover_up_20"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ART / "replay_vs_store")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    daily = pd.read_parquet(ART / "regime_mechanism" / "regime_mechanism_daily.parquet")
    daily["signal_date"] = pd.to_datetime(daily["signal_date"]).dt.date
    daily = daily.set_index("signal_date")
    print(f"§BE daily series: {len(daily)} days {daily.index.min()} .. {daily.index.max()}", flush=True)

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import new_holder, query
    holder = new_holder()

    result = {
        "question": (
            "Does the YesLab store's own daily valued share track the power base's positive share, for "
            "each of the six sparse factors -- and if not, for which one?"
        ),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "safety": "SELECT-only on dfs://YesLab; cache and §BE's parquet read in place; no calculator run",
        "window": {"start": START, "end": END, "days_in_be_series": int(len(daily))},
        "note_on_comparability": (
            "store_share is measured on the store's own code set, which is wider than the frozen panel's "
            "universe, so levels can differ; the test is whether the two series move together and sit at "
            "a similar level, and whether frozen_share equals base_share as §BE established"
        ),
        "factors": {},
    }

    for table, name, fid, base in FACTORS:
        sql = (f"select trade_date, count(*) as n_all, count(value) as n_val "
               f"from loadTable('dfs://YesLab', '{table}') "
               f"where factor_name = '{name}' and trade_date between date('{START}') and date('{END}') "
               f"group by trade_date")
        ok, frame = query(holder, sql)
        if not ok or not isinstance(frame, pd.DataFrame) or frame.empty:
            result["factors"][fid] = {"error": "query failed or empty"}
            continue
        frame["d"] = pd.to_datetime(frame["trade_date"]).dt.date
        s = frame.groupby("d")[["n_all", "n_val"]].sum()
        s["store_share"] = s["n_val"] / s["n_all"]
        s = s.reindex(daily.index)
        rec = {
            "source_name": name, "table": table, "base": base,
            "days_with_store_rows": int(s["n_all"].notna().sum()),
            "store_rows_all_median_per_day": float(s["n_all"].median()),
            "store_share": {"mean": float(s["store_share"].mean()),
                            "std": float(s["store_share"].std(ddof=1)),
                            "min": float(s["store_share"].min()),
                            "median": float(s["store_share"].median()),
                            "max": float(s["store_share"].max())},
            "frozen_share": {"mean": float(daily[f"cov_{fid}"].mean()),
                             "std": float(daily[f"cov_{fid}"].std(ddof=1)),
                             "min": float(daily[f"cov_{fid}"].min()),
                             "median": float(daily[f"cov_{fid}"].median()),
                             "max": float(daily[f"cov_{fid}"].max())},
        }
        if base:
            b = daily[BASE_COL[base]]
            rec["base_share"] = {"mean": float(b.mean()), "std": float(b.std(ddof=1)),
                                 "min": float(b.min()), "median": float(b.median()),
                                 "max": float(b.max())}
            both = s["store_share"].notna()
            rec["store_share_vs_base_share"] = {
                "pearson": float(np.corrcoef(s["store_share"][both], b[both])[0, 1]),
                "spearman": float(pd.Series(s["store_share"][both]).corr(
                    pd.Series(b[both].to_numpy()), method="spearman")),
                "mean_store_minus_base": float((s["store_share"][both] - b[both]).mean()),
                "median_absolute_gap": float((s["store_share"][both] - b[both]).abs().median()),
                "share_of_days_within_0.05": float(((s["store_share"][both] - b[both]).abs() <= 0.05).mean()),
            }
            rec["frozen_share_vs_base_share"] = {
                "pearson": float(np.corrcoef(daily[f"cov_{fid}"], b)[0, 1]),
                "mean_frozen_minus_base": float((daily[f"cov_{fid}"] - b).mean()),
                "share_of_days_within_0.05": float(((daily[f"cov_{fid}"] - b).abs() <= 0.05).mean()),
            }
            rec["store_share_vs_frozen_share"] = {
                "pearson": float(np.corrcoef(s["store_share"][both], daily[f"cov_{fid}"][both])[0, 1]),
                "mean_store_minus_frozen": float((s["store_share"][both] - daily[f"cov_{fid}"][both]).mean()),
                "share_of_days_within_0.05": float(
                    ((s["store_share"][both] - daily[f"cov_{fid}"][both]).abs() <= 0.05).mean()),
            }
            worst = (s["store_share"] - daily[f"cov_{fid}"]).abs().nlargest(8)
            rec["eight_days_where_store_and_frozen_disagree_most"] = {
                str(k): {"store_share": float(s.loc[k, "store_share"]),
                         "frozen_share": float(daily.loc[k, f"cov_{fid}"]),
                         "base_share": float(b.loc[k]),
                         "store_rows_all": int(s.loc[k, "n_all"]),
                         "store_rows_valued": int(s.loc[k, "n_val"])}
                for k in worst.index
            }
        result["factors"][fid] = rec
        line = (f"  {fid:12} {name:8} store_share med={rec['store_share']['median']:.4f} "
                f"std={rec['store_share']['std']:.4f} | frozen_share med={rec['frozen_share']['median']:.4f}")
        if base:
            line += (f" | base med={rec['base_share']['median']:.4f} "
                     f"store~base r={rec['store_share_vs_base_share']['pearson']:.4f} "
                     f"within.05={rec['store_share_vs_base_share']['share_of_days_within_0.05']:.3f}")
        print(line, flush=True)

    out = args.out / "store_valued_share_vs_base.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    for fid, v in result["factors"].items():
        if "error" in v or not v.get("base"):
            continue
        print(f"\n{fid} ({v['source_name']}, base={v['base']}) -- eight worst store-vs-frozen days:")
        for d, x in v["eight_days_where_store_and_frozen_disagree_most"].items():
            print(f"   {d} store={x['store_share']:.4f} ({x['store_rows_valued']}/{x['store_rows_all']}) "
                  f"frozen={x['frozen_share']:.4f} base={x['base_share']:.4f}")


if __name__ == "__main__":
    main()
