#!/usr/bin/env python3
"""Read-only: day by day over 2022-2025, does each frozen factor's row count track its YesLab source?

§BB sampled four days and found one factor -- yeslab_375 (GP_006) -- whose frozen row count is far below
what the YesLab store holds for the same day inside the same universe, while its three same-batch
neighbours matched exactly. Reading materialize_frozen_factor_daily.py:151-160 then showed why the
"copy vs back-fill" framing was wrong: the frozen rows are not copied from YesLab at all. They are
produced by running the producer's own calculator class against a Tushare market session
(audit_formula_replay_consistency._replay_factor), then restricted to the official universe and to
finite values. So a YesLab back-fill cannot change them, and the GP_006 gap is a replay-versus-store
disagreement rather than a lost copy.

This measures that disagreement over the whole OOS window instead of four days: per factor, per
trading day, the YesLab valued-row count against the frozen factor_daily row count, and the ratio's
distribution. A ratio that is stable and then collapses on identifiable days is a replay-side event; a
ratio that is uniformly low is a code-set breadth difference and means nothing on its own.

SAFETY: SELECT-only on dfs://YesLab and dfs://QuantResearchV2Test. No writes to any database, no
artifact mutation, no weight loading, no calculator is executed here -- this probe only counts rows.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import median

import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
ART = Path("/data/zhangyudong/qrv2_research_artifacts")
START, END = "2022.01.01", "2025.12.31"
TARGET_DB = "dfs://QuantResearchV2Test"
FACTORS = (
    ("factor_data8", "GP_003", "yeslab_372", "sparse cluster-A"),
    ("factor_data9", "GP_020", "yeslab_389", "sparse cluster-A"),
    ("factor_data9", "GP_021", "yeslab_390", "sparse cluster-A"),
    ("factor_data8", "GP_006", "yeslab_375", "sparse cluster-B, the §BB gap"),
    ("factor_data8", "GP_015", "yeslab_384", "sparse cluster-B"),
    ("factor_data9", "GP_037", "yeslab_406", "sparse cluster-B"),
    ("factor_data8", "GP_012", "yeslab_381", "CONTROL dense, factor_data8"),
    ("factor_data9", "GP_056", "yeslab_425", "CONTROL dense, factor_data9"),
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ART / "replay_vs_store")
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import new_holder, query

    holder = new_holder()
    result = {
        "question": (
            "Over the whole 2022-2025 OOS window, does each frozen factor's daily row count track the "
            "YesLab store's daily valued-row count, and if not, on which days and for which factor?"
        ),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "safety": "SELECT-only; no calculator was executed, no database was written",
        "window": {"start": args.start, "end": args.end},
        "why_the_ratio_is_not_one": (
            "the frozen table is restricted to the official universe per date while the store covers a "
            "wider code set, so the ratio sits near that universe-in share (~0.36 for GP_012); only "
            "DEVIATIONS from a factor's own typical ratio are informative"
        ),
        "factors": {},
    }

    for table, name, fid, role in FACTORS:
        sql_up = (f"select trade_date, count(*) as n_all, count(value) as n_val "
                  f"from loadTable('dfs://YesLab', '{table}') "
                  f"where factor_name = '{name}' and trade_date between date('{args.start}') and date('{args.end}') "
                  f"group by trade_date")
        ok, up = query(holder, sql_up)
        sql_fz = (f"select signal_date, data_version, count(*) as n, count(value) as n_val "
                  f"from loadTable('{TARGET_DB}', 'factor_daily') "
                  f"where factor_id = '{fid}' and signal_date between date('{args.start}') and date('{args.end}') "
                  f"group by signal_date, data_version")
        ok2, fz = query(holder, sql_fz)
        if not ok or not isinstance(up, pd.DataFrame) or not ok2 or not isinstance(fz, pd.DataFrame):
            result["factors"][fid] = {"error": "query failed"}
            continue
        up["d"] = pd.to_datetime(up["trade_date"]).dt.date
        up = up.groupby("d")[["n_all", "n_val"]].sum()
        fz["d"] = pd.to_datetime(fz["signal_date"]).dt.date
        versions_per_day = fz.groupby("d")["data_version"].nunique()
        # one row per day: if a day carries several data_versions, take the largest, and record that
        fz_max = fz.sort_values("n_val").groupby("d").tail(1).set_index("d")
        joined = up.join(fz_max[["n", "n_val"]].rename(columns={"n": "fz_n", "n_val": "fz_n_val"}),
                         how="outer").fillna(0)
        ratio = joined["fz_n_val"] / joined["n_val"].where(joined["n_val"] > 0)
        low = ratio.dropna().nsmallest(12)
        result["factors"][fid] = {
            "source_name": name, "table": table, "role": role,
            "days_in_store": int((joined["n_val"] > 0).sum()),
            "days_in_frozen": int((joined["fz_n_val"] > 0).sum()),
            "days_in_either": int(len(joined)),
            "store_valued_rows_total": int(joined["n_val"].sum()),
            "frozen_rows_total": int(joined["fz_n_val"].sum()),
            "ratio_frozen_over_store": {
                "n_days": int(ratio.notna().sum()),
                "min": float(ratio.min()) if ratio.notna().any() else None,
                "p01": float(ratio.quantile(0.01)) if ratio.notna().any() else None,
                "p05": float(ratio.quantile(0.05)) if ratio.notna().any() else None,
                "median": float(ratio.median()) if ratio.notna().any() else None,
                "p95": float(ratio.quantile(0.95)) if ratio.notna().any() else None,
                "max": float(ratio.max()) if ratio.notna().any() else None,
                "std": float(ratio.std(ddof=1)) if ratio.notna().sum() > 1 else None,
            },
            "twelve_lowest_ratio_days": {str(k): {"ratio": float(v),
                                                  "store_valued": int(joined.loc[k, "n_val"]),
                                                  "frozen": int(joined.loc[k, "fz_n_val"])}
                                          for k, v in low.items()},
            "days_with_several_data_versions": int((versions_per_day > 1).sum()),
            "days_where_store_has_values_but_frozen_has_none": int(
                ((joined["n_val"] > 0) & (joined["fz_n_val"] == 0)).sum()
            ),
            "days_where_frozen_has_rows_but_store_has_none": int(
                ((joined["fz_n_val"] > 0) & (joined["n_val"] == 0)).sum()
            ),
        }
        r = result["factors"][fid]["ratio_frozen_over_store"]
        print(f"  {fid:12} {name:8} days={result['factors'][fid]['days_in_either']:4} "
              f"ratio med={r['median']:.4f} p05={r['p05']:.4f} min={r['min']:.4f} max={r['max']:.4f} "
              f"store>frozen-none={result['factors'][fid]['days_where_store_has_values_but_frozen_has_none']}",
              flush=True)

    out = args.out / "replay_vs_store_daily.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    for fid, v in result["factors"].items():
        if "error" in v:
            continue
        print(f"\n{fid} ({v['source_name']}, {v['role']}) -- twelve lowest ratio days:")
        for d, x in v["twelve_lowest_ratio_days"].items():
            print(f"   {d}  ratio={x['ratio']:.4f}  store_valued={x['store_valued']:6}  frozen={x['frozen']:6}")


if __name__ == "__main__":
    main()
