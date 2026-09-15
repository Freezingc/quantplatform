#!/usr/bin/env python3
"""Read-only: the full-window replay-fidelity distribution for the factors §AR judged redundant.

§BF censused 14 frozen factors on 8 days and found yeslab_471 worst (median Spearman 0.8698, one day at
0.4792, and only 70-90% sign agreement) and yeslab_464 fourth worst (0.9529, one day at 0.6357). Those
are exactly the two factors §AR/§AV judged redundant inside the baseline's own dense nine, so how
representative the bad days are decides how much weight those verdicts can carry. Eight days cannot say.

This pulls both sides for the whole 969-day OOS window in one query each and computes the daily Spearman
locally, so the output is a distribution rather than a sample: how often fidelity is below 0.95, below
0.90, below 0.80, what the worst days are, and whether the bad days cluster in time or by fold. Sign
agreement is computed per day too, because §BF found 471 disagreeing in sign on 10-30% of rows and a
median is not enough to characterise that.

yeslab_357 and yeslab_381 ride along: 357 shares 471's PIT industry provider and its missingness chain,
and 381 is the factor that absorbed 398, 405 and 464 in the redundancy screens.

SAFETY: SELECT-only on dfs://YesLab and dfs://QuantResearchV2Test. No calculator executed, no database
written, no artifact mutated, no weight loaded.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
TARGET_DB = "dfs://QuantResearchV2Test"
START, END = "2022.01.01", "2025.12.31"
FACTORS = (
    ("factor_data10", "AttentionSpillover", "yeslab_471", "§AR FAIL, §BF worst fidelity"),
    ("factor_data10", "GP_095", "yeslab_464", "§AR FAIL, §BF 4th worst"),
    ("factor_data8", "行业内收益标准化绝对值", "yeslab_357", "shares 471's PIT provider and missingness chain"),
    ("factor_data8", "GP_012", "yeslab_381", "the absorber in every redundancy screen"),
)
FOLD_OF_YEAR = {2022: "F1", 2023: "F2", 2024: "F3", 2025: "F4"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ART / "fidelity_full")
    ap.add_argument("--min-rows", type=int, default=100)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import new_holder, query
    holder = new_holder()

    result = {
        "question": (
            "Over the whole 969-day OOS window, how often is each frozen column a faithful rank "
            "reproduction of its catalogued source, and do the bad days cluster?"
        ),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "safety": "SELECT-only; no calculator executed; nothing written",
        "window": {"start": START, "end": END},
        "minimum_rows_per_day_to_score": args.min_rows,
        "factors": {},
    }

    for table, name, fid, role in FACTORS:
        ok_s, st = query(holder,
            f"select trade_date, ts_code, value from loadTable('dfs://YesLab', '{table}') "
            f"where factor_name = '{name}' and trade_date between date('{START}') and date('{END}')")
        if not ok_s or not isinstance(st, pd.DataFrame):
            result["factors"][fid] = {"error": "store query failed"}
            continue
        ok_f, fz = query(holder,
            f"select signal_date, ts_code, value from loadTable('{TARGET_DB}', 'factor_daily') "
            f"where factor_id = '{fid}' and signal_date between date('{START}') and date('{END}')")
        if not ok_f or not isinstance(fz, pd.DataFrame):
            result["factors"][fid] = {"error": "frozen query failed"}
            continue
        print(f"  {fid}: store {len(st)} rows, frozen {len(fz)} rows", flush=True)
        for frame, dcol in ((st, "trade_date"), (fz, "signal_date")):
            frame["d"] = pd.to_datetime(frame[dcol]).dt.date
            frame["ts_code"] = frame["ts_code"].astype(str)
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        # the store can hold more than one row per key (§BF.4); keep one per key deterministically and
        # record how many keys were ambiguous so the number is not silently chosen
        dup = st.dropna(subset=["value"]).duplicated(["d", "ts_code"], keep=False)
        st_u = st.dropna(subset=["value"]).drop_duplicates(["d", "ts_code"], keep="first")
        fz_u = fz.dropna(subset=["value"]).drop_duplicates(["d", "ts_code"], keep="first")
        merged = fz_u.merge(st_u, on=["d", "ts_code"], how="inner", suffixes=("_fz", "_st"))
        rows = []
        for d, g in merged.groupby("d"):
            if len(g) < args.min_rows:
                continue
            rho = stats.spearmanr(g["value_fz"], g["value_st"]).statistic
            rows.append({
                "d": d, "n": len(g),
                "spearman": float(rho) if np.isfinite(rho) else None,
                "sign_agreement": float((np.sign(g["value_fz"]) == np.sign(g["value_st"])).mean()),
                "median_rel_diff": float(((g["value_fz"] - g["value_st"]).abs()
                                          / g["value_st"].abs().replace(0, np.nan)).median()),
            })
        daily = pd.DataFrame(rows).dropna(subset=["spearman"]).sort_values("d").reset_index(drop=True)
        sp = daily["spearman"]
        by_year = {str(y): {"days": int((daily["d"].map(lambda x: x.year) == y).sum()),
                            "spearman_median": float(sp[daily["d"].map(lambda x: x.year) == y].median()),
                            "spearman_min": float(sp[daily["d"].map(lambda x: x.year) == y].min())}
                   for y in sorted({x.year for x in daily["d"]})}
        worst = daily.nsmallest(10, "spearman")
        result["factors"][fid] = {
            "source_name": name, "table": table, "role": role,
            "store_rows": int(len(st)), "frozen_rows": int(len(fz)),
            "store_rows_on_an_ambiguous_key": int(dup.sum()),
            "store_distinct_keys": int(len(st_u)), "frozen_distinct_keys": int(len(fz_u)),
            "keys_in_both": int(len(merged)),
            "days_scored": int(len(daily)),
            "spearman": {"median": float(sp.median()), "mean": float(sp.mean()),
                         "p01": float(sp.quantile(0.01)), "p05": float(sp.quantile(0.05)),
                         "p25": float(sp.quantile(0.25)), "min": float(sp.min()),
                         "max": float(sp.max()), "std": float(sp.std(ddof=1))},
            "share_of_days_below": {thr: float((sp < thr).mean()) for thr in (0.99, 0.95, 0.90, 0.80, 0.70)},
            "days_below": {str(thr): int((sp < thr).sum()) for thr in (0.99, 0.95, 0.90, 0.80, 0.70)},
            "sign_agreement": {"median": float(daily["sign_agreement"].median()),
                               "min": float(daily["sign_agreement"].min()),
                               "share_of_days_below_0.95": float((daily["sign_agreement"] < 0.95).mean())},
            "median_relative_difference": {"median": float(daily["median_rel_diff"].median()),
                                           "max": float(daily["median_rel_diff"].max())},
            "by_year": by_year,
            "ten_worst_days": [{"date": str(r["d"]), "spearman": r["spearman"], "n": r["n"],
                                "sign_agreement": r["sign_agreement"],
                                "fold": FOLD_OF_YEAR.get(r["d"].year)}
                               for r in worst.to_dict("records")],
        }
        s = result["factors"][fid]["spearman"]
        print(f"  {fid:12} days={result['factors'][fid]['days_scored']:4} "
              f"med={s['median']:.4f} p05={s['p05']:.4f} min={s['min']:.4f} "
              f"days<0.95={result['factors'][fid]['days_below']['0.95']} "
              f"days<0.90={result['factors'][fid]['days_below']['0.9']} "
              f"sign_min={result['factors'][fid]['sign_agreement']['min']:.3f}", flush=True)

    out = args.out / "fidelity_full_window.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    for fid, v in result["factors"].items():
        if "error" in v:
            continue
        print(f"\n{fid} ({v['source_name']}) -- {v['role']}")
        print(f"   days scored {v['days_scored']}, ambiguous store rows {v['store_rows_on_an_ambiguous_key']}")
        print(f"   spearman {json.dumps(v['spearman'])}")
        print(f"   days below {v['days_below']}  shares {v['share_of_days_below']}")
        print(f"   sign agreement {json.dumps(v['sign_agreement'])}")
        print(f"   by year {json.dumps(v['by_year'])}")
        print("   ten worst:", ", ".join(f"{w['date']}={w['spearman']:.3f}" for w in v["ten_worst_days"]))


if __name__ == "__main__":
    main()
