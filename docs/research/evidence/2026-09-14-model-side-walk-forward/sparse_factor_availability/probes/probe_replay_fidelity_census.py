#!/usr/bin/env python3
"""Read-only census: for every frozen-v2 factor, does the frozen column reproduce the catalogued source?

The frozen chain does not copy factor values from dfs://YesLab. materialize_frozen_factor_daily.py
imports the producer's own calculator class and replays it against a Tushare market session, then keeps
the rows inside that day's official universe with a finite value. So every frozen column is a
REPRODUCTION of a catalogued factor, and reproduction is a measurable property rather than an
assumption.

§Q measured this for part of the dense nine and found yeslab_484 at Spearman 0.994-0.9999 against
yeslab_471 at 0.82-0.88. §BF then found yeslab_375 at 0.927 (min 0.798) while its three siblings
reproduced at 0.996-1.000. This runs one uniform instrument over every frozen-v2 factor that has a
catalogue entry, on the same days, so the owner gets a single table rather than scattered numbers.

Two of the fifteen are reported separately rather than censused: yeslab_443_reg30_max20_v1 has no
source_catalog entry (it is an approved repair variant), and yeslab_206 and yeslab_443 take special
branches in the materializer rather than REPLAY_DEFINITIONS.

SAFETY: SELECT-only on dfs://YesLab and dfs://QuantResearchV2Test. No calculator is executed, no
database is written, no artifact is mutated, no weight is loaded.
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
# source_catalog.yaml, read rather than retyped from memory
CATALOG = (
    ("factor_data5", "60日成交量去均值相对偏离", "yeslab_206", "dense; materializer special branch MEDIAN_206"),
    ("factor_data8", "行业内收益标准化绝对值", "yeslab_357", "dense; PIT replay with the industry provider"),
    ("factor_data8", "GP_003", "yeslab_372", "sparse cluster A"),
    ("factor_data8", "GP_006", "yeslab_375", "sparse cluster B"),
    ("factor_data8", "GP_012", "yeslab_381", "dense"),
    ("factor_data8", "GP_015", "yeslab_384", "sparse cluster B"),
    ("factor_data9", "GP_020", "yeslab_389", "sparse cluster A"),
    ("factor_data9", "GP_021", "yeslab_390", "sparse cluster A"),
    ("factor_data9", "GP_037", "yeslab_406", "sparse cluster B"),
    ("factor_data9", "GP_056", "yeslab_425", "dense"),
    ("factor_data10", "GP_095", "yeslab_464", "dense; one of §AR's two FAILs"),
    ("factor_data10", "AttentionSpillover", "yeslab_471", "dense; PIT replay; one of §AR's two FAILs"),
    ("factor_data10", "羊群效应_CSAD波动率21", "yeslab_484", "dense; §Q's highest-fidelity factor"),
    ("factor_data3", "修正日间反转", "yeslab_96", "dense"),
)
DAYS = ("2022-03-15", "2023-06-14", "2024-02-28", "2024-09-30", "2024-10-09", "2024-10-11",
        "2024-11-05", "2025-06-16")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ART / "replay_vs_store")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import new_holder, query
    holder = new_holder()

    result = {
        "question": (
            "For each frozen-v2 factor, does the frozen factor_daily column reproduce the catalogued "
            "YesLab source, on the rows both sides have?"
        ),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "safety": "SELECT-only; no calculator executed; nothing written to any database or artifact",
        "days": list(DAYS),
        "not_censused": {
            "yeslab_443_reg30_max20_v1": "no source_catalog entry; an approved repair variant",
            "yeslab_363": "zero rows in factor_daily (§AM/§BC), so there is nothing to compare",
        },
        "factors": {},
    }
    for table, name, fid, role in CATALOG:
        if "'" in name:
            raise ValueError(f"factor name would need escaping: {name!r}")
        per_day, rhos = {}, []
        for ds in DAYS:
            dl = ds.replace("-", ".")
            ok_s, st = query(holder,
                f"select ts_code, value from loadTable('dfs://YesLab', '{table}') "
                f"where factor_name = '{name}' and trade_date = date('{dl}')")
            ok_f, fz = query(holder,
                f"select ts_code, value from loadTable('{TARGET_DB}', 'factor_daily') "
                f"where factor_id = '{fid}' and signal_date = date('{dl}')")
            if not ok_s or not ok_f or not isinstance(st, pd.DataFrame) or not isinstance(fz, pd.DataFrame):
                per_day[ds] = {"error": "query failed"}
                continue
            for frame in (st, fz):
                frame["ts_code"] = frame["ts_code"].astype(str)
                frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
            st = st.dropna(subset=["value"])
            both = fz.merge(st, on="ts_code", how="inner", suffixes=("_frozen", "_store")) \
                     .dropna(subset=["value_frozen", "value_store"])
            entry = {"store_valued_rows": int(len(st)), "frozen_rows": int(len(fz)),
                     "rows_in_both": int(len(both))}
            if len(both) >= 30:
                rho = stats.spearmanr(both["value_frozen"], both["value_store"])
                rel = ((both["value_frozen"] - both["value_store"]).abs()
                       / both["value_store"].abs().replace(0, np.nan))
                entry.update({
                    "spearman": float(rho.statistic),
                    "pearson": float(stats.pearsonr(both["value_frozen"], both["value_store"])[0]),
                    "median_relative_difference": float(rel.median()),
                    "sign_agreement": float((np.sign(both["value_frozen"])
                                             == np.sign(both["value_store"])).mean()),
                })
                rhos.append(entry["spearman"])
            per_day[ds] = entry
        result["factors"][fid] = {
            "source_name": name, "table": table, "role": role, "per_day": per_day,
            "days_measured": len(rhos),
            "spearman_median": float(np.median(rhos)) if rhos else None,
            "spearman_min": min(rhos) if rhos else None,
            "spearman_max": max(rhos) if rhos else None,
        }
        m = result["factors"][fid]
        print(f"  {fid:30} {name[:22]:24} days={m['days_measured']} "
              f"spearman med={m['spearman_median']} min={m['spearman_min']}", flush=True)

    ranked = sorted(
        ((f, v["spearman_median"]) for f, v in result["factors"].items()
         if v["spearman_median"] is not None),
        key=lambda kv: kv[1],
    )
    result["ranking_by_median_spearman"] = [
        {"factor_id": f, "source_name": result["factors"][f]["source_name"],
         "spearman_median": v, "spearman_min": result["factors"][f]["spearman_min"]}
        for f, v in ranked
    ]
    result["summary"] = {
        "factors_censused": len(ranked),
        "reproducing_at_or_above_0.999": [f for f, v in ranked if v >= 0.999],
        "between_0.95_and_0.999": [f for f, v in ranked if 0.95 <= v < 0.999],
        "below_0.95": [f for f, v in ranked if v < 0.95],
        "reading": (
            "a frozen column is a reproduction, not a copy: it is the producer's calculator class run "
            "against a Tushare market session and then restricted to the official universe. Rank "
            "correlation against the catalogued source is therefore the right fidelity measure, and a "
            "factor sitting well below the others is a factor whose frozen column is a proxy rather "
            "than the catalogued quantity."
        ),
    }
    out = args.out / "replay_fidelity_census.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
