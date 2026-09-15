#!/usr/bin/env python3
"""Read-only: two loose ends, one query each.

(a) §BG.5 inferred from materialize_frozen_factor_daily.py:56 -- which reads factor_set_frozen_v1.json --
    that yeslab_398 and yeslab_405, removed from v2 for redundancy against yeslab_381, should still have
    rows in factor_daily because they are in v1's 17. That was an inference, not a measurement. This
    measures it, and measures yeslab_363 in the same query as the negative control that should be zero.

(b) The handoff list carries an unexplained item: three 2022 rows where all nine dense factors are
    missing (2022-03-30, 2022-03-31, 2022-05-31). Those dates matter because the nine dense factors'
    joint completeness is exactly what decides the 0.90 coverage floor and therefore G6. This locates
    every such row in all four frozen test panels, says which factors are missing, and says whether the
    day was one of the 24 the baseline side drops.

SAFETY: (a) is SELECT-only on dfs://QuantResearchV2Test. (b) reads the frozen parquet panels in place.
No calculator is executed, nothing is written to any database or artifact directory, no weight is loaded.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
TARGET_DB = "dfs://QuantResearchV2Test"
FOLD_DATASETS = {
    "F1": ("226a8dfc497c2de6b07516711f058f26a9270675916c9e67dbf7f087a07d2538", 2022),
    "F2": ("eb9bba1b5a0bd66d0ad6c82536b295de1fc079aec125cb27bb68491bdb21a234", 2023),
    "F3": ("9766e69f342511c5bda30188df15a7b9b17d936bdc26b9b60ba244e8e7e0c9e7", 2024),
    "F4": ("6f9478d7645c8c745a73f00c03b6122e17d0c06c05cc159ece531fb00df9fb9d", 2025),
}
WORKSPACES = ("m2_v1", "m0r_v2", "m1_full_v1", "m0_v3", "m0_v2")
FACTOR_BY_INDEX = (
    "yeslab_206", "yeslab_357", "yeslab_372", "yeslab_375", "yeslab_381", "yeslab_384",
    "yeslab_389", "yeslab_390", "yeslab_406", "yeslab_425", "yeslab_443_reg30_max20_v1",
    "yeslab_464", "yeslab_471", "yeslab_484", "yeslab_96",
)
DENSE_IDX = (0, 1, 4, 9, 10, 11, 12, 13, 14)
DENSE = [FACTOR_BY_INDEX[i] for i in DENSE_IDX]
REMOVED_IN_V2 = ("yeslab_398", "yeslab_405")
NEGATIVE_CONTROL = ("yeslab_363",)
YEARS = (2020, 2021, 2022, 2023, 2024, 2025)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ART / "loose_ends")
    ap.add_argument("--skip-db", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    result = {
        "question": [
            "do the two factors v2 removed still have rows in factor_daily, as §BG.5 inferred from the "
            "materializer reading frozen v1?",
            "which rows in the four frozen test panels have ALL nine dense factors missing, and were "
            "those days among the 24 the baseline side drops?",
        ],
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True, "weights_loaded": False, "calculator_executed": False,
    }

    # ---- (a) the removed factors -------------------------------------------------
    if not args.skip_db:
        sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
        from walk_forward_common import new_holder, query
        holder = new_holder()
        ids = REMOVED_IN_V2 + NEGATIVE_CONTROL + ("yeslab_381",)
        quoted = ", ".join(f"'{i}'" for i in ids)
        per = {}
        for y in YEARS:
            sql = (f"select factor_id, count(*) as rows_all, count(value) as rows_val "
                   f"from loadTable('{TARGET_DB}', 'factor_daily') "
                   f"where factor_id in ({quoted}) and signal_date between {y}.01.01 : {y}.12.31 "
                   f"group by factor_id")
            ok, frame = query(holder, sql)
            if not ok or not isinstance(frame, pd.DataFrame):
                per[str(y)] = {"error": "query failed"}
                continue
            per[str(y)] = {str(r["factor_id"]): {"rows_all": int(r["rows_all"]),
                                                 "rows_with_value": int(r["rows_val"])}
                           for r in frame.to_dict("records")}
            print(f"  {y}: " + " ".join(f"{k}={v['rows_with_value']}" for k, v in per[str(y)].items()),
                  flush=True)
        result["removed_and_control_row_counts"] = {
            "per_year": per,
            "totals": {
                fid: sum(per[str(y)].get(fid, {}).get("rows_with_value", 0) for y in YEARS
                         if "error" not in per[str(y)])
                for fid in ids
            },
            "what_it_tests": (
                "yeslab_381 is the positive control (it is in both frozen sets and must be non-zero); "
                "yeslab_363 is the negative control (§AM/§BC: zero rows); yeslab_398 and yeslab_405 are "
                "the inference under test"
            ),
        }

    # ---- (b) rows where all nine dense factors are missing -----------------------
    panels = {}
    for fold, (ds, year) in FOLD_DATASETS.items():
        for ws in WORKSPACES:
            paths = sorted(glob.glob(str(ART / ws / "datasets" / ds / "split=test" / "*" / "*.parquet")))
            if paths:
                break
        cols = ["signal_date", "ts_code", "label_valid", "raw_return_5d"] + \
               [f"f{i:02d}" for i in DENSE_IDX]
        te = pd.concat([pd.read_parquet(p, columns=cols) for p in paths], ignore_index=True)
        te["signal_date"] = pd.to_datetime(te["signal_date"]).dt.date
        panels[fold] = (te, year)

    found = {}
    for fold, (te, year) in panels.items():
        feat = te[[f"f{i:02d}" for i in DENSE_IDX]].apply(pd.to_numeric, errors="coerce")
        all_missing = feat.isna().all(axis=1).to_numpy()
        n_missing_dense = feat.isna().sum(axis=1).to_numpy()
        scoreable = (te["label_valid"].fillna(False).astype(bool)
                     & np.isfinite(pd.to_numeric(te["raw_return_5d"], errors="coerce"))).to_numpy()
        sub = te.loc[all_missing]
        per_day = sub.groupby("signal_date").size()
        days_all_nine_missing = {str(k): int(v) for k, v in per_day.items()}
        found[fold] = {
            "test_year": year,
            "panel_rows": int(len(te)),
            "scoreable_rows": int(scoreable.sum()),
            "rows_with_all_nine_dense_missing": int(all_missing.sum()),
            "of_which_scoreable": int((all_missing & scoreable).sum()),
            "days_affected": len(days_all_nine_missing),
            "rows_per_affected_day": days_all_nine_missing,
            "distribution_of_how_many_dense_factors_are_missing": {
                str(k): int(v) for k, v in
                pd.Series(n_missing_dense[scoreable]).value_counts().sort_index().items()
            },
            "examples": [
                {"signal_date": str(r["signal_date"]), "ts_code": str(r["ts_code"]),
                 "label_valid": bool(r["label_valid"])}
                for r in sub.head(12).to_dict("records")
            ],
        }
        print(f"  {fold} ({year}): rows {len(te)}, all-nine-missing {int(all_missing.sum())} "
              f"on {len(days_all_nine_missing)} day(s) {days_all_nine_missing}", flush=True)
    result["all_nine_dense_missing"] = found
    result["all_nine_dense_missing"]["note"] = (
        "the nine dense factors' joint completeness is what the 0.90 coverage floor reads, so a day with "
        "rows missing all nine is a day the baseline side cannot score at all; the handoff list named "
        "2022-03-30, 2022-03-31 and 2022-05-31, and this is the measurement of whether they are there"
    )

    out = args.out / "removed_factors_and_allnine_rows.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
