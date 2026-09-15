#!/usr/bin/env python3
"""Read-only: do our frozen yeslab_375 VALUES agree with the YesLab store's GP_006 values?

The availability evidence has narrowed to one place. §BE proved our frozen availability equals the
share of the cross-section whose power base is positive (r=0.99970 for GP_006). The store's own valued
share tracks that same base for five of the six sparse factors (r=0.981-0.989) but NOT for GP_006
(r=0.3895), and a lag scan from -10 to +10 trading days found no offset that fixes it (best lag = 0 for
all six). GP_015 shares GP_006's base expression `tanh(ts_delta_ratio_20(vol))` and matches at 0.9812,
which rules out "the producer's vol differs". What is left is the expression itself: either our
translation of GP_006 does not compute the producer's GP_006, or it computes the same thing and only
the missing-value pattern differs.

Those two are separable by comparing VALUES on the rows both sides have. If the Spearman correlation of
our frozen value against the store's value is near 1 on days where both are broadly populated, the two
are the same quantity and only availability differs. If it is low, the frozen column is not the
producer's GP_006, and every result in this pack that used yeslab_375 -- including §AO's cluster-B
redundancy FAILs -- was computed on a different factor than the catalogue names.

GP_015 (yeslab_384) and GP_003 (yeslab_372) ride along as controls that should correlate near 1.

SAFETY: SELECT-only on dfs://YesLab and dfs://QuantResearchV2Test. No calculator is executed, no
database is written, no artifact is mutated, no weight is loaded.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
TARGET_DB = "dfs://QuantResearchV2Test"
FACTORS = (
    ("factor_data8", "GP_006", "yeslab_375", "TARGET the one factor whose availability disagrees"),
    ("factor_data8", "GP_015", "yeslab_384", "CONTROL same base expression, availability agrees"),
    ("factor_data8", "GP_003", "yeslab_372", "CONTROL cluster A, availability agrees"),
    ("factor_data9", "GP_037", "yeslab_406", "CONTROL cluster B, availability agrees"),
)
DAYS = ("2024-10-09", "2024-02-28", "2024-10-11", "2024-11-05", "2022-03-15", "2025-06-16",
        "2024-09-30", "2023-06-14")


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
            "Do the frozen yeslab_375 values equal the YesLab store's GP_006 values on the rows both "
            "sides have, or is the frozen column a different quantity?"
        ),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "safety": "SELECT-only; no calculator executed; nothing written",
        "days": list(DAYS),
        "factors": {},
    }
    for table, name, fid, role in FACTORS:
        per_day = {}
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
            st["ts_code"] = st["ts_code"].astype(str)
            fz["ts_code"] = fz["ts_code"].astype(str)
            st["value"] = pd.to_numeric(st["value"], errors="coerce")
            fz["value"] = pd.to_numeric(fz["value"], errors="coerce")
            st = st.dropna(subset=["value"])
            merged = fz.merge(st, on="ts_code", how="inner", suffixes=("_frozen", "_store"))
            both = merged.dropna(subset=["value_frozen", "value_store"])
            entry = {
                "store_valued_rows": int(len(st)),
                "frozen_rows": int(len(fz)),
                "rows_in_both": int(len(both)),
                "overlap_share_of_frozen": (len(both) / len(fz)) if len(fz) else None,
            }
            if len(both) >= 30:
                rho = stats.spearmanr(both["value_frozen"], both["value_store"])
                pear = stats.pearsonr(both["value_frozen"], both["value_store"])
                rel = ((both["value_frozen"] - both["value_store"]).abs()
                       / both["value_store"].abs().replace(0, np.nan))
                entry.update({
                    "spearman": float(rho.statistic), "spearman_p": float(rho.pvalue),
                    "pearson": float(pear.statistic),
                    "median_relative_difference": float(rel.median()),
                    "share_within_1e-6_relative": float((rel <= 1e-6).mean()),
                    "exact_matches": int((both["value_frozen"] == both["value_store"]).sum()),
                    "sign_agreement": float((np.sign(both["value_frozen"])
                                             == np.sign(both["value_store"])).mean()),
                })
            per_day[ds] = entry
        rhos = [v["spearman"] for v in per_day.values() if "spearman" in v]
        result["factors"][fid] = {
            "source_name": name, "table": table, "role": role, "per_day": per_day,
            "spearman_median_over_days": float(np.median(rhos)) if rhos else None,
            "spearman_min_over_days": min(rhos) if rhos else None,
            "days_measured": len(rhos),
        }
        print(f"  {fid:12} {name:8} spearman median="
              f"{result['factors'][fid]['spearman_median_over_days']} min="
              f"{result['factors'][fid]['spearman_min_over_days']} days={len(rhos)}", flush=True)

    result["reading"] = (
        "a median Spearman near 1 with a small median relative difference means the frozen column IS the "
        "producer's factor and only its missing-value pattern differs; a low Spearman means the frozen "
        "column is a different quantity, which would put every result in this pack that used "
        "yeslab_375 -- including §AO's cluster-B redundancy FAILs -- on a factor other than the one the "
        "catalogue names"
    )
    out = args.out / "gp006_values_vs_store.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    for fid, v in result["factors"].items():
        print(f"\n{fid} ({v['source_name']}) -- {v['role']}")
        for d, e in v["per_day"].items():
            if "spearman" in e:
                print(f"   {d} both={e['rows_in_both']:5} spearman={e['spearman']:+.4f} "
                      f"pearson={e['pearson']:+.4f} med_rel_diff={e['median_relative_difference']:.3e} "
                      f"within1e-6={e['share_within_1e-6_relative']:.3f} sign_agree={e['sign_agreement']:.3f}")
            else:
                print(f"   {d} {e}")


if __name__ == "__main__":
    main()
