#!/usr/bin/env python3
"""Read-only: on a cluster-B 'off' day, are GP_006's upstream values inside our universe or outside it?

`analyze_regime_shapes.py` found one factor whose frozen/upstream ratio is NOT stable between its own
low and high days: yeslab_375 (GP_006, factor_data8). On 2024-11-05 upstream reports 17,909 rows of
which 10,811 carry a value, yet the frozen factor_daily holds 536 and the panel coverage is 0.138 --
while on 2024-10-08 upstream reports 15,139 valued rows and the frozen table holds 3,745. If the
universe-in share were a constant property of GP_006's code set, both days would give the same ratio.

Two readings, and they matter in opposite directions:

  the valued rows on the 'off' day are mostly codes OUTSIDE the cn_liquid_a universe
      -> legitimate: the producer still publishes broadly but stopped covering our universe, and the
         frozen table is right
  a substantial number of valued rows ARE inside that day's universe but never reached factor_daily
      -> our materialization dropped usable rows, which is a defect and needs its own section

One query settles it: pull the day's rows for GP_006 and for the dense same-table control GP_012, and
intersect their codes with the codes the frozen panel actually has for that signal date.

SAFETY: SELECT-only on dfs://YesLab. No writes anywhere, no weights, no artifact mutation.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
ART = Path("/data/zhangyudong/qrv2_research_artifacts")
FOLD_OF_YEAR = {2022: "226a8dfc497c2de6b07516711f058f26a9270675916c9e67dbf7f087a07d2538",
                2023: "eb9bba1b5a0bd66d0ad6c82536b295de1fc079aec125cb27bb68491bdb21a234",
                2024: "9766e69f342511c5bda30188df15a7b9b17d936bdc26b9b60ba244e8e7e0c9e7",
                2025: "6f9478d7645c8c745a73f00c03b6122e17d0c06c05cc159ece531fb00df9fb9d"}
WORKSPACES = ("m2_v1", "m0r_v2", "m1_full_v1", "m0_v3", "m0_v2")
DAYS = ("2024-11-05", "2024-11-04", "2025-01-13", "2024-10-08")
NAMES = (("GP_006", "yeslab_375", "target cluster-B"),
         ("GP_015", "yeslab_384", "target cluster-B"),
         ("GP_037", "yeslab_406", "target cluster-B"),
         ("GP_012", "yeslab_381", "CONTROL dense, factor_data8"))
COL_OF = {"yeslab_375": "f03", "yeslab_384": "f05", "yeslab_406": "f08", "yeslab_381": "f04"}


def panel_codes(day: date) -> tuple[set, dict]:
    ds = FOLD_OF_YEAR[day.year]
    for ws in WORKSPACES:
        p = sorted(glob.glob(str(ART / ws / "datasets" / ds / "split=test" / "*" / "*.parquet")))
        if p:
            break
    else:
        raise FileNotFoundError(ds)
    frame = pd.concat(
        [pd.read_parquet(x, columns=["signal_date", "ts_code", "label_valid", "raw_return_5d"]
                         + list(COL_OF.values())) for x in p], ignore_index=True)
    frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.date
    frame = frame[frame["signal_date"] == day]
    codes = set(frame["ts_code"].astype(str))
    present = {f: set(frame.loc[pd.to_numeric(frame[c], errors="coerce").notna(), "ts_code"].astype(str))
               for f, c in COL_OF.items()}
    return codes, present


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ART / "sparse_upstream")
    args = ap.parse_args()
    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import new_holder, query

    holder = new_holder()
    result = {"question": "On a cluster-B 'off' day, are GP_006's upstream valued rows inside the "
                          "frozen panel's universe for that day, or outside it?",
              "generated_at_local": datetime.now().isoformat(timespec="seconds"),
              "safety": "SELECT-only on dfs://YesLab; panels read in place", "days": []}
    for ds_str in DAYS:
        day = date.fromisoformat(ds_str)
        universe, present = panel_codes(day)
        rec = {"signal_date": ds_str, "panel_codes_that_day": len(universe), "factors": {}}
        for name, fid, role in NAMES:
            sql = (f"select ts_code, value from loadTable('dfs://YesLab', 'factor_data8') "
                   f"where factor_name = '{name}' and trade_date = date('{ds_str.replace('-', '.')}')") \
                if name.startswith("GP_0") and fid in ("yeslab_375", "yeslab_384", "yeslab_381") else \
                (f"select ts_code, value from loadTable('dfs://YesLab', 'factor_data9') "
                 f"where factor_name = '{name}' and trade_date = date('{ds_str.replace('-', '.')}')")
            ok, frame = query(holder, sql)
            if not ok or not isinstance(frame, pd.DataFrame):
                rec["factors"][fid] = {"error": "query failed"}
                continue
            frame["ts_code"] = frame["ts_code"].astype(str)
            valued = frame.loc[pd.to_numeric(frame["value"], errors="coerce").notna(), "ts_code"]
            vset = set(valued)
            inside = vset & universe
            rec["factors"][fid] = {
                "source_name": name, "role": role,
                "upstream_rows": int(len(frame)),
                "upstream_valued_rows": int(len(valued)),
                "upstream_valued_codes": len(vset),
                "valued_codes_inside_that_days_panel": len(inside),
                "valued_codes_outside_that_days_panel": len(vset - universe),
                "share_of_valued_codes_inside_the_panel": len(inside) / len(vset) if vset else None,
                "panel_codes_with_a_value_for_this_factor": len(present[fid]),
                "panel_coverage": len(present[fid]) / len(universe) if universe else None,
                "valued_inside_panel_but_absent_from_the_panel": len(inside - present[fid]),
                "verdict": (
                    "MATERIALIZATION_LOSS_SUSPECTED" if len(inside - present[fid]) > 0.02 * len(universe)
                    else "FROZEN_TABLE_MATCHES_THE_UPSTREAM_VALUES_INSIDE_THE_UNIVERSE"
                ),
            }
        result["days"].append(rec)
        print(f"  {ds_str}: panel codes {len(universe)}; " + " ".join(
            f"{f.replace('yeslab_','')}={v.get('valued_codes_inside_that_days_panel')}"
            f"/{v.get('panel_codes_with_a_value_for_this_factor')}"
            f"(lost {v.get('valued_inside_panel_but_absent_from_the_panel')})"
            for f, v in rec["factors"].items()), flush=True)

    out = args.out / "gp006_codeset.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
