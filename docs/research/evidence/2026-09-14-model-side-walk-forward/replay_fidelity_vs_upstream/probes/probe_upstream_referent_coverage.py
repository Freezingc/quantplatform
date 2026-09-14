"""For how much of the walk-forward window does an authoritative upstream referent exist at all?

The fidelity census compares factor_daily against dfs://YesLab. Before 2020 that comparison is not
merely weak, it is impossible: every one of the eight catalogued dense factors has zero upstream rows
for 2014-2019, so six of the eight training years have no authoritative referent to be faithful to.
This is what makes a training-year fidelity census unanswerable rather than merely unrun, and it bounds
any option that proposes injecting stored values.

Method: count each factor's own (source_table, source_factor_name) over ONE trading week per year,
06-10..06-20. A week rather than a full year because a full-year scan of these tables did not return in
ten minutes, while a week returns in under a second; and a week rather than a single day so that one
missing date cannot read as a missing year. Row counts are also divisible by the per-date row count, so
the window length is recoverable from the result and a short week cannot masquerade as a gap.

SAFETY: SELECT-only against dfs://YesLab. No writes except --out. No weights, no repository file touched.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
import yaml

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
DENSE9 = ["yeslab_206", "yeslab_357", "yeslab_381", "yeslab_425", "yeslab_443_reg30_max20_v1",
          "yeslab_464", "yeslab_471", "yeslab_484", "yeslab_96"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2014)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    years = list(range(args.start_year, args.end_year + 1))

    cat = yaml.safe_load((REPO / "configs" / "factors" / "source_catalog.yaml").read_text())
    by = {f["factor_id"]: f for f in cat["factors"]}

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import new_holder, query
    holder = new_holder()

    window = {y: (f"{y}.06.10", f"{y}.06.20") for y in years}
    out = {
        "question": "For how much of the walk-forward window does an authoritative upstream referent exist at all?",
        "method": ("row count of each dense factor's own (source_table, source_factor_name) in dfs://YesLab "
                   "over ONE trading week per year, 06-10..06-20; a zero means no referent that year"),
        "why_a_week_not_a_year": "a full-year scan of these tables did not return in ten minutes; a week returns in under a second",
        "why_a_week_not_a_day": "so that one missing date cannot read as a missing year",
        "window_per_year": {str(k): list(v) for k, v in window.items()},
        "dense_nine": DENSE9,
        "factors_without_a_catalog_entry": [f for f in DENSE9 if f not in by],
        "years": years, "per_factor": {}, "database_writes": False, "weights_written": False,
    }
    for fid in DENSE9:
        if fid not in by:
            out["per_factor"][fid] = {"in_catalog": False,
                                      "note": "a repair variant; it has no upstream referent by construction"}
            print(f"{fid:28s} -- not in source_catalog.yaml --", flush=True)
            continue
        entry = by[fid]; table, name = entry["source_table"], entry["source_factor_name"]
        counts = {}
        for year in years:
            lo, hi = window[year]
            ok, df = query(holder,
                f"select count(*) as n from loadTable('dfs://YesLab','{table}') "
                f"where factor_name = '{name}' and trade_date between date('{lo}') and date('{hi}')")
            counts[year] = int(df.iloc[0]["n"]) if ok and isinstance(df, pd.DataFrame) else None
        first = next((y for y in years if counts[y]), None)
        out["per_factor"][fid] = {
            "in_catalog": True, "source_table": table, "source_factor_name": name,
            "owner": entry.get("owner"), "display_name": entry.get("display_name"),
            "rows_in_the_one_week_window_per_year": counts, "first_year_with_rows": first,
            "training_years_2014_2021_without_a_referent": [y for y in range(2014, 2022) if not counts[y]],
            "oos_years_2022_2025_without_a_referent": [y for y in range(2022, 2026) if not counts[y]],
            "rows_per_date_implied_by_2020": (counts[2020] // 8) if counts.get(2020) else None,
        }
        print(f"{fid:28s}{table:14s}" + "".join(f"{(counts[y] or 0):>9,}" for y in years), flush=True)

    covered = [v for v in out["per_factor"].values() if v["in_catalog"]]
    out["verdict"] = {
        "catalogued_dense_factors": len(covered),
        "factors_whose_upstream_referent_starts_after_2014":
            sum(1 for v in covered if v["first_year_with_rows"] and v["first_year_with_rows"] > 2014),
        "earliest_year_any_factor_has_a_referent":
            min((v["first_year_with_rows"] for v in covered if v["first_year_with_rows"]), default=None),
        "training_years_with_no_referent_for_any_dense_factor":
            sorted({y for v in covered for y in v["training_years_2014_2021_without_a_referent"]}),
        "oos_years_with_no_referent": sorted({y for v in covered for y in v["oos_years_2022_2025_without_a_referent"]}),
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out["verdict"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
