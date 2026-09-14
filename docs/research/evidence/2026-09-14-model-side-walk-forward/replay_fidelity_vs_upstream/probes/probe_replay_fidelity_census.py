"""A fidelity census of the frozen nine: how closely does the replay reproduce the authoritative store?

Everything the walk-forward chain consumes comes from factor_daily, which materialize_frozen_factor_daily.py
produces by REPLAYING each factor's calculator -- it does not copy the upstream YesLab observation. The
preregistered rule is explicit about not mixing historical YesLab values in, so a replay that differs from
the store is by design. But how much it differs, per factor, had never been measured, and the control run
in probe_replay_fidelity_controls.py turned up a max absolute difference of 1.4e7 on yeslab_381 and a
Spearman of 0.700 on yeslab_464. Those are large enough to matter for interpretation and for any option
that proposes injecting stored values.

This walks the frozen dense nine, resolves each one's upstream table and factor name from
configs/factors/source_catalog.yaml rather than retyping them, and reports a scale-free relative
difference alongside the rank correlations, so a unit convention difference is distinguishable from a
genuine signal difference.

SAFETY: SELECT-only against dfs://YesLab and dfs://QuantResearchV2Test. No writes except --out.
No weights, no artifact mutation, no repository file touched.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
CATALOG = REPO / "configs" / "factors" / "source_catalog.yaml"
DENSE9 = ["yeslab_206", "yeslab_357", "yeslab_381", "yeslab_425", "yeslab_443_reg30_max20_v1",
          "yeslab_464", "yeslab_471", "yeslab_484", "yeslab_96"]
PIT = {"yeslab_357", "yeslab_363", "yeslab_471"}


def sql_list(v) -> str:
    return ", ".join("'" + str(x).replace("'", "''") + "'" for x in v)


def classify(r: dict) -> str:
    if r["pairs"] == 0:
        return "NO_OVERLAP"
    if r["exact_match_rate"] >= 0.999:
        return "REPRODUCES_EXACTLY"
    ratio = r["std_ratio_replayed_over_upstream"]
    scale_off = ratio is None or ratio < 0.9 or ratio > 1.1
    if r["spearman"] >= 0.999 and not scale_off and r["median_relative_difference"] < 1e-6:
        return "SAME_SIGNAL_FLOAT_NOISE"
    if scale_off and r["spearman"] >= 0.99:
        return "SAME_RANK_DIFFERENT_SCALE"
    if r["spearman"] >= 0.99:
        return "SAME_RANK_DIFFERENT_VALUES"
    if r["spearman"] >= 0.90:
        return "CLOSE_RANK_MATERIAL_DIFFERENCE"
    return "DIVERGENT"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    days = sorted(args.days)

    cat = yaml.safe_load(CATALOG.read_text())
    by_id = {f["factor_id"]: f for f in cat["factors"]}
    resolved, unresolved = {}, []
    for fid in DENSE9:
        entry = by_id.get(fid)
        if entry is None:
            unresolved.append(fid); continue
        resolved[fid] = (entry["source_table"], entry["source_factor_name"],
                         entry.get("owner"), entry.get("unit"), entry.get("display_name"))

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    holder = new_holder()
    fpins = current_versions(holder, "factor_daily")

    rows = []
    for day in days:
        part = f"{day[:4]}-{day[5:7]}"
        if part not in fpins:
            raise SystemExit(f"unpublished partition {part}")
        fl = sql_list([fpins[part]]); dl = day.replace("-", ".")
        ok, fd = query(holder,
            f"select ts_code, factor_id, value from loadTable('{TARGET_DATABASE}','factor_daily') "
            f"where signal_date = date('{dl}') and factor_id in [{sql_list(resolved)}] and data_version in [{fl}]")
        if not ok or not isinstance(fd, pd.DataFrame):
            raise SystemExit(f"factor_daily query failed {day}: {fd}")
        fd["ts_code"] = fd["ts_code"].astype(str)
        fd["value"] = pd.to_numeric(fd["value"], errors="coerce")

        for fid, (table, name, owner, unit, disp) in resolved.items():
            ok2, up = query(holder,
                f"select ts_code, value from loadTable('dfs://YesLab','{table}') "
                f"where factor_name = '{name}' and trade_date = date('{dl}')")
            if not ok2 or not isinstance(up, pd.DataFrame):
                rows.append({"day": day, "factor_id": fid, "pairs": 0, "error": str(up)[:160]}); continue
            up["ts_code"] = up["ts_code"].astype(str)
            up["value"] = pd.to_numeric(up["value"], errors="coerce")
            a = fd[fd.factor_id == fid].dropna(subset=["value"]).groupby("ts_code")["value"].first()
            b = up.dropna(subset=["value"]).groupby("ts_code")["value"].first()
            j = pd.concat([a.rename("replayed"), b.rename("upstream")], axis=1, join="inner").dropna()
            if j.empty:
                rows.append({"day": day, "factor_id": fid, "display_name": disp, "owner": owner,
                             "unit": unit, "takes_industry_provider": fid in PIT, "pairs": 0}); continue
            diff = (j["replayed"] - j["upstream"]).abs()
            rel = (diff / j["upstream"].abs().replace(0, np.nan)).dropna()
            r = {
                "day": day, "factor_id": fid, "display_name": disp, "owner": owner, "unit": unit,
                "source_table": table, "source_factor_name": name,
                "takes_industry_provider": fid in PIT,
                "pairs": int(len(j)), "replayed_rows": int(len(a)), "upstream_rows": int(len(b)),
                "exact_match_rate": float((diff == 0).mean()),
                "within_1e_9_rate": float((diff <= 1e-9).mean()),
                "max_abs_difference": float(diff.max()),
                "mean_abs_difference": float(diff.mean()),
                "median_relative_difference": float(rel.median()) if len(rel) else None,
                "p95_relative_difference": float(rel.quantile(0.95)) if len(rel) else None,
                "pearson": float(np.corrcoef(j["replayed"], j["upstream"])[0, 1]),
                "spearman": float(j["replayed"].corr(j["upstream"], method="spearman")),
                "replayed_std": float(j["replayed"].std()), "upstream_std": float(j["upstream"].std()),
                "std_ratio_replayed_over_upstream":
                    float(j["replayed"].std() / j["upstream"].std()) if j["upstream"].std() else None,
                "replayed_mean": float(j["replayed"].mean()), "upstream_mean": float(j["upstream"].mean()),
            }
            r["classification"] = classify(r)
            rows.append(r)
        print(f"  {day}: {len(resolved)} factors", flush=True)

    df = pd.DataFrame([r for r in rows if r.get("pairs")])
    per_factor = {}
    for fid, g in df.groupby("factor_id"):
        per_factor[fid] = {
            "display_name": g["display_name"].iloc[0], "owner": g["owner"].iloc[0], "unit": g["unit"].iloc[0],
            "takes_industry_provider": bool(g["takes_industry_provider"].iloc[0]),
            "days": int(len(g)), "pairs_total": int(g["pairs"].sum()),
            "exact_match_rate_max": float(g["exact_match_rate"].max()),
            "spearman_min": float(g["spearman"].min()), "spearman_median": float(g["spearman"].median()),
            "pearson_min": float(g["pearson"].min()),
            "median_relative_difference_median": float(g["median_relative_difference"].median()),
            "std_ratio_min": float(g["std_ratio_replayed_over_upstream"].min()),
            "std_ratio_max": float(g["std_ratio_replayed_over_upstream"].max()),
            "max_abs_difference_max": float(g["max_abs_difference"].max()),
            "classification_by_day": sorted(set(g["classification"])),
        }
    out = {
        "question": "How closely does the frozen chain's replay reproduce the authoritative upstream store, "
                    "per factor, across the dense nine?",
        "databases_read": ["dfs://YesLab", TARGET_DATABASE],
        "catalog": str(CATALOG), "days": days,
        "factors_resolved_from_catalog": len(resolved),
        "factors_not_in_catalog": unresolved,
        "factors_reproducing_upstream_exactly": [f for f, v in per_factor.items()
                                                 if v["exact_match_rate_max"] >= 0.999],
        "per_factor": per_factor, "per_day_per_factor": rows,
        "database_writes": False, "weights_written": False,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_day_per_factor"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
