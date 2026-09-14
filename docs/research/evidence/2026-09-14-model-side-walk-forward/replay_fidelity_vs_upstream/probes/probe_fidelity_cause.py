"""Why does the replayed value of factor X differ from the stored one? Three candidates, worst-first.

Generalises probe_206_scale_cause.py to any factor in source_catalog.yaml. The order matters: the first
two candidates are ways THIS MEASUREMENT could be wrong, so they are tested before the finding is
allowed to be about the replay.

  1 DEDUPLICATION. The census collapsed the upstream side to one row per key. If the upstream table
    carries several rows per (code, date) for this factor and they disagree, every ratio reported for it
    was against an arbitrary pick. Tested by counting rows per key and how many keys disagree, and by
    recomputing the comparison under four dedup rules (first / mean / max / min).
  2 MULTIPLE FACTOR NAMES. If the same table holds a near-duplicate name, the replay may read a different
    series than the catalog names. Tested by scoring every factor_name in the table against the replayed
    column and asking whether any beats the catalog's.
  3 A REAL DIFFERENCE. Only if 1 and 2 are clean is the gap a property of the replay. The shape then says
    what kind: a pure rescaling preserves ranks and puts the pointwise ratio equal to the OLS slope with a
    zero intercept, so violating any of those three distinguishes "different scale" from "different
    computation".

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


def sql_list(v) -> str:
    return ", ".join("'" + str(x).replace("'", "''") + "'" for x in v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", required=True)
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--min-overlap", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    fid, days = args.factor, sorted(args.days)

    cat = yaml.safe_load((REPO / "configs" / "factors" / "source_catalog.yaml").read_text())
    matches = [f for f in cat["factors"] if f["factor_id"] == fid]
    if not matches:
        raise SystemExit(f"{fid} has no entry in source_catalog.yaml -- it has no upstream referent to "
                         "compare against (this is itself the finding for repair variants)")
    entry = matches[0]
    table, name = entry["source_table"], entry["source_factor_name"]

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    holder = new_holder()
    fpins = current_versions(holder, "factor_daily")

    dup_rows, per_day = [], []
    for day in days:
        part = f"{day[:4]}-{day[5:7]}"
        if part not in fpins:
            raise SystemExit(f"unpublished partition {part}")
        dl = day.replace("-", "."); fl = sql_list([fpins[part]])
        ok, fd = query(holder,
            f"select ts_code, value from loadTable('{TARGET_DATABASE}','factor_daily') "
            f"where signal_date = date('{dl}') and factor_id = '{fid}' and data_version in [{fl}]")
        ok2, up = query(holder,
            f"select ts_code, value from loadTable('dfs://YesLab','{table}') "
            f"where factor_name = '{name}' and trade_date = date('{dl}')")
        if not ok or not ok2 or not isinstance(fd, pd.DataFrame) or not isinstance(up, pd.DataFrame):
            raise SystemExit(f"query failed on {day}: {fd if not ok else up}")
        for df in (fd, up):
            df["ts_code"] = df["ts_code"].astype(str)
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

        stat = up.dropna(subset=["value"]).groupby("ts_code")["value"].agg(
            ["count", "nunique", "min", "max", "mean", "first"])
        dup = stat[stat["count"] > 1]; disagree = dup[dup["nunique"] > 1]
        dup_rows.append({"day": day, "upstream_rows": int(len(up)), "distinct_codes": int(len(stat)),
                         "keys_with_multiple_rows": int(len(dup)),
                         "keys_whose_rows_disagree": int(len(disagree)),
                         "max_spread_within_a_key": float((dup["max"] - dup["min"]).max()) if len(dup) else 0.0})

        a = fd.dropna(subset=["value"]).groupby("ts_code")["value"].first()
        v = {"day": day, "replayed_rows": int(len(a)), "replayed_std": float(a.std()) if len(a) > 1 else None,
             "replayed_mean": float(a.mean()) if len(a) else None}
        for label in ("first", "mean", "max", "min"):
            j = pd.concat([a.rename("r"), stat[label].rename("u")], axis=1, join="inner").dropna()
            if len(j) < 2:
                continue
            slope, icept = np.polyfit(j["u"], j["r"], 1)
            v[label] = {"pairs": int(len(j)), "upstream_std": float(j["u"].std()),
                        "std_ratio": float(j["r"].std() / j["u"].std()) if j["u"].std() else None,
                        "spearman": float(j["r"].corr(j["u"], method="spearman")),
                        "pearson": float(np.corrcoef(j["r"], j["u"])[0, 1]),
                        "ols_slope": float(slope), "ols_intercept": float(icept),
                        "median_ratio_pointwise": float((j["r"] / j["u"].replace(0, np.nan)).median()),
                        "exact_match_rate": float(((j["r"] - j["u"]).abs() == 0).mean())}
        per_day.append(v)
        print(f"  {day}: upstream rows {len(up)}, keys {len(stat)}, disagreeing {len(disagree)}, "
              f"pairs {v.get('mean', {}).get('pairs')}", flush=True)

    # candidate 2: scan every other factor_name in the same table against the replayed column
    mid = days[len(days) // 2]; dl = mid.replace("-", ".")
    ok, names = query(holder, f"select distinct factor_name from loadTable('dfs://YesLab','{table}')")
    all_names = [str(x) for x in names["factor_name"].tolist()] if ok and isinstance(names, pd.DataFrame) else []
    ok, fd = query(holder,
        f"select ts_code, value from loadTable('{TARGET_DATABASE}','factor_daily') "
        f"where signal_date = date('{dl}') and factor_id = '{fid}' "
        f"and data_version in [{sql_list([fpins[f'{mid[:4]}-{mid[5:7]}']])}]")
    fd["ts_code"] = fd["ts_code"].astype(str); fd["value"] = pd.to_numeric(fd["value"], errors="coerce")
    a = fd.dropna(subset=["value"]).groupby("ts_code")["value"].first()
    scored = []
    for nm in all_names:
        if nm == name:
            continue
        ok3, up2 = query(holder,
            f"select ts_code, value from loadTable('dfs://YesLab','{table}') "
            f"where factor_name = '{nm}' and trade_date = date('{dl}')")
        if not ok3 or not isinstance(up2, pd.DataFrame) or up2.empty:
            continue
        up2["ts_code"] = up2["ts_code"].astype(str)
        up2["value"] = pd.to_numeric(up2["value"], errors="coerce")
        b = up2.dropna(subset=["value"]).groupby("ts_code")["value"].mean()
        j = pd.concat([a.rename("r"), b.rename("u")], axis=1, join="inner").dropna()
        if len(j) < args.min_overlap:
            continue
        scored.append({"factor_name": nm, "pairs": int(len(j)),
                       "spearman": float(j["r"].corr(j["u"], method="spearman")),
                       "std_ratio": float(j["r"].std() / j["u"].std()) if j["u"].std() else None,
                       "exact_match_rate": float(((j["r"] - j["u"]).abs() == 0).mean())})
    scored.sort(key=lambda r: -r["spearman"])

    ms = [v["mean"] for v in per_day if "mean" in v]
    catalog_spear = float(np.median([m["spearman"] for m in ms])) if ms else None
    out = {
        "question": f"Why does the replayed {fid} differ from the stored value?",
        "factor_id": fid, "display_name": entry.get("display_name"), "owner": entry.get("owner"),
        "source_table": table, "source_factor_name": name, "unit": entry.get("unit"),
        "databases_read": ["dfs://YesLab", TARGET_DATABASE], "days": days,
        "cause_1_deduplication": {
            "keys_with_multiple_rows_total": sum(r["keys_with_multiple_rows"] for r in dup_rows),
            "keys_whose_rows_disagree_total": sum(r["keys_whose_rows_disagree"] for r in dup_rows),
            "max_spread_within_a_key": max(r["max_spread_within_a_key"] for r in dup_rows),
            "dedup_rules_agree": len({round(v["mean"]["std_ratio"], 12) for v in per_day if "mean" in v}) > 0
                                 and all(abs(v["first"]["std_ratio"] - v["mean"]["std_ratio"]) < 1e-12
                                         for v in per_day if "mean" in v),
            "verdict": None, "per_day": dup_rows,
        },
        "cause_1b_ratio_under_four_dedup_rules": {
            rule: {"std_ratio_min": min((v[rule]["std_ratio"] for v in per_day if rule in v), default=None),
                   "std_ratio_max": max((v[rule]["std_ratio"] for v in per_day if rule in v), default=None),
                   "spearman_median": float(np.median([v[rule]["spearman"] for v in per_day if rule in v])) if any(rule in v for v in per_day) else None,
                   "ols_slope_median": float(np.median([v[rule]["ols_slope"] for v in per_day if rule in v])) if any(rule in v for v in per_day) else None}
            for rule in ("first", "mean", "max", "min")
        },
        "cause_2_other_names_in_the_same_table": {
            "names_in_table": len(all_names), "names_scored": len(scored),
            "catalog_name_spearman_median": catalog_spear,
            "top_5_by_spearman": scored[:5],
            "best_alternative_beats_the_catalog_name":
                bool(scored and catalog_spear is not None and scored[0]["spearman"] > catalog_spear + 0.01),
        },
        "cause_3_shape": {
            "std_ratio_range": [min(m["std_ratio"] for m in ms), max(m["std_ratio"] for m in ms)] if ms else None,
            "ols_slope_range": [min(m["ols_slope"] for m in ms), max(m["ols_slope"] for m in ms)] if ms else None,
            "ols_intercept_range": [min(m["ols_intercept"] for m in ms), max(m["ols_intercept"] for m in ms)] if ms else None,
            "spearman_range": [min(m["spearman"] for m in ms), max(m["spearman"] for m in ms)] if ms else None,
            "pointwise_ratio_range": [min(m["median_ratio_pointwise"] for m in ms),
                                      max(m["median_ratio_pointwise"] for m in ms)] if ms else None,
            "pure_rescaling_would_require": {
                "spearman": 1.0, "intercept": 0.0, "pointwise_ratio_equals_slope": True},
            "is_pure_rescaling": None,
        },
        "per_day": per_day, "database_writes": False, "weights_written": False,
    }
    d1 = out["cause_1_deduplication"]
    d1["verdict"] = ("RULED OUT -- no upstream key disagrees with itself, so the dedup pick cannot "
                     "explain the gap" if d1["keys_whose_rows_disagree_total"] == 0 else
                     f"POSSIBLE ARTIFACT -- {d1['keys_whose_rows_disagree_total']} keys disagree, "
                     f"max spread {d1['max_spread_within_a_key']}")
    c3 = out["cause_3_shape"]
    if ms:
        c3["is_pure_rescaling"] = bool(
            c3["spearman_range"][0] > 0.999
            and abs(c3["ols_intercept_range"][0]) < 1e-9 and abs(c3["ols_intercept_range"][1]) < 1e-9
            and all(abs(m["median_ratio_pointwise"] - m["ols_slope"]) < 1e-6 for m in ms))
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_day",)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
