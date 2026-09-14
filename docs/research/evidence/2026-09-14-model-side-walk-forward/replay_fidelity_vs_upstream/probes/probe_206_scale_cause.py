"""Why is the replayed yeslab_206 about twice the scale of the upstream stored value?

The fidelity census found std(replayed)/std(upstream) = 1.78-2.37 for yeslab_206 with a median rank
correlation of 0.753 -- the worst of the dense nine, and a shape that is not rounding or a unit suffix.
Before treating that as a property of the replay, rule out the measurement itself. Three candidate
causes, tested in the order that would most embarrass this census:

  1 DEDUPLICATION. The upstream tables carry several rows per (code, date). The census took .first().
    If those rows disagree for 206, the comparison was against an arbitrary pick and the ratio is an
    artifact. Tested by counting rows per key and how many keys disagree.
  2 MULTIPLE FACTOR NAMES. The catalog maps yeslab_206 to one source_factor_name. If the same table
    holds a near-duplicate name (a v2, a variant, a renamed series), the replay may be reading a
    different series than the census compared against. Tested by listing every factor_name in the table
    whose values correlate with the replayed column.
  3 A REAL TRANSFORM. Only if 1 and 2 come back clean is the ratio a property of the replay, and then
    the shape of the relationship says what kind: a constant multiplier, an affine map, or a different
    window. Tested by regressing replayed on upstream and by looking at whether the ratio is stable
    across days and across the cross-section.

SAFETY: SELECT-only against dfs://YesLab and dfs://QuantResearchV2Test. No writes except --out.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
FID = "yeslab_206"


def sql_list(v) -> str:
    return ", ".join("'" + str(x).replace("'", "''") + "'" for x in v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    days = sorted(args.days)

    cat = yaml.safe_load((REPO / "configs" / "factors" / "source_catalog.yaml").read_text())
    entry = next(f for f in cat["factors"] if f["factor_id"] == FID)
    table, name = entry["source_table"], entry["source_factor_name"]

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    holder = new_holder()
    fpins = current_versions(holder, "factor_daily")

    dup_rows, per_day = [], []
    name_scan = {}
    for day in days:
        part = f"{day[:4]}-{day[5:7]}"
        if part not in fpins:
            raise SystemExit(f"unpublished partition {part}")
        dl = day.replace("-", ".")
        fl = sql_list([fpins[part]])
        ok, fd = query(holder,
            f"select ts_code, value from loadTable('{TARGET_DATABASE}','factor_daily') "
            f"where signal_date = date('{dl}') and factor_id = '{FID}' and data_version in [{fl}]")
        ok2, up = query(holder,
            f"select ts_code, value from loadTable('dfs://YesLab','{table}') "
            f"where factor_name = '{name}' and trade_date = date('{dl}')")
        if not ok or not ok2:
            raise SystemExit(f"query failed on {day}")
        fd["ts_code"] = fd["ts_code"].astype(str); up["ts_code"] = up["ts_code"].astype(str)
        fd["value"] = pd.to_numeric(fd["value"], errors="coerce")
        up["value"] = pd.to_numeric(up["value"], errors="coerce")

        # --- cause 1: duplication ---
        g = up.dropna(subset=["value"]).groupby("ts_code")["value"]
        stat = g.agg(["count", "nunique", "min", "max", "mean", "first"])
        dup = stat[stat["count"] > 1]
        disagree = dup[dup["nunique"] > 1]
        dup_rows.append({"day": day, "upstream_rows": int(len(up)), "distinct_codes": int(len(stat)),
                         "keys_with_multiple_rows": int(len(dup)),
                         "keys_whose_rows_disagree": int(len(disagree)),
                         "max_spread_within_a_key": float((dup["max"] - dup["min"]).max()) if len(dup) else 0.0})

        # --- comparison under three dedup rules, so the ratio cannot be a pick artifact ---
        a = fd.dropna(subset=["value"]).groupby("ts_code")["value"].first()
        variants = {"first": stat["first"], "mean": stat["mean"], "max": stat["max"], "min": stat["min"]}
        v = {"day": day, "replayed_std": float(a.std()), "replayed_mean": float(a.mean())}
        for label, s in variants.items():
            j = pd.concat([a.rename("r"), s.rename("u")], axis=1, join="inner").dropna()
            if j.empty:
                continue
            slope = float(np.polyfit(j["u"], j["r"], 1)[0])
            v[label] = {"pairs": int(len(j)), "upstream_std": float(j["u"].std()),
                        "std_ratio": float(j["r"].std() / j["u"].std()),
                        "spearman": float(j["r"].corr(j["u"], method="spearman")),
                        "ols_slope_replayed_on_upstream": slope,
                        "ols_intercept": float(np.polyfit(j["u"], j["r"], 1)[1]),
                        "median_ratio_pointwise": float((j["r"] / j["u"].replace(0, np.nan)).median())}
        per_day.append(v)
        print(f"  {day}: upstream rows {len(up)}, keys {len(stat)}, disagreeing keys {len(disagree)}", flush=True)

    # --- cause 2: is there a better-matching factor_name in the same table? ---
    day = days[len(days) // 2]; dl = day.replace("-", ".")
    ok, names = query(holder,
        f"select distinct factor_name from loadTable('dfs://YesLab','{table}')")
    candidate_names = []
    if ok and isinstance(names, pd.DataFrame):
        candidate_names = [str(x) for x in names["factor_name"].tolist()]
    a = None
    ok, fd = query(holder,
        f"select ts_code, value from loadTable('{TARGET_DATABASE}','factor_daily') "
        f"where signal_date = date('{dl}') and factor_id = '{FID}' "
        f"and data_version in [{sql_list([fpins[f'{day[:4]}-{day[5:7]}']])}]")
    fd["ts_code"] = fd["ts_code"].astype(str); fd["value"] = pd.to_numeric(fd["value"], errors="coerce")
    a = fd.dropna(subset=["value"]).groupby("ts_code")["value"].first()
    scored = []
    for nm in candidate_names:
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
        if len(j) < 50:
            continue
        scored.append({"factor_name": nm, "pairs": int(len(j)),
                       "spearman": float(j["r"].corr(j["u"], method="spearman")),
                       "std_ratio": float(j["r"].std() / j["u"].std()) if j["u"].std() else None,
                       "exact_match_rate": float(((j["r"] - j["u"]).abs() == 0).mean())})
    scored.sort(key=lambda r: -r["spearman"])

    ratios = [v["mean"]["std_ratio"] for v in per_day if "mean" in v]
    out = {
        "question": "Why is replayed yeslab_206 about twice the scale of the stored value?",
        "factor_id": FID, "display_name": entry.get("display_name"), "owner": entry.get("owner"),
        "source_table": table, "source_factor_name": name, "unit": entry.get("unit"),
        "databases_read": ["dfs://YesLab", TARGET_DATABASE], "days": days,
        "cause_1_deduplication": {
            "keys_whose_rows_disagree_total": sum(r["keys_whose_rows_disagree"] for r in dup_rows),
            "keys_with_multiple_rows_total": sum(r["keys_with_multiple_rows"] for r in dup_rows),
            "max_spread_within_a_key": max(r["max_spread_within_a_key"] for r in dup_rows),
            "verdict": None, "per_day": dup_rows,
        },
        "cause_1b_ratio_under_four_dedup_rules": {
            rule: {"min": min((v[rule]["std_ratio"] for v in per_day if rule in v), default=None),
                   "max": max((v[rule]["std_ratio"] for v in per_day if rule in v), default=None),
                   "median_spearman": float(np.median([v[rule]["spearman"] for v in per_day if rule in v])) if any(rule in v for v in per_day) else None,
                   "median_ols_slope": float(np.median([v[rule]["ols_slope_replayed_on_upstream"] for v in per_day if rule in v])) if any(rule in v for v in per_day) else None}
            for rule in ("first", "mean", "max", "min")
        },
        "cause_2_other_names_in_the_same_table": {
            "names_in_table": len(candidate_names),
            "names_scored_against_the_replayed_column": len(scored),
            "top_5_by_spearman": scored[:5],
            "best_alternative_beats_the_catalog_name": None,
        },
        "cause_3_shape": {"std_ratio_range_across_days": [min(ratios), max(ratios)] if ratios else None},
        "per_day": per_day,
        "database_writes": False, "weights_written": False,
    }
    d = out["cause_1_deduplication"]
    d["verdict"] = ("RULED OUT -- no upstream key disagrees with itself, so the dedup pick cannot "
                    "explain the ratio" if d["keys_whose_rows_disagree_total"] == 0 else
                    f"POSSIBLE ARTIFACT -- {d['keys_whose_rows_disagree_total']} keys disagree")
    catalog_spear = float(np.median([v["mean"]["spearman"] for v in per_day if "mean" in v]))
    best = scored[0] if scored else None
    out["cause_2_other_names_in_the_same_table"]["best_alternative_beats_the_catalog_name"] = (
        bool(best and best["spearman"] > catalog_spear + 0.01))
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_day"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
