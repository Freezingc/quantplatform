"""Control test: is the replayed-versus-upstream gap specific to the industry-dependent factors?

probe_option3_upstream_ceiling.py found that factor_daily does not reproduce the upstream YesLab values
for yeslab_357 (1-2% exact, Spearman 0.95-0.996) or yeslab_471 (0% exact on every day, Spearman
0.82-0.88). Both are in PIT_FACTORS at materialize_frozen_factor_daily.py:50 and are recomputed through
replay._replay_factor(..., industry_provider=prov), so a plausible reading is "same formula, different
industry partition". That reading is only warranted if factors that do NOT take an industry provider
reproduce upstream closely. This measures exactly that on the same days.

Controls, chosen the way star_upstream_coverage.py chose them -- same table, same owner neighbourhood:
  yeslab_381  factor_data8   GP_012                     control on 357's table
  yeslab_464  factor_data10  GP_095                     control on 471's table
  yeslab_484  factor_data10  羊群效应_CSAD波动率21        control sharing 471's owner
None of the three is in PIT_FACTORS.

SAFETY: SELECT-only against dfs://YesLab and dfs://QuantResearchV2Test. No writes except --out.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
FACTORS = {
    "yeslab_357": ("factor_data8", "行业内收益标准化绝对值", "TARGET industry-dependent"),
    "yeslab_471": ("factor_data10", "AttentionSpillover", "TARGET industry-dependent"),
    "yeslab_381": ("factor_data8", "GP_012", "CONTROL same table as 357"),
    "yeslab_464": ("factor_data10", "GP_095", "CONTROL same table as 471"),
    "yeslab_484": ("factor_data10", "羊群效应_CSAD波动率21", "CONTROL same owner as 471"),
}


def sql_list(v) -> str:
    return ", ".join("'" + str(x).replace("'", "''") + "'" for x in v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    days = sorted(args.days)

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    holder = new_holder()
    fpins = current_versions(holder, "factor_daily")

    rows = []
    for day in days:
        parts = [f"{day[:4]}-{day[5:7]}"]
        miss = [p for p in parts if p not in fpins]
        if miss:
            raise SystemExit(f"unpublished partition: {miss}")
        fl = sql_list(sorted({fpins[p] for p in parts}))
        dl = day.replace("-", ".")
        ok, fd = query(holder,
            f"select ts_code, factor_id, value from loadTable('{TARGET_DATABASE}','factor_daily') "
            f"where signal_date = date('{dl}') and factor_id in [{sql_list(FACTORS)}] and data_version in [{fl}]")
        if not ok or not isinstance(fd, pd.DataFrame):
            raise SystemExit(f"factor_daily query failed {day}: {fd}")
        fd["ts_code"] = fd["ts_code"].astype(str)
        fd["value"] = pd.to_numeric(fd["value"], errors="coerce")

        for fid, (table, name, role) in FACTORS.items():
            ok2, up = query(holder,
                f"select ts_code, trade_date, value from loadTable('dfs://YesLab','{table}') "
                f"where factor_name = '{name}' and trade_date = date('{dl}')")
            if not ok2 or not isinstance(up, pd.DataFrame):
                rows.append({"day": day, "factor_id": fid, "role": role, "error": str(up)[:160]}); continue
            up["ts_code"] = up["ts_code"].astype(str)
            up["value"] = pd.to_numeric(up["value"], errors="coerce")
            a = fd[fd.factor_id == fid].dropna(subset=["value"]).groupby("ts_code")["value"].first()
            b = up.dropna(subset=["value"]).groupby("ts_code")["value"].first()
            j = pd.concat([a.rename("replayed"), b.rename("upstream")], axis=1, join="inner").dropna()
            if j.empty:
                rows.append({"day": day, "factor_id": fid, "role": role, "pairs": 0}); continue
            diff = (j["replayed"] - j["upstream"]).abs()
            denom = j["upstream"].abs().replace(0, np.nan)
            rows.append({
                "day": day, "factor_id": fid, "role": role, "pairs": int(len(j)),
                "exact_matches": int((diff == 0).sum()), "exact_match_rate": float((diff == 0).mean()),
                "within_1e_9_rate": float((diff <= 1e-9).mean()),
                "within_1e_6_rate": float((diff <= 1e-6).mean()),
                "max_abs_difference": float(diff.max()), "mean_abs_difference": float(diff.mean()),
                "median_rel_difference": float((diff / denom).median()) if denom.notna().any() else None,
                "pearson": float(np.corrcoef(j["replayed"], j["upstream"])[0, 1]),
                "spearman": float(j["replayed"].corr(j["upstream"], method="spearman")),
                "replayed_std": float(j["replayed"].std()), "upstream_std": float(j["upstream"].std()),
                "replayed_mean": float(j["replayed"].mean()), "upstream_mean": float(j["upstream"].mean()),
                "replayed_rows": int(len(a)), "upstream_rows": int(len(b)),
            })
        print(f"  {day}: {len(FACTORS)} factors compared", flush=True)

    df = pd.DataFrame([r for r in rows if r.get("pairs")])
    summary = {}
    for (fid, role), g in df.groupby(["factor_id", "role"]):
        summary[fid] = {
            "role": role, "days": int(len(g)), "pairs_total": int(g["pairs"].sum()),
            "exact_match_rate_min": float(g["exact_match_rate"].min()),
            "exact_match_rate_max": float(g["exact_match_rate"].max()),
            "spearman_min": float(g["spearman"].min()), "spearman_max": float(g["spearman"].max()),
            "pearson_min": float(g["pearson"].min()),
            "max_abs_difference_max": float(g["max_abs_difference"].max()),
            "mean_abs_difference_median": float(g["mean_abs_difference"].median()),
        }
    controls = [v for k, v in summary.items() if v["role"].startswith("CONTROL")]
    targets = [v for k, v in summary.items() if v["role"].startswith("TARGET")]
    out = {
        "question": "Is the replayed-versus-upstream gap specific to the two industry-dependent factors?",
        "databases_read": ["dfs://YesLab", TARGET_DATABASE],
        "days": days, "per_factor": summary, "per_day_per_factor": rows,
        "control_spearman_min": min((c["spearman_min"] for c in controls), default=None),
        "control_exact_match_rate_min": min((c["exact_match_rate_min"] for c in controls), default=None),
        "target_spearman_min": min((t["spearman_min"] for t in targets), default=None),
        "target_exact_match_rate_max": max((t["exact_match_rate_max"] for t in targets), default=None),
        "database_writes": False, "weights_written": False,
    }
    out["verdict"] = (
        "INDUSTRY-SPECIFIC" if (out["control_spearman_min"] or 0) > 0.999
        and (out["target_spearman_min"] or 1) < 0.99 else
        "GENERAL -- the controls diverge too, so the gap is not attributable to the industry input"
    )
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_day_per_factor",)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
