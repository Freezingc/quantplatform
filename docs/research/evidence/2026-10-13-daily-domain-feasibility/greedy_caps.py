"""Effective-dimension count under both redundancy caps, plus the decision table.

redundancy_survivors.py's greedy pass uses the 0.90 BLOCK cap, which admits buy_intensity and
repay_intensity together even though their daily |rho| is 0.881 -- i.e. it counts one economic
signal (margin turnover) as two dimensions.  This re-runs the same greedy rule at 0.90 and 0.80
from the persisted JSON (no refitting, no new data reads) and emits the markdown decision table
used by the README, joining in the 2022-2024 pooled ICIR and the benchmark row.
"""
from __future__ import annotations

import json
import pathlib

import pandas as pd

OUT = pathlib.Path(__file__).resolve().parent
ANA = OUT / "analysis"
RETENTION_BAR = 0.30
ICIR_BAR = 0.50
RESIDUAL_IC_BAR = 0.02  # same magnitude as the signed RankIC leg of the production admission rule


def retention_flag(ret: float | None) -> str:
    """Signed reading of set-incremental retention.

    A negative retention is not a weak pass: the residual RankIC has the opposite sign to the raw
    RankIC, so conditioning on the dense nine does not merely absorb the signal, it reverses it.
    winner_rate is exactly that case here (-0.0317 raw -> +0.0097 residual), which a |retention|
    comparison would have scored as a pass.
    """
    if ret is None:
        return "NO_ESTIMATE"
    if ret < 0:
        return "SIGN_FLIP"
    return "PASS" if ret >= RETENTION_BAR else "WEAK"


def greedy(entries: list[dict], cap: float, require_retention: bool = False,
           ) -> tuple[list[str], dict[str, list[str]]]:
    order = sorted(entries, key=lambda e: -abs(e["screen_icir"]))
    admitted: list[str] = []
    rejected: dict[str, list[str]] = {}
    for e in order:
        blockers = []
        fw = e.get("worst_frozen_median_abs_rho") or 0.0
        if fw >= cap:
            blockers.append(f"frozen:{e['worst_frozen_partner']}={fw:.3f}")
        for kept in admitted:
            r = e["pairwise_vs_other_survivors"].get(kept)
            if r and r["median_abs_rho"] >= cap:
                blockers.append(f"admitted:{kept}={r['median_abs_rho']:.3f}")
        if require_retention:
            ret = e["set_incremental_given_dense9"].get("retention")
            flag = retention_flag(ret)
            if flag != "PASS":
                blockers.append(f"retention:{'n/a' if ret is None else round(ret, 3)}={flag}")
        if blockers:
            rejected[e["candidate"]] = blockers
        else:
            admitted.append(e["candidate"])
    return admitted, rejected


def main() -> int:
    res = json.loads((ANA / "redundancy_survivors.json").read_text())
    entries = res["candidates"]
    pooled = pd.read_csv(ANA / "ic_screen_pooled_windows.csv")
    bench = pd.read_csv(ANA / "benchmark_frozen_v2_icir_2022_2024.csv")

    out = {}
    for cap in (0.90, 0.80):
        admitted, rejected = greedy(entries, cap)
        out[f"cap_{cap:.2f}"] = {"admitted": admitted, "count": len(admitted), "rejected": rejected}
        print(f"greedy @ median|rho| cap {cap:.2f}: {len(admitted)} dimensions -> {admitted}")
        for k, v in rejected.items():
            print(f"    rejected {k}: {v}")
    admitted, rejected = greedy(entries, 0.80, require_retention=True)
    out["cap_0.80_retention_aware"] = {"admitted": admitted, "count": len(admitted), "rejected": rejected}
    print(f"\ngreedy @ cap 0.80 AND signed retention >= {RETENTION_BAR}: {len(admitted)} dimensions -> {admitted}")
    for k, v in rejected.items():
        print(f"    rejected {k}: {v}")

    rows = []
    for e in entries:
        p = pooled[(pooled["candidate"] == e["candidate"]) & (pooled["lag"] == e["lag"])]
        si = e["set_incremental_given_dense9"]
        ret = si.get("retention")
        rows.append({
            "candidate": e["candidate"], "source": e["source"], "lag": e["lag"],
            "used_safe_lag": e["used_safe_lag"],
            "icir_2022_2025": round(e["screen_icir"], 3),
            "icir_2022_2024": round(float(p["icir_2022_2024"].iloc[0]), 3) if len(p) else None,
            "mean_ic": round(e["screen_mean_rank_ic"], 4),
            "coverage": round(e["coverage_of_labelled_days"], 3),
            "worst_frozen_rho": round(e["worst_frozen_median_abs_rho"] or 0, 3),
            "worst_frozen_partner": e["worst_frozen_partner"],
            "worst_survivor_rho": round(e["worst_survivor_median_abs_rho"] or 0, 3),
            "worst_survivor_partner": e["worst_survivor_partner"],
            "retention_given_dense9": round(ret, 3) if ret is not None else None,
            "retention_flag": retention_flag(ret),
            "residual_rank_ic": round(si.get("mean_residual_rank_ic"), 4) if si.get("days") else None,
            "raw_rank_ic_on_retention_days": round(si.get("mean_raw_rank_ic"), 4) if si.get("days") else None,
            "retention_days": si.get("days"),
            "within_day_coverage": round(si.get("mean_within_day_coverage"), 4) if si.get("days") else None,
            "clears_icir_bar": abs(e["screen_icir"]) >= ICIR_BAR,
            "clears_residual_ic_bar": bool(si.get("days")) and abs(si.get("mean_residual_rank_ic") or 0) >= RESIDUAL_IC_BAR
            and retention_flag(ret) != "SIGN_FLIP",
            "dim_at_cap_090": e["candidate"] in out["cap_0.90"]["admitted"],
            "dim_at_cap_080": e["candidate"] in out["cap_0.80"]["admitted"],
            "dim_retention_aware": e["candidate"] in out["cap_0.80_retention_aware"]["admitted"],
        })
    df = pd.DataFrame(rows)
    df["abs_residual_ic"] = df["residual_rank_ic"].abs()
    df = df.sort_values("abs_residual_ic", ascending=False).drop(columns=["abs_residual_ic"])
    df.to_csv(ANA / "decision_table.csv", index=False)

    summary = {"scope": "DAILY_DOMAIN_DIMENSION_SUMMARY_V1", "window_redundancy": "2025",
               "window_screen": "2022-2025", "icir_bar": ICIR_BAR, "retention_bar": RETENTION_BAR,
               "greedy": out, "decision_table": json.loads(df.to_json(orient="records")),
               "benchmark_frozen_v2_2022_2024": {
                   "min_abs_icir": float(bench["abs_icir"].min()),
                   "median_abs_icir": float(bench["abs_icir"].median()),
                   "max_abs_icir": float(bench["abs_icir"].max()),
                   "weakest_factor": bench.loc[bench["abs_icir"].idxmin(), "factor_id"],
                   "factors": json.loads(bench.to_json(orient="records"))}}
    (ANA / "dimension_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))

    print("\n=== decision table ===")
    print(df[["candidate", "source", "lag", "icir_2022_2025", "mean_ic", "coverage",
              "worst_frozen_rho", "worst_survivor_rho", "retention_given_dense9", "retention_flag",
              "residual_rank_ic", "clears_icir_bar", "clears_residual_ic_bar",
              "dim_at_cap_080", "dim_retention_aware"]].to_string(index=False))
    print(f"\nfrozen v2 benchmark |ICIR| 2022-2024: min={bench['abs_icir'].min():.3f} "
          f"({bench.loc[bench['abs_icir'].idxmin(), 'factor_id']}) median={bench['abs_icir'].median():.3f} "
          f"max={bench['abs_icir'].max():.3f}")
    print("wrote analysis/decision_table.csv and analysis/dimension_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
