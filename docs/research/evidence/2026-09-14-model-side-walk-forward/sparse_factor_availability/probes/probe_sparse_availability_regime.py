#!/usr/bin/env python3
"""Read-only: is the six sparse factors' missingness a security-level property or a day-level regime?

`probe_sparse_factor_coverage_in_panels.py` refuted both a new-entrant explanation (missing-rate
enrichment 0.94-1.02 for all six) and a board explanation (0.47-0.54 on every board) -- the six are
missing about half the time for every kind of security equally. But their DAILY coverage has a standard
deviation of 0.144-0.229, which is 16-26x what independent per-security missingness at p=0.5 would
produce over ~3,300 rows a day. That gap is the whole question: if missingness were a security
attribute the daily mean would be pinned near 0.5, so the variance has to be coming from days.

This measures the day-level structure directly: the distribution of daily coverage, how much of each
factor's total missingness sits in its worst days, and the correlation between the six factors' daily
coverage series -- which is what §AN.4's two clusters (within-cluster row-count correlation
0.9992-1.0000, across-cluster 0.266-0.297) turned out to be, but measured on coverage rather than on
row counts and over all 969 days rather than a sample.

Read-only. No database, no weights, no model, no artifact writes.
"""
from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
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
SPARSE = ["yeslab_372", "yeslab_375", "yeslab_384", "yeslab_389", "yeslab_390", "yeslab_406"]
DENSE = [n for n in FACTOR_BY_INDEX if n not in SPARSE]
COL_OF = {n: f"f{i:02d}" for i, n in enumerate(FACTOR_BY_INDEX)}
CLUSTER_A = ["yeslab_372", "yeslab_389", "yeslab_390"]
CLUSTER_B = ["yeslab_375", "yeslab_384", "yeslab_406"]


def paths(artifacts: Path, ds: str, split: str) -> list[str]:
    for ws in WORKSPACES:
        p = sorted(glob.glob(str(artifacts / ws / "datasets" / ds / f"split={split}" / "*" / "*.parquet")))
        if p:
            return p
    raise FileNotFoundError(f"{ds} split={split}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=ART)
    ap.add_argument("--out", type=Path, default=ART / "sparse_coverage")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cols = [COL_OF[n] for n in FACTOR_BY_INDEX]
    frames = []
    for fold, (ds, year) in FOLD_DATASETS.items():
        te = pd.concat(
            [pd.read_parquet(p, columns=["signal_date", "label_valid", "raw_return_5d"] + cols)
             for p in paths(args.artifacts, ds, "test")],
            ignore_index=True,
        )
        te["signal_date"] = pd.to_datetime(te["signal_date"]).dt.date
        te = te.loc[
            te["label_valid"].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(te["raw_return_5d"], errors="coerce"))
        ]
        frames.append(te)
        print(f"  {fold} ({year}): {len(te)} scoreable rows", flush=True)
    panel = pd.concat(frames, ignore_index=True)

    present = pd.DataFrame(
        {n: pd.to_numeric(panel[COL_OF[n]], errors="coerce").notna() for n in FACTOR_BY_INDEX}
    )
    present["_day"] = panel["signal_date"].to_numpy()
    daily = present.groupby("_day").mean().sort_index()
    n_per_day = present.groupby("_day").size()

    result = {
        "question": (
            "Is the six sparse factors' missingness a security-level property or a day-level regime, and "
            "what is §AN.4's two-cluster structure when measured as daily coverage over all 969 days?"
        ),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True, "database_access": False, "weights_loaded": False,
        "days": int(len(daily)),
        "scoreable_rows": int(len(panel)),
        "scoreable_rows_per_day": {"min": int(n_per_day.min()), "median": float(n_per_day.median()),
                                   "max": int(n_per_day.max())},
        "factors": {},
    }

    for name in FACTOR_BY_INDEX:
        s = daily[name]
        p_mean = float(s.mean())
        n_med = float(n_per_day.median())
        binomial_std = float(np.sqrt(max(p_mean * (1 - p_mean), 0.0) / n_med))
        worst = s.nsmallest(24)
        result["factors"][name] = {
            "is_candidate_only": name in SPARSE,
            "daily_coverage": {
                "mean": p_mean, "std": float(s.std(ddof=1)), "min": float(s.min()),
                "p05": float(s.quantile(0.05)), "p25": float(s.quantile(0.25)),
                "median": float(s.median()), "p75": float(s.quantile(0.75)),
                "p95": float(s.quantile(0.95)), "max": float(s.max()),
            },
            "variance_vs_independent_per_security_missingness": {
                "observed_daily_std": float(s.std(ddof=1)),
                "std_if_missingness_were_independent_across_securities": binomial_std,
                "ratio": float(s.std(ddof=1) / binomial_std) if binomial_std > 0 else None,
                "reading": (
                    "a ratio near 1 means each security is missing on its own schedule; a large ratio "
                    "means whole days are missing together"
                ),
            },
            "concentration_of_missingness_in_the_worst_days": {
                "share_of_all_days": 24 / len(s),
                "share_of_all_missing_row_days_they_carry": float(
                    (1.0 - worst).sum() / (1.0 - s).sum()
                ) if float((1.0 - s).sum()) > 0 else None,
                "their_coverage_range": [float(worst.min()), float(worst.max())],
            },
            "days_below_half_coverage": int((s < 0.5).sum()),
            "days_below_a_quarter_coverage": int((s < 0.25).sum()),
            "days_above_90pct_coverage": int((s > 0.90).sum()),
        }

    corr = daily[SPARSE].corr(method="pearson")
    result["daily_coverage_correlation_between_the_six"] = {
        a: {b: float(corr.loc[a, b]) for b in SPARSE} for a in SPARSE
    }
    def block(x, y):
        v = [float(corr.loc[a, b]) for a in x for b in y if a != b]
        return {"min": min(v), "median": float(median(v)), "max": max(v)}
    result["cluster_structure_on_daily_coverage"] = {
        "cluster_A": CLUSTER_A, "cluster_B": CLUSTER_B,
        "within_A": block(CLUSTER_A, CLUSTER_A),
        "within_B": block(CLUSTER_B, CLUSTER_B),
        "across": block(CLUSTER_A, CLUSTER_B),
        "an4_reported_on_row_counts": {"within_A": "0.9992-1.0000", "within_B": "0.9991-0.9995",
                                       "across": "0.266-0.297"},
    }
    # do the two clusters switch on and off together?
    on_a = (daily[CLUSTER_A].mean(axis=1) > 0.75).to_numpy()
    on_b = (daily[CLUSTER_B].mean(axis=1) > 0.75).to_numpy()
    result["regime_switching"] = {
        "definition": "a cluster is 'on' for a day when its three factors' mean daily coverage > 0.75",
        "days_A_on": int(on_a.sum()), "days_B_on": int(on_b.sum()),
        "days_both_on": int((on_a & on_b).sum()), "days_neither_on": int((~on_a & ~on_b).sum()),
        "days_exactly_one_on": int((on_a ^ on_b).sum()),
        "days_total": int(len(daily)),
    }

    out = args.out / "sparse_availability_regime.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    print(f"\n{'factor':28} {'mean':>7} {'std':>7} {'min':>7} {'med':>7} {'max':>7} "
          f"{'ratio':>7} {'<0.5':>5} {'<0.25':>6} {'>0.9':>5}")
    for name in FACTOR_BY_INDEX:
        f = result["factors"][name]
        c = f["daily_coverage"]
        r = f["variance_vs_independent_per_security_missingness"]["ratio"]
        tag = "*" if name in SPARSE else " "
        print(f"{tag}{name:27} {c['mean']:7.4f} {c['std']:7.4f} {c['min']:7.4f} {c['median']:7.4f} "
              f"{c['max']:7.4f} {r if r is None else round(r,1):>7} "
              f"{f['days_below_half_coverage']:5} {f['days_below_a_quarter_coverage']:6} "
              f"{f['days_above_90pct_coverage']:5}")
    print("\n(* = candidate_only)")
    print(json.dumps(result["cluster_structure_on_daily_coverage"], ensure_ascii=False, indent=1))
    print(json.dumps(result["regime_switching"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
