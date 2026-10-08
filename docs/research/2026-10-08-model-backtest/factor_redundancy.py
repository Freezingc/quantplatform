#!/usr/bin/env python3
"""How many INDEPENDENT signals are in the frozen 15, and how few names carry the P&L?

The per-date IC measurement showed 14 of the 15 frozen factors have a NEGATIVE
cross-sectional IC against raw_return_5d, clustered between -0.046 and -0.092. If those
14 are the same reversal signal measured 14 ways, then "add more factors" is the wrong
recommendation and "add an ORTHOGONAL factor family" is the right one. Two measurements
settle it:

  1. FACTOR REDUNDANCY -- the mean cross-date average of the per-date cross-sectional
     Spearman correlation matrix among the 15 factors, plus the participation ratio
     (sum(lambda))^2 / sum(lambda^2) of that matrix, which is the effective number of
     independent directions. 15 means fully independent; 1 means one signal.
  2. IC CO-MOVEMENT -- the correlation among the 15 per-date IC time series. Factors can
     be uncorrelated in level and still fail at the same time; this is the version that
     matters for a model that combines them.
  3. P&L CONCENTRATION -- across the whole buffered top-30 backtest, what share of the
     total profit is produced by the best 10 / 100 / 1% of individual (day, name)
     positions? If a handful of positions carry the result, the strategy is fragile and
     any execution assumption that blocks the biggest winners (a limit-up filter, for
     instance) removes most of the return.

Read-only over the signed artifacts; writes only --out.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimization_levers import ARTIFACTS, DENSE, FACTOR_MAP, WORKSPACES  # noqa: E402
from optimization_probes import FOLDS, build_portfolio, ensemble_frames  # noqa: E402
from run_backtest import load_predictions  # noqa: E402

FACTORS = sorted(FACTOR_MAP)


def dataset_ids(artifacts: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for manifest in sorted((artifacts / WORKSPACES["M0R"] / "predictions").glob("*/manifest.json")):
        meta = json.loads(manifest.read_text())["metadata"]
        if meta["kind"] == "candidate":
            out[meta["fold_id"]] = meta["dataset_id"]
    return out


def redundancy(artifacts: Path) -> dict[str, Any]:
    """Mean per-date factor correlation matrix, effective dimension, and IC co-movement."""
    ids = dataset_ids(artifacts)
    corr_sum = np.zeros((len(FACTORS), len(FACTORS)))
    corr_weight = 0.0
    ic_series: dict[str, list[tuple[Any, float]]] = {f: [] for f in FACTORS}
    per_fold_corr: dict[str, list[float]] = {}
    for fold in FOLDS:
        shards = sorted((artifacts / WORKSPACES["M0R"] / "datasets" / ids[fold] / "split=test").rglob("*.parquet"))
        fold_offdiag: list[float] = []
        for shard in shards:
            panel = pd.read_parquet(shard, columns=["signal_date", "raw_return_5d", "label_valid", *FACTORS])
            panel["signal_date"] = pd.to_datetime(panel["signal_date"]).dt.date
            panel = panel.loc[panel["label_valid"].fillna(False).astype(bool)].copy()
            panel["raw_return_5d"] = pd.to_numeric(panel["raw_return_5d"], errors="coerce")
            for f in FACTORS:
                panel[f] = pd.to_numeric(panel[f], errors="coerce")
            panel = panel.loc[np.isfinite(panel["raw_return_5d"])]
            ret_rank = panel.groupby("signal_date")["raw_return_5d"].rank()
            for day, group in panel.groupby("signal_date", sort=True):
                ranked = group[FACTORS].rank()
                if len(ranked) < 100:
                    continue
                c = ranked.corr().to_numpy(dtype=float)
                if not np.isfinite(c).all():
                    continue
                corr_sum += c * len(ranked)
                corr_weight += len(ranked)
                iu = np.triu_indices(len(FACTORS), k=1)
                fold_offdiag.extend(np.abs(c[iu]).tolist())
                for j, f in enumerate(FACTORS):
                    col = ranked[f].to_numpy(dtype=float)
                    ok = np.isfinite(col)
                    if ok.sum() >= 100:
                        x, y = col[ok], ret_rank.loc[group.index].to_numpy(dtype=float)[ok]
                        ok2 = np.isfinite(y)
                        if ok2.sum() >= 100:
                            ic_series[f].append((day, float(np.corrcoef(x[ok2], y[ok2])[0, 1])))
            del panel, ranked
        per_fold_corr[fold] = [float(np.mean(fold_offdiag)), float(np.median(fold_offdiag)),
                               float(np.max(fold_offdiag)), len(fold_offdiag)]
    mean_corr = corr_sum / corr_weight
    eigen = np.linalg.eigvalsh(mean_corr)
    eigen = np.clip(eigen, 0, None)
    participation = float(eigen.sum() ** 2 / (eigen ** 2).sum())
    iu = np.triu_indices(len(FACTORS), k=1)
    offdiag = np.abs(mean_corr[iu])

    ic_frame = pd.DataFrame({f: pd.Series(dict(v)).sort_index() for f, v in ic_series.items()})
    ic_corr = ic_frame.corr().to_numpy(dtype=float)
    ic_off = np.abs(ic_corr[iu])
    ic_eigen = np.clip(np.linalg.eigvalsh(np.nan_to_num(ic_corr, nan=0.0)), 0, None)

    pairs = sorted(
        ((FACTOR_MAP[a], FACTOR_MAP[b], float(mean_corr[i, j]), float(ic_corr[i, j]))
         for i, a in enumerate(FACTORS) for j, b in enumerate(FACTORS) if i < j),
        key=lambda t: -abs(t[2]),
    )
    return {
        "weighted_by_rows": int(corr_weight),
        "mean_abs_factor_correlation": float(offdiag.mean()),
        "median_abs_factor_correlation": float(np.median(offdiag)),
        "max_abs_factor_correlation": float(offdiag.max()),
        "effective_dimensions_participation_ratio": participation,
        "eigenvalues_top5": [float(v) for v in np.sort(eigen)[::-1][:5]],
        "variance_explained_by_pc1": float(np.sort(eigen)[::-1][0] / eigen.sum()),
        "mean_abs_ic_series_correlation": float(np.nanmean(ic_off)),
        "ic_effective_dimensions": float(ic_eigen.sum() ** 2 / (ic_eigen ** 2).sum()),
        "ic_pc1_variance_share": float(np.sort(ic_eigen)[::-1][0] / ic_eigen.sum()),
        "per_fold_abs_offdiag_mean_median_max_n": per_fold_corr,
        "top10_pairs_by_abs_correlation": [
            {"a": a, "b": b, "level_corr": round(c, 4), "ic_series_corr": round(ic, 4)} for a, b, c, ic in pairs[:10]
        ],
        "note": ("level_corr is the cross-date average of the per-date cross-sectional Spearman "
                 "correlation between two factors; ic_series_corr is the correlation between their "
                 "per-date IC time series. participation ratio 15 = 15 independent signals, 1 = one."),
    }


def pnl_concentration(n_enter: int = 30, n_exit: int = 60) -> dict[str, Any]:
    """Share of total profit produced by the best few (day, name) positions."""
    artifacts = Path("/data/zhangyudong/qrv2_research_artifacts")
    ens, labs = ensemble_frames(artifacts)
    contributions: list[float] = []
    daily_totals: list[float] = []
    for fold in FOLDS:
        frame = ens[fold].merge(labs[fold], on=["signal_date", "ts_code"], how="inner")
        frame = frame.loc[frame["label_valid"] & np.isfinite(frame["raw_return_5d"])]
        held: list[str] = []
        for _, group in frame.groupby("signal_date", sort=True):
            ranked = group.sort_values("score", ascending=False)
            names = ranked["ts_code"].astype(str).tolist()
            rets = dict(zip(ranked["ts_code"].astype(str), ranked["raw_return_5d"].astype(float)))
            keep = [c for c in held if c in names[:n_exit]]
            fill = [c for c in names[: max(n_enter, n_exit)] if c not in keep]
            new = (keep + fill)[:n_enter]
            held = new
            per_name = [rets[c] / len(new) for c in new]
            contributions.extend(per_name)
            daily_totals.append(float(np.sum(per_name)))
    arr = np.sort(np.asarray(contributions, dtype=float))[::-1]
    total = float(arr.sum())
    positive = float(arr[arr > 0].sum())
    out = {
        "positions": int(len(arr)),
        "total_sum_of_daily_basket_returns": total,
        "sum_of_winning_positions": positive,
        "mean_daily_basket_return_pp": float(np.mean(daily_totals) * 100),
        "share_of_days_positive": float(np.mean([t > 0 for t in daily_totals])),
    }
    for k in (3, 10, 30, 100, 300, 1000):
        out[f"top{k}_positions_share_of_gross_profit"] = float(arr[:k].sum() / positive) if positive else None
    for frac in (0.001, 0.01, 0.05):
        k = max(1, int(len(arr) * frac))
        out[f"top{frac*100:g}pct_positions_share_of_gross_profit"] = float(arr[:k].sum() / positive) if positive else None
    out["largest_single_position_pp"] = float(arr[0] * 100)
    out["note"] = ("Each position contributes its 5-day return divided by that day's basket size. "
                   "Shares are of the sum of WINNING positions, so they read as 'what fraction of "
                   "all the profit these positions made comes from the best k of them'.")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "factor_redundancy_results.json")
    parser.add_argument("--skip-pnl", action="store_true")
    args = parser.parse_args()

    print("[1/2] factor redundancy + IC co-movement (969 dates x 15 factors) ...", flush=True)
    result: dict[str, Any] = {"redundancy": redundancy(ARTIFACTS)}
    r = result["redundancy"]
    print(f"      行数 {r['weighted_by_rows']}  平均|因子相关| {r['mean_abs_factor_correlation']:.3f}  "
          f"最大 {r['max_abs_factor_correlation']:.3f}", flush=True)
    print(f"      有效维度(参与率) {r['effective_dimensions_participation_ratio']:.2f} / 15   "
          f"PC1 解释 {r['variance_explained_by_pc1']*100:.1f}%", flush=True)
    print(f"      IC序列 平均|相关| {r['mean_abs_ic_series_correlation']:.3f}  "
          f"有效维度 {r['ic_effective_dimensions']:.2f} / 15  PC1 {r['ic_pc1_variance_share']*100:.1f}%", flush=True)
    print("      最冗余的5对:", flush=True)
    for pair in r["top10_pairs_by_abs_correlation"][:5]:
        print(f"        {pair['a']:<26} ~ {pair['b']:<26} level {pair['level_corr']:>+.3f}  "
              f"IC {pair['ic_series_corr']:>+.3f}", flush=True)

    if not args.skip_pnl:
        print("[2/2] P&L concentration over the whole buffered top-30 backtest ...", flush=True)
        result["pnl_concentration"] = pnl_concentration()
        p = result["pnl_concentration"]
        print(f"      持仓笔数 {p['positions']}  盈利日占比 {p['share_of_days_positive']*100:.1f}%", flush=True)
        for k in (3, 10, 30, 100, 300, 1000):
            v = p.get(f"top{k}_positions_share_of_gross_profit")
            if v is not None:
                print(f"      最好的 {k:>4} 笔占总毛利 {v*100:>6.2f}%", flush=True)
        print(f"      最好 0.1% 笔占 {p['top0.1pct_positions_share_of_gross_profit']*100:.2f}%   "
              f"最好 1% 笔占 {p['top1pct_positions_share_of_gross_profit']*100:.2f}%", flush=True)

    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
