#!/usr/bin/env python3
"""Do the two levers that worked STACK, and is the drawdown win one lucky episode?

`optimization_levers.py` found two retraining-free levers that each help:
  * turnover -- a wider holding buffer (N_exit 30 -> 150) and trailing score smoothing
    both raise net return AND lower drawdown, because cost is ~28% of gross return;
  * exposure -- a trailing 20-day trend filter on the style index cuts max drawdown
    from -25.2% to -8.0% and lifts Sharpe from 0.960 to 1.357, for ~3pp of annual return.

Three questions this script answers before either is recommended:

  1. BOUNDARY -- MA20 beat MA60 and MA120 at all three indices. Is the optimum interior,
     or does Sharpe keep rising as the window shrinks towards noise? Tests MA5..MA60.
  2. STACK -- grid the turnover lever against the exposure lever. If they are independent
     the combination should keep most of both; if they overlap it will not.
  3. SINGLE-EPISE -- the max drawdown window is 2023-07-26 -> 2024-01-29, the small-cap
     liquidity crash. Recompute every variant with those cycles EXCLUDED: if the whole
     benefit disappears, the filter is one saved episode, not a repeatable edge.

Also restates the tail-shape statistic conditional on days the basket actually made money,
which is the only subset where a "top-3 share of basket return" is interpretable.

Read-only over the signed artifacts plus index queries; writes only --out.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimization_levers import (  # noqa: E402
    INDICES,
    apply_exposure,
    cycles_with_days,
    load_index_closes,
    trend_exposure,
)
from optimization_probes import FOLD_YEAR, FOLDS, build_portfolio, ensemble_frames, metrics  # noqa: E402
from run_backtest import CYCLES_PER_YEAR, HOLDING_DAYS  # noqa: E402

BP = 15.0


def smoothed(ens: dict[str, pd.DataFrame], k: int) -> dict[str, pd.DataFrame]:
    """Trailing mean of the last k observed scores per name (k=1 is the raw score)."""
    if k <= 1:
        return ens
    out: dict[str, pd.DataFrame] = {}
    for fold, frame in ens.items():
        wide = frame.pivot(index="signal_date", columns="ts_code", values="score").sort_index()
        wide = wide.rolling(k, min_periods=1).mean()
        long = wide.stack().reset_index()
        long.columns = ["signal_date", "ts_code", "score"]
        out[fold] = long
    return out


def make_cycles(ens: dict[str, pd.DataFrame], labs: dict[str, pd.DataFrame],
                n_enter: int, n_exit: int) -> list[dict[str, Any]]:
    """Per-fold cycle grouping (192 = 4 x 48); a cycle never straddles a fold boundary."""
    cycles: list[dict[str, Any]] = []
    for fold in FOLDS:
        cycles.extend(cycles_with_days(build_portfolio(ens[fold], labs[fold], n_enter, n_exit)))
    return cycles


def yearly(cycles: list[dict[str, Any]], bp: float) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for fold, year in FOLD_YEAR.items():
        sub = [c for c in cycles if int(str(c["start"])[:4]) == year]
        if sub:
            m = metrics(sub, bp)
            out[str(year)] = {"net_annualized": m["net_annualized"], "net_sharpe": m["net_sharpe"],
                              "max_drawdown": m["max_drawdown"], "cycles": m["cycles"]}
    return out


def excluding(cycles: list[dict[str, Any]], start: date, end: date) -> list[dict[str, Any]]:
    """Drop every cycle that overlaps [start, end] -- the max-drawdown episode."""
    return [c for c in cycles
            if not (c["start"] <= end and c["end"] >= start)]


def scaled_cycles(cycles: list[dict[str, Any]], exposure: dict[date, float], floor: float) -> list[dict[str, Any]]:
    """The exposure-scaled cycle list itself, so it can be sliced by year or by window."""
    out: list[dict[str, Any]] = []
    prev: float | None = None
    for cycle in cycles:
        scales = [exposure.get(d, 1.0) for d in cycle.get("days", [cycle["start"]])]
        scale = floor + (1.0 - floor) * float(np.mean(scales))
        switch = abs(scale - prev) if prev is not None else 0.0
        prev = scale
        out.append({**cycle, "ret": cycle["ret"] * scale,
                    "turnover": cycle["turnover"] * scale + switch})
    return out


def tail_shape_fixed(ens: dict[str, pd.DataFrame], labs: dict[str, pd.DataFrame],
                     n_enter: int, n_exit: int) -> dict[str, Any]:
    """Basket-return concentration, split by whether the day's basket made money."""
    pos: list[dict[str, float]] = []
    neg: list[dict[str, float]] = []
    for fold in FOLDS:
        frame = ens[fold].merge(labs[fold], on=["signal_date", "ts_code"], how="inner")
        frame = frame.loc[frame["label_valid"] & np.isfinite(frame["raw_return_5d"])]
        for _, group in frame.groupby("signal_date", sort=True):
            top = group.nlargest(n_enter, "score")["raw_return_5d"].astype(float).to_numpy()
            if len(top) < 10:
                continue
            ordered = np.sort(top)[::-1]
            total = float(ordered.sum())
            row = {"mean": float(ordered.mean()), "median": float(np.median(ordered)),
                   "top3_share": float(ordered[:3].sum() / total),
                   "top5_share": float(ordered[:5].sum() / total),
                   "max_name": float(ordered[0]),
                   "share_gt_9pct": float((ordered > 0.09).mean()),
                   "share_negative": float((ordered < 0).mean())}
            (pos if total > 0 else neg).append(row)

    def summarise(rows: list[dict[str, float]]) -> dict[str, float]:
        if not rows:
            return {"days": 0}
        f = pd.DataFrame(rows)
        return {"days": int(len(f)),
                "mean_top3_share": float(f["top3_share"].mean()),
                "mean_top5_share": float(f["top5_share"].mean()),
                "median_of_daily_mean_pp": float(f["mean"].median() * 100),
                "median_of_daily_median_pp": float(f["median"].median() * 100),
                "mean_daily_best_name_pp": float(f["max_name"].mean() * 100),
                "mean_share_names_above_9pct": float(f["share_gt_9pct"].mean()),
                "mean_share_names_negative": float(f["share_negative"].mean())}

    return {"positive_days": summarise(pos), "negative_days": summarise(neg),
            "share_of_days_positive": len(pos) / max(1, len(pos) + len(neg)),
            "note": ("top3_share is only interpretable when the basket total is positive, so it "
                     "is reported separately for winning and losing days; averaging it over both "
                     "(as the first pass did) produces a meaningless negative number."),
            }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bp", type=float, default=BP)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "optimization_stack_results.json")
    args = parser.parse_args()

    ens, labs = ensemble_frames(Path("/data/zhangyudong/qrv2_research_artifacts"))
    closes = load_index_closes()
    result: dict[str, Any] = {"config": {
        "round_trip_bp": args.bp, "cycles_per_year": CYCLES_PER_YEAR,
        "cycle_grouping": "per fold, 192 = 4 x 48", "source": "dfs://YesLab index_daily_data",
    }}

    # ---- 1. boundary check on the trend window
    print("[1/4] trend-window boundary check ...", flush=True)
    base_cycles = make_cycles(ens, labs, 30, 60)
    base = metrics(base_cycles, args.bp)
    boundary: dict[str, Any] = {"none": {**base, "in_market": 1.0}}
    print(f"      {'no filter':<26} 年化 {base['net_annualized']*100:>6.2f}%  Sharpe {base['net_sharpe']:.3f}  "
          f"回撤 {base['max_drawdown']*100:>6.2f}%", flush=True)
    for code in ("932000.CSI", "000985.CSI"):
        for ma in (2, 3, 5, 10, 20, 30, 40, 60):
            exp = trend_exposure(closes[code], ma)
            m = apply_exposure(base_cycles, exp, args.bp, 0.0)
            m["in_market"] = float(np.mean([exp.get(c["start"], 1.0) for c in base_cycles]))
            # how often the filter flips, i.e. how much of the cost is the overlay itself
            seq = [exp.get(c["start"], 1.0) for c in base_cycles]
            m["flips_per_year"] = float(sum(1 for a, b in zip(seq, seq[1:]) if abs(a - b) > 1e-9)
                                        / len(seq) * CYCLES_PER_YEAR)
            boundary[f"{code}_ma{ma}"] = m
            print(f"      {INDICES[code]:<12} MA{ma:<4} floor0.0  年化 {m['net_annualized']*100:>6.2f}%  "
                  f"Sharpe {m['net_sharpe']:.3f}  回撤 {m['max_drawdown']*100:>6.2f}%  "
                  f"在场 {m['in_market']*100:>5.1f}%  换手 {m['mean_turnover']:.3f}  "
                  f"切换 {m['flips_per_year']:>4.1f}/年", flush=True)
    result["boundary"] = boundary

    # ---- 2. stack grid
    print("[2/4] stacking the turnover lever against the exposure lever ...", flush=True)
    overlays = {
        "none": None,
        "csi2000_ma5_f0.0": ("932000.CSI", 5, 0.0),
        "csi2000_ma10_f0.0": ("932000.CSI", 10, 0.0),
        "csi2000_ma20_f0.0": ("932000.CSI", 20, 0.0),
        "csi2000_ma20_f0.5": ("932000.CSI", 20, 0.5),
        "csiall_ma30_f0.0": ("000985.CSI", 30, 0.0),
    }
    expo_cache: dict[str, dict[date, float] | None] = {}
    for tag, spec in overlays.items():
        expo_cache[tag] = None if spec is None else trend_exposure(closes[spec[0]], spec[1])
    grid: dict[str, Any] = {}
    for n_enter in (20, 30, 50):
        for n_exit, smooth in ((60, 1), (90, 1), (150, 1), (90, 2)):
            cyc = make_cycles(smoothed(ens, smooth), labs, n_enter, n_exit)
            for tag, spec in overlays.items():
                scaled = cyc if spec is None else scaled_cycles(cyc, expo_cache[tag], spec[2])
                m = metrics(scaled, args.bp)
                key = f"top{n_enter}_exit{n_exit}_sm{smooth}_{tag}"
                grid[key] = {"metrics": m, "yearly": yearly(scaled, args.bp),
                             "n_enter": n_enter, "n_exit": n_exit, "smooth_k": smooth, "overlay": tag}
            best = grid[f"top{n_enter}_exit{n_exit}_sm{smooth}_none"]["metrics"]
            print(f"      top{n_enter:<3} exit{n_exit:<4} sm{smooth}  无过滤: 年化 {best['net_annualized']*100:>6.2f}%  "
                  f"Sharpe {best['net_sharpe']:.3f}  回撤 {best['max_drawdown']*100:>6.2f}%  "
                  f"换手 {best['mean_turnover']:.3f}", flush=True)
            for tag in overlays:
                if tag == "none":
                    continue
                m = grid[f"top{n_enter}_exit{n_exit}_sm{smooth}_{tag}"]["metrics"]
                print(f"                       +{tag:<20} 年化 {m['net_annualized']*100:>6.2f}%  "
                      f"Sharpe {m['net_sharpe']:.3f}  回撤 {m['max_drawdown']*100:>6.2f}%  "
                      f"累计 {m['total_return']*100:>6.2f}%", flush=True)
    result["stack_grid"] = grid

    # ---- 3. single-episode test
    print("[3/4] excluding the max-drawdown episode 2023-07-26 .. 2024-01-29 ...", flush=True)
    start, end = date(2023, 7, 26), date(2024, 1, 29)
    episode: dict[str, Any] = {"window": [str(start), str(end)]}
    for tag, spec in overlays.items():
        for label, cyc in (("top30_exit60", base_cycles),
                           ("top30_exit150", make_cycles(ens, labs, 30, 150))):
            scaled = cyc if spec is None else scaled_cycles(cyc, expo_cache[tag], spec[2])
            keep = excluding(scaled, start, end)
            full = metrics(scaled, args.bp)
            trimmed = metrics(keep, args.bp)
            episode[f"{label}_{tag}"] = {
                "full": {k: full[k] for k in ("net_annualized", "net_sharpe", "max_drawdown", "cycles")},
                "ex_episode": {k: trimmed[k] for k in ("net_annualized", "net_sharpe", "max_drawdown", "cycles")},
                "cycles_dropped": full["cycles"] - trimmed["cycles"],
            }
            print(f"      {label:<13} {tag:<20} 全样本 Sharpe {full['net_sharpe']:.3f} 回撤 {full['max_drawdown']*100:>6.2f}%  "
                  f"| 剔除{full['cycles']-trimmed['cycles']:>2}周期后 Sharpe {trimmed['net_sharpe']:.3f} "
                  f"回撤 {trimmed['max_drawdown']*100:>6.2f}%", flush=True)
    result["episode_test"] = episode

    # ---- 4. tail shape, fixed
    print("[4/4] basket concentration on winning vs losing days ...", flush=True)
    result["tail_shape"] = tail_shape_fixed(ens, labs, 30, 60)
    t = result["tail_shape"]
    for side in ("positive_days", "negative_days"):
        s = t[side]
        print(f"      {side:<14} days {s['days']:>3}  top3占篮子 {s['mean_top3_share']*100:>6.1f}%  "
              f"top5 {s['mean_top5_share']*100:>6.1f}%  >9%个股 {s['mean_share_names_above_9pct']*100:>4.1f}%",
              flush=True)
    print(f"      篮子为正的天数占比 {t['share_of_days_positive']*100:.1f}%", flush=True)

    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
