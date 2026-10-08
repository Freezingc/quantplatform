#!/usr/bin/env python3
"""Measure every optimisation lever that does NOT need retraining, on frozen scores.

The owner asked: to raise return and cut max drawdown, should we change the factors,
the label, the model, or something else? `optimization_probes.py` already answered two
sub-questions (a holding buffer helps; volatility targeting does not). This script
measures the rest, so the recommendation rests on numbers rather than on priors.

Six measurements, all read-only over the signed TEST artifacts plus one index query:

  A. DRAWDOWN ATTRIBUTION -- over the exact peak-to-trough window the strategy actually
     suffered, what did each index do? Separates "the market fell" from "selection
     failed". Determines whether drawdown is an exposure problem or an alpha problem.
  B. EXPOSURE OVERLAY -- a trailing style-index trend filter (close vs its own MA, both
     known at the close of the signal date, trade at the next open: no lookahead), with
     the extra round-trip cost of going to cash and back charged explicitly.
  C. FACTOR DIAGNOSTICS -- per-factor cross-sectional Spearman IC and top-50 tail spread
     on the frozen TEST panels, with the f00..f14 -> yeslab_* mapping taken from the
     driver source. Descriptive of what the models were given, NOT a retraining result.
  D. ENSEMBLE VARIANTS -- equal-weight raw score (current), rank average, z-score
     average, and consensus counting. The models only agree on 8.7 of their top 50, so
     how they are combined is itself a lever.
  E. TURNOVER LEVERS -- buffer width sweep (N_exit) and trailing score smoothing, both
     of which cut the cost drag that currently eats ~28% of gross return.
  F. LOTTERY CHECK -- is the top-N return carried by a handful of extreme winners? If
     so the label is the binding constraint and any headline number is fragile.

Writes only --out.
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
from optimization_probes import (  # noqa: E402
    ARTIFACTS,
    BP,
    FOLD_YEAR,
    FOLDS,
    WORKSPACES,
    build_portfolio,
    ensemble_frames,
    metrics,
    to_cycles,
)
from run_backtest import CYCLES_PER_YEAR, HOLDING_DAYS, load_labels, load_predictions  # noqa: E402

SCRIPTS = "/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/scripts"

# f00..f14 -> factor id, from FROZEN_FACTOR_SET_V2 in walk_forward_factor_daily.py.
# The artifacts store only positional names, so this mapping is reproduced here.
FACTOR_MAP = {
    "f00": "yeslab_206", "f01": "yeslab_357", "f02": "yeslab_372", "f03": "yeslab_375",
    "f04": "yeslab_381", "f05": "yeslab_384", "f06": "yeslab_389", "f07": "yeslab_390",
    "f08": "yeslab_406", "f09": "yeslab_425", "f10": "yeslab_443_reg30_max20_v1",
    "f11": "yeslab_464", "f12": "yeslab_471", "f13": "yeslab_484", "f14": "yeslab_96",
}
DENSE = {"yeslab_206", "yeslab_357", "yeslab_381", "yeslab_425",
         "yeslab_443_reg30_max20_v1", "yeslab_464", "yeslab_471", "yeslab_484", "yeslab_96"}
INDICES = {
    "000001.SH": "上证综指",
    "000985.CSI": "中证全指",
    "000852.SH": "中证1000",
    "932000.CSI": "中证2000 (style match)",
}
TREND_INDEX = "932000.CSI"


# ---------------------------------------------------------------- A. attribution
def load_index_closes() -> dict[str, pd.Series]:
    """ts_code -> close indexed by trade_date, from 2021 so MA60 is warm at F1 start."""
    sys.path.insert(0, SCRIPTS)
    from walk_forward_common import new_holder, query

    holder = new_holder()
    codes = "', '".join(INDICES)
    _, frame = query(
        holder,
        "select ts_code, trade_date, open, close from "
        "loadTable('dfs://YesLab','index_daily_data') "
        f"where ts_code in ['{codes}'] and trade_date >= 2021.01.01 and trade_date <= 2026.03.31",
    )
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    for column in ("open", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    out: dict[str, pd.Series] = {}
    for code, group in frame.groupby("ts_code"):
        g = group.sort_values("trade_date").set_index("trade_date")
        out[str(code)] = g["close"]
    return out


def window_return(closes: pd.Series, start: date, end: date) -> float | None:
    """Close-to-close index return over [start, end] using the nearest trading days."""
    series = closes.loc[(closes.index >= start) & (closes.index <= end)]
    if len(series) < 2:
        return None
    return float(series.iloc[-1] / series.iloc[0] - 1.0)


# ---------------------------------------------------------------- B. exposure
def trend_exposure(closes: pd.Series, ma_window: int) -> dict[date, float]:
    """1.0 when close > its own trailing MA, else 0.0. Known at the close of the day."""
    ma = closes.rolling(ma_window, min_periods=ma_window).mean()
    signal = (closes > ma).astype(float)
    return {day: float(signal.loc[day]) for day in signal.index if not math.isnan(float(ma.loc[day]))}


def apply_exposure(cycles: list[dict[str, Any]], exposure: dict[date, float], bp: float,
                   floor: float) -> dict[str, float]:
    """Scale each cycle by its members' exposure; charge the cash round-trip explicitly."""
    out: list[dict[str, Any]] = []
    prev: float | None = None
    for cycle in cycles:
        days = cycle.get("days")
        scales = [exposure.get(d, 1.0) for d in days] if days else [1.0]
        scale = float(np.mean(scales))
        scale = floor + (1.0 - floor) * scale
        switch = 0.0
        if prev is not None and abs(scale - prev) > 1e-9:
            switch = abs(scale - prev)
        prev = scale
        out.append({
            "start": cycle["start"], "end": cycle["end"], "n": cycle["n"],
            "ret": cycle["ret"] * scale,
            "turnover": cycle["turnover"] * scale + switch,
        })
    return metrics(out, bp)


def cycles_with_days(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """to_cycles, but remembering the signal dates inside each cycle (for exposure)."""
    out: list[dict[str, Any]] = []
    for start in range(0, len(rows) - HOLDING_DAYS + 1, HOLDING_DAYS):
        chunk = rows[start:start + HOLDING_DAYS]
        out.append({
            "start": chunk[0]["signal_date"], "end": chunk[-1]["signal_date"],
            "ret": float(np.mean([c["ret"] for c in chunk])),
            "turnover": float(np.mean([c["turnover"] for c in chunk])),
            "n": float(np.mean([c["n"] for c in chunk])),
            "days": [c["signal_date"] for c in chunk],
        })
    return out


# ---------------------------------------------------------------- C. factors
def factor_diagnostics(artifacts: Path) -> dict[str, Any]:
    """Per-factor daily cross-sectional Spearman IC and top-50 tail spread on TEST.

    The IC MUST be computed per signal date and then averaged. Pooling every (date, stock)
    pair into one correlation mixes cross-sectional signal with cross-date drift in the
    factor and in the market, and yields a single meaningless number per fold -- that was
    the first version of this function, and it is why its output was discarded.
    """
    cache = load_predictions(artifacts / WORKSPACES["M0R"])
    dataset_ids: dict[str, str] = {}
    for manifest in sorted((artifacts / WORKSPACES["M0R"] / "predictions").glob("*/manifest.json")):
        meta = json.loads(manifest.read_text())["metadata"]
        if meta["kind"] == "candidate":
            dataset_ids[meta["fold_id"]] = meta["dataset_id"]
    factors = sorted(FACTOR_MAP)
    per_fold: dict[str, dict[str, dict[str, float]]] = {}
    nan_rate: dict[str, list[float]] = {f: [] for f in factors}
    pooled_ic: dict[str, list[float]] = {f: [] for f in factors}
    pooled_sp: dict[str, list[float]] = {f: [] for f in factors}
    for fold in FOLDS:
        shards = sorted(
            (artifacts / WORKSPACES["M0R"] / "datasets" / dataset_ids[fold] / "split=test").rglob("*.parquet")
        )
        ic_acc: dict[str, list[float]] = {f: [] for f in factors}
        sp_acc: dict[str, list[float]] = {f: [] for f in factors}
        for shard in shards:
            panel = pd.read_parquet(shard, columns=["signal_date", "ts_code", "raw_return_5d",
                                                    "label_valid", *factors])
            panel["signal_date"] = pd.to_datetime(panel["signal_date"]).dt.date
            panel = panel.loc[panel["label_valid"].fillna(False).astype(bool)].copy()
            panel["raw_return_5d"] = pd.to_numeric(panel["raw_return_5d"], errors="coerce")
            panel = panel.loc[np.isfinite(panel["raw_return_5d"])]
            for f in factors:
                panel[f] = pd.to_numeric(panel[f], errors="coerce")
                nan_rate[f].append(float(panel[f].isna().mean()))
            day = panel["signal_date"]
            ret_rank = panel.groupby("signal_date")["raw_return_5d"].rank()
            day_mean_ret = panel.groupby("signal_date")["raw_return_5d"].transform("mean")
            for f in factors:
                rank_f = panel.groupby("signal_date")[f].rank()
                tmp = pd.DataFrame({"d": day, "x": rank_f, "y": ret_rank}).dropna()
                if len(tmp):
                    tmp["xy"] = tmp["x"] * tmp["y"]
                    tmp["x2"] = tmp["x"] * tmp["x"]
                    tmp["y2"] = tmp["y"] * tmp["y"]
                    g = tmp.groupby("d").agg(n=("x", "size"), sx=("x", "sum"), sy=("y", "sum"),
                                             sxy=("xy", "sum"), sx2=("x2", "sum"), sy2=("y2", "sum"))
                    g = g.loc[g["n"] >= 100]
                    if len(g):
                        mx, my = g["sx"] / g["n"], g["sy"] / g["n"]
                        cov = g["sxy"] / g["n"] - mx * my
                        vx = (g["sx2"] / g["n"] - mx * mx).clip(lower=0)
                        vy = (g["sy2"] / g["n"] - my * my).clip(lower=0)
                        den = np.sqrt(vx * vy)
                        corr = (cov / den).loc[den > 0]
                        ic_acc[f].extend([float(v) for v in corr if math.isfinite(v)])
                # top-50 BY FACTOR VALUE within each date, minus that date's own mean
                top_rank = rank_f.groupby(day).transform("max")
                top = panel.loc[(rank_f > top_rank - 50) & rank_f.notna()]
                if len(top):
                    excess = top["raw_return_5d"] - top["signal_date"].map(day_mean_ret.groupby(day).mean())
                    per_day = pd.DataFrame({"d": top["signal_date"], "e": excess}).groupby("d")["e"].mean()
                    counts = top.groupby("signal_date").size()
                    sp_acc[f].extend([float(v) for v in per_day.loc[counts.reindex(per_day.index) >= 30]
                                      if math.isfinite(v)])
            del panel, rank_f, ret_rank, tmp, top
        per_fold[fold] = {}
        for f in factors:
            ic = np.asarray(ic_acc[f], dtype=float)
            sp = np.asarray(sp_acc[f], dtype=float)
            per_fold[fold][f] = {
                "ic_mean": float(ic.mean()) if len(ic) else float("nan"),
                "ic_std": float(ic.std(ddof=1)) if len(ic) > 1 else float("nan"),
                "ic_days": int(len(ic)),
                "ic_ir_daily": float(ic.mean() / ic.std(ddof=1)) if len(ic) > 2 and ic.std(ddof=1) > 0 else float("nan"),
                "ic_t_stat": float(ic.mean() / (ic.std(ddof=1) / math.sqrt(len(ic)))) if len(ic) > 2 and ic.std(ddof=1) > 0 else float("nan"),
                "ic_positive_share": float((ic > 0).mean()) if len(ic) else float("nan"),
                "top50_spread_pp": float(sp.mean() * 100) if len(sp) else float("nan"),
                "top50_spread_days": int(len(sp)),
            }
            pooled_ic[f].extend(ic_acc[f])
            pooled_sp[f].extend(sp_acc[f])
    allf = {}
    for f in factors:
        ic = np.asarray(pooled_ic[f], dtype=float)
        allf[f] = {
            "factor_id": FACTOR_MAP[f],
            "density": "dense" if FACTOR_MAP[f] in DENSE else "sparse",
            "nan_rate": float(np.mean(nan_rate[f])) if nan_rate[f] else float("nan"),
            "ic_days": int(len(ic)),
            "ic_mean": float(ic.mean()) if len(ic) else float("nan"),
            "ic_std": float(ic.std(ddof=1)) if len(ic) > 1 else float("nan"),
            "ic_ir_daily": float(ic.mean() / ic.std(ddof=1)) if len(ic) > 2 and ic.std(ddof=1) > 0 else float("nan"),
            "ic_t_stat": float(ic.mean() / (ic.std(ddof=1) / math.sqrt(len(ic)))) if len(ic) > 2 and ic.std(ddof=1) > 0 else float("nan"),
            "ic_positive_share": float((ic > 0).mean()) if len(ic) else float("nan"),
            "top50_spread_pp": float(np.mean(pooled_sp[f]) * 100) if pooled_sp[f] else float("nan"),
            "per_fold_ic": {k: per_fold[k][f]["ic_mean"] for k in FOLDS},
            "per_fold_top50_pp": {k: per_fold[k][f]["top50_spread_pp"] for k in FOLDS},
        }
    return {
        "factor_map": FACTOR_MAP,
        "note": ("Descriptive statistics of the frozen TEST panels the models were given, from the "
                 "M0R datasets. IC is the per-signal-date cross-sectional Spearman correlation with "
                 "raw_return_5d, averaged over dates; ic_ir_daily is mean/std across dates (NOT "
                 "annualised); ic_t_stat assumes independent days, which overlapping 5-day labels "
                 "violate, so it is an upper bound on significance. top50_spread is the mean over "
                 "dates of (mean return of that date's top-50 by factor value - that date's "
                 "cross-sectional mean), in percentage points. This is NOT a retraining result: a "
                 "factor with low standalone IC can still contribute inside a nonlinear model, and "
                 "a negative IC simply means the useful direction is short the factor."),
        "per_fold": per_fold,
        "all_folds": allf,
    }


# ---------------------------------------------------------------- D. ensembling
def ensemble_variants(ens: dict[str, pd.DataFrame], labs: dict[str, pd.DataFrame],
                      members: tuple[str, ...], n_top: int, bp: float) -> dict[str, Any]:
    """Combine the same three models four ways and backtest each at top-N."""
    raw: dict[str, pd.DataFrame] = {}
    for fold in FOLDS:
        joined = None
        for member in members:
            frame = ens[member][fold]
            joined = frame if joined is None else joined.merge(frame, on=["signal_date", "ts_code"])
        assert joined is not None
        raw[fold] = joined
    out: dict[str, Any] = {}
    for tag in ("score_mean", "rank_mean", "zscore_mean", "consensus"):
        cycles: list[dict[str, Any]] = []
        for fold in FOLDS:
            frame = raw[fold].copy()
            if tag == "score_mean":
                frame["score"] = frame[list(members)].mean(axis=1)
            elif tag == "rank_mean":
                frame["score"] = frame.groupby("signal_date")[list(members)].rank(pct=True).mean(axis=1)
            elif tag == "zscore_mean":
                def z(block: pd.DataFrame) -> pd.Series:
                    std = block.std(axis=1, ddof=0)
                    return ((block.T - block.mean(axis=1)) / std.replace(0, np.nan)).T.mean(axis=1)
                frame["score"] = z(frame[list(members)])
            else:
                frame["score"] = sum(
                    frame.groupby("signal_date")[m].rank(ascending=False, method="first") <= n_top
                    for m in members
                ).astype(float) + frame[list(members)].rank(pct=True).mean(axis=1) * 1e-6
            frame = frame.loc[:, ["signal_date", "ts_code", "score"]]
            cycles.extend(to_cycles(build_portfolio(frame, labs[fold], n_top, None)))
        out[tag] = metrics(cycles, bp)
    return out


def member_frames(artifacts: Path) -> dict[str, dict[str, pd.DataFrame]]:
    """model -> fold -> score frame, inner-joined across the three members."""
    cache = {name: load_predictions(artifacts / ws) for name, ws in WORKSPACES.items()}
    out: dict[str, dict[str, pd.DataFrame]] = {m: {} for m in WORKSPACES}
    for fold in FOLDS:
        keys = None
        for member in WORKSPACES:
            frame = cache[member][(fold, "candidate")][0]
            piece = frame.loc[:, ["signal_date", "ts_code"]]
            keys = piece if keys is None else keys.merge(piece, on=["signal_date", "ts_code"])
        for member in WORKSPACES:
            frame = cache[member][(fold, "candidate")][0].loc[:, ["signal_date", "ts_code", "score"]]
            frame = frame.rename(columns={"score": member})
            out[member][fold] = frame.merge(keys, on=["signal_date", "ts_code"])
    return out


# ---------------------------------------------------------------- E. turnover
def buffer_sweep(ens: dict[str, pd.DataFrame], labs: dict[str, pd.DataFrame],
                 n_enter: int, exits: tuple[int, ...], bp: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for n_exit in exits:
        cycles: list[dict[str, Any]] = []
        for fold in FOLDS:
            cycles.extend(to_cycles(build_portfolio(ens[fold], labs[fold], n_enter,
                                                    None if n_exit == n_enter else n_exit)))
        out[str(n_exit)] = metrics(cycles, bp)
    return out


def smoothing_sweep(ens: dict[str, pd.DataFrame], labs: dict[str, pd.DataFrame],
                    n_enter: int, n_exit: int, windows: tuple[int, ...], bp: float) -> dict[str, Any]:
    """Trailing average of the last k observed scores, then the same buffered top-N."""
    out: dict[str, Any] = {}
    for k in windows:
        cycles: list[dict[str, Any]] = []
        for fold in FOLDS:
            frame = ens[fold].copy()
            if k > 1:
                wide = frame.pivot(index="signal_date", columns="ts_code", values="score").sort_index()
                wide = wide.rolling(k, min_periods=1).mean()
                frame = wide.stack().reset_index()
                frame.columns = ["signal_date", "ts_code", "score"]
            cycles.extend(to_cycles(build_portfolio(frame, labs[fold], n_enter, n_exit)))
        out[str(k)] = metrics(cycles, bp)
    return out


# ---------------------------------------------------------------- F. lottery
def tail_shape(ens: dict[str, pd.DataFrame], labs: dict[str, pd.DataFrame], n_enter: int,
               n_exit: int) -> dict[str, Any]:
    """How concentrated is the top-N return inside each day's own basket?"""
    stats: list[dict[str, float]] = []
    for fold in FOLDS:
        frame = ens[fold].merge(labs[fold], on=["signal_date", "ts_code"], how="inner")
        frame = frame.loc[frame["label_valid"] & np.isfinite(frame["raw_return_5d"])]
        for _, group in frame.groupby("signal_date", sort=True):
            top = group.nlargest(n_enter, "score")["raw_return_5d"].astype(float).to_numpy()
            if len(top) < 10:
                continue
            ordered = np.sort(top)[::-1]
            total = float(ordered.sum())
            stats.append({
                "mean": float(ordered.mean()),
                "median": float(np.median(ordered)),
                "top3_share": float(ordered[:3].sum() / total) if abs(total) > 1e-12 else float("nan"),
                "top5_share": float(ordered[:5].sum() / total) if abs(total) > 1e-12 else float("nan"),
                "max_name": float(ordered[0]),
                "share_gt_9pct": float((ordered > 0.09).mean()),
                "share_negative": float((ordered < 0).mean()),
            })
    frame = pd.DataFrame(stats).dropna()
    return {
        "days": int(len(frame)),
        "mean_of_daily_basket_mean_pp": float(frame["mean"].mean() * 100),
        "mean_of_daily_basket_median_pp": float(frame["median"].mean() * 100),
        "mean_top3_share_of_basket_return": float(frame["top3_share"].mean()),
        "mean_top5_share_of_basket_return": float(frame["top5_share"].mean()),
        "mean_daily_best_name_pp": float(frame["max_name"].mean() * 100),
        "mean_share_of_names_above_9pct": float(frame["share_gt_9pct"].mean()),
        "mean_share_of_names_negative": float(frame["share_negative"].mean()),
        "note": ("top3_share is the fraction of the day's basket total return produced by its "
                 "three best names; it is only interpretable on days the basket made money, "
                 "so the mean is reported alongside the share of negative days."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--n-enter", type=int, default=30)
    parser.add_argument("--n-exit", type=int, default=60)
    parser.add_argument("--bp", type=float, default=BP)
    parser.add_argument("--ma", type=int, default=60)
    parser.add_argument("--skip-factors", action="store_true")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "optimization_levers_results.json")
    args = parser.parse_args()

    ens, labs = ensemble_frames(args.artifacts)
    result: dict[str, Any] = {"config": {
        "n_enter": args.n_enter, "n_exit": args.n_exit, "round_trip_bp": args.bp,
        "trend_index": TREND_INDEX, "ma_window": args.ma, "cycles_per_year": CYCLES_PER_YEAR,
        "ensemble_members": list(WORKSPACES), "source": "dfs://YesLab index_daily_data",
    }}

    print("[1/6] buffered portfolio cycles ...", flush=True)
    # Grouped PER FOLD, exactly as run_backtest.py does (192 = 4 folds x 48 cycles): a
    # 5-day cycle may not straddle a fold boundary, because the model that produced the
    # scores changes there. Concatenating the daily rows first silently merges the last
    # days of one fold with the first of the next and moves every headline number.
    cycles: list[dict[str, Any]] = []
    for fold in FOLDS:
        cycles.extend(cycles_with_days(build_portfolio(ens[fold], labs[fold], args.n_enter, args.n_exit)))
    result["reference"] = {"buffered_top30": metrics(cycles, args.bp)}
    ref = result["reference"]["buffered_top30"]
    print(f"      年化 {ref['net_annualized']*100:.2f}%  Sharpe {ref['net_sharpe']:.3f}  "
          f"回撤 {ref['max_drawdown']*100:.2f}%  窗口 {ref['dd_peak_date']}→{ref['dd_trough_date']}", flush=True)

    print("[2/6] index closes + drawdown attribution ...", flush=True)
    closes = load_index_closes()
    start = date.fromisoformat(ref["dd_peak_date"])
    end = date.fromisoformat(ref["dd_trough_date"])
    strat = float(np.prod([1.0 + c["ret"] - c["turnover"] * args.bp / 10_000
                           for c in cycles if start <= c["start"] <= end]) - 1.0)
    attribution = {"window": [str(start), str(end)], "strategy_net": strat, "indices": {}}
    for code, label in INDICES.items():
        r = window_return(closes[code], start, end)
        attribution["indices"][code] = {"label": label, "close_to_close": r,
                                        "strategy_minus_index_pp": None if r is None else (strat - r) * 100}
    result["drawdown_attribution"] = attribution
    for code, entry in attribution["indices"].items():
        print(f"      {entry['label']:<24} {entry['close_to_close']*100:>7.2f}%   "
              f"策略-指数 {entry['strategy_minus_index_pp']:>7.2f}pp", flush=True)

    print("[3/6] trend-filter exposure overlay ...", flush=True)
    result["exposure_overlay"] = {}
    for code in ("932000.CSI", "000985.CSI", "000001.SH"):
        for ma in (20, 60, 120):
            exposure = trend_exposure(closes[code], ma)
            in_market = float(np.mean([exposure.get(c["start"], 1.0) for c in cycles]))
            for floor in (0.0, 0.5):
                m = apply_exposure(cycles, exposure, args.bp, floor)
                m["share_of_cycles_in_market"] = in_market
                key = f"{code}_ma{ma}_floor{floor:.1f}"
                result["exposure_overlay"][key] = m
                print(f"      {INDICES[code]:<20} MA{ma:<4} floor={floor:.1f}  "
                      f"年化 {m['net_annualized']*100:>6.2f}%  Sharpe {m['net_sharpe']:.3f}  "
                      f"回撤 {m['max_drawdown']*100:>6.2f}%  累计 {m['total_return']*100:>6.2f}%  "
                      f"在场 {in_market*100:>5.1f}%", flush=True)

    print("[4/6] buffer width + score smoothing ...", flush=True)
    result["buffer_sweep"] = buffer_sweep(ens, labs, args.n_enter, (30, 45, 60, 90, 150), args.bp)
    for k, v in result["buffer_sweep"].items():
        print(f"      N_exit={k:>3}  年化 {v['net_annualized']*100:>6.2f}%  Sharpe {v['net_sharpe']:.3f}  "
              f"回撤 {v['max_drawdown']*100:>6.2f}%  换手 {v['mean_turnover']:.3f}", flush=True)
    result["smoothing_sweep"] = smoothing_sweep(ens, labs, args.n_enter, args.n_exit, (1, 2, 3, 5), args.bp)
    for k, v in result["smoothing_sweep"].items():
        print(f"      smooth_k={k}  年化 {v['net_annualized']*100:>6.2f}%  Sharpe {v['net_sharpe']:.3f}  "
              f"回撤 {v['max_drawdown']*100:>6.2f}%  换手 {v['mean_turnover']:.3f}", flush=True)

    print("[5/6] ensemble combination variants + tail shape ...", flush=True)
    members = member_frames(args.artifacts)
    for n_top in (30, 100):
        result.setdefault("ensemble_variants", {})[str(n_top)] = ensemble_variants(
            members, labs, tuple(WORKSPACES), n_top, args.bp)
        for tag, v in result["ensemble_variants"][str(n_top)].items():
            print(f"      top{n_top:<3} {tag:<13} 年化 {v['net_annualized']*100:>6.2f}%  "
                  f"Sharpe {v['net_sharpe']:.3f}  回撤 {v['max_drawdown']*100:>6.2f}%  "
                  f"换手 {v['mean_turnover']:.3f}", flush=True)
    result["tail_shape"] = tail_shape(ens, labs, args.n_enter, args.n_exit)
    t = result["tail_shape"]
    print(f"      top-3 占当日篮子收益 {t['mean_top3_share_of_basket_return']*100:.1f}%  "
          f">9%的个股占比 {t['mean_share_of_names_above_9pct']*100:.1f}%  "
          f"负收益占比 {t['mean_share_of_names_negative']*100:.1f}%", flush=True)

    if not args.skip_factors:
        print("[6/6] per-factor diagnostics on the frozen TEST panels ...", flush=True)
        result["factor_diagnostics"] = factor_diagnostics(args.artifacts)
        for f, v in sorted(result["factor_diagnostics"]["all_folds"].items(),
                           key=lambda kv: -abs(kv[1]["ic_mean"]) if math.isfinite(kv[1]["ic_mean"]) else 0):
            print(f"      {f}={FACTOR_MAP[f]:<26} {v['density']:<6} IC {v['ic_mean']:>+.4f}  "
                  f"IR {v['ic_ir']:>+.3f}  top50 {v['top50_spread_pp']:>+6.2f}pp  NaN {v['nan_rate']*100:>5.1f}%",
                  flush=True)
    else:
        print("[6/6] skipped (--skip-factors)", flush=True)

    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
