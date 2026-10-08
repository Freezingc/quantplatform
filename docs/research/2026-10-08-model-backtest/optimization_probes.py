#!/usr/bin/env python3
"""Three cheap, retraining-free probes over the already-frozen scores.

The owner asked what to optimise for higher return and lower drawdown. Before
proposing factor or label or model changes -- all of which cost days of retraining --
these three questions can be answered from artifacts that already exist:

  1. WHERE is the max drawdown, and what did the market do over the same window?
     If the index fell as much, the drawdown is beta and no amount of better stock
     selection will remove it; the lever is exposure, not alpha.
  2. Does a holding BUFFER cut turnover without costing gross return? Turnover is
     0.42-0.68 per cycle, so cost is 28% of gross return; if the gross signal is
     insensitive to which of the top ~60 names you hold, a buffer is nearly free
     net return.
  3. Does VOLATILITY TARGETING cut the drawdown? Applied with trailing realised vol
     only, so it uses no future information.

Read-only over signed artifacts plus one index query; writes only --out.
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
from run_backtest import CYCLES_PER_YEAR, HOLDING_DAYS, load_labels, load_predictions  # noqa: E402

ARTIFACTS = Path("/data/zhangyudong/qrv2_research_artifacts")
WORKSPACES = {"M0R": "m0r_v2", "M1": "m1_full_v1", "M2": "m2_v1"}
FOLDS = ("F1", "F2", "F3", "F4")
FOLD_YEAR = {"F1": 2022, "F2": 2023, "F3": 2024, "F4": 2025}
BP = 15.0


def ensemble_frames(artifacts: Path) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Per fold: the ensemble score frame and its labels, on the intersection rows."""
    cache: dict[str, dict] = {}
    ens: dict[str, pd.DataFrame] = {}
    labs: dict[str, pd.DataFrame] = {}
    label_cache: dict[str, pd.DataFrame] = {}
    for fold in FOLDS:
        if WORKSPACES["M0R"] not in cache:
            for name, ws in WORKSPACES.items():
                cache[name] = load_predictions(artifacts / ws)
        joined = None
        for member in WORKSPACES:
            frame = cache[member][(fold, "candidate")][0]
            piece = frame.loc[:, ["signal_date", "ts_code", "score"]].rename(columns={"score": member})
            joined = piece if joined is None else joined.merge(piece, on=["signal_date", "ts_code"], how="inner")
        assert joined is not None
        joined["score"] = joined[list(WORKSPACES)].mean(axis=1)
        ens[fold] = joined.loc[:, ["signal_date", "ts_code", "score"]]
        dataset_id = None
        for manifest in sorted((artifacts / WORKSPACES["M0R"] / "predictions").glob("*/manifest.json")):
            meta = json.loads(manifest.read_text())["metadata"]
            if meta["fold_id"] == fold:
                dataset_id = meta["dataset_id"]
                break
        assert dataset_id is not None
        if dataset_id not in label_cache:
            label_cache[dataset_id] = load_labels(artifacts / WORKSPACES["M0R"] / "datasets", dataset_id)[0]
        labs[fold] = label_cache[dataset_id]
    return ens, labs


def build_portfolio(
    scores: pd.DataFrame, labels: pd.DataFrame, n_enter: int, n_exit: int | None
) -> list[dict[str, Any]]:
    """Daily equal-weight holdings, optionally with a holding buffer.

    ``n_exit`` is None for the plain rule (hold exactly the top ``n_enter``). Otherwise a
    name already held is kept until its rank falls below ``n_exit``, and the vacated slots
    are filled from the top of the entry list -- the standard buffer that cuts turnover.
    """
    frame = scores.merge(labels, on=["signal_date", "ts_code"], how="inner")
    frame = frame.loc[frame["label_valid"] & np.isfinite(frame["raw_return_5d"])]
    rows: list[dict[str, Any]] = []
    held: list[str] = []
    for day, group in frame.groupby("signal_date", sort=True):
        ranked = group.sort_values("score", ascending=False)
        names = ranked["ts_code"].astype(str).tolist()
        rets = dict(zip(ranked["ts_code"].astype(str), ranked["raw_return_5d"].astype(float)))
        if n_exit is None:
            new = names[:n_enter]
        else:
            keep = [c for c in held if c in names[:n_exit]]
            fill = [c for c in names[: max(n_enter, n_exit)] if c not in keep]
            new = (keep + fill)[:n_enter]
        if not new:
            continue
        turnover = 1.0 if not held else 1.0 - len(set(new) & set(held)) / len(new)
        rows.append(
            {
                "signal_date": day,
                "ret": float(np.mean([rets[c] for c in new])),
                "turnover": turnover,
                "n": len(new),
            }
        )
        held = new
    return rows


def to_cycles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group five consecutive signal dates into one non-overlapping cycle."""
    out: list[dict[str, Any]] = []
    for start in range(0, len(rows) - HOLDING_DAYS + 1, HOLDING_DAYS):
        chunk = rows[start:start + HOLDING_DAYS]
        out.append(
            {
                "start": chunk[0]["signal_date"],
                "end": chunk[-1]["signal_date"],
                "ret": float(np.mean([c["ret"] for c in chunk])),
                "turnover": float(np.mean([c["turnover"] for c in chunk])),
                "n": float(np.mean([c["n"] for c in chunk])),
            }
        )
    return out


def metrics(cycles: list[dict[str, Any]], bp: float, scale: np.ndarray | None = None) -> dict[str, float]:
    ret = np.asarray([c["ret"] for c in cycles], dtype=float)
    turn = np.asarray([c["turnover"] for c in cycles], dtype=float)
    net = ret - turn * (bp / 10_000.0)
    if scale is not None:
        net = net * scale
        turn = turn * scale
    equity = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    trough = int(np.argmin(dd))
    peak_idx = int(np.argmax(equity[: trough + 1])) if trough > 0 else 0
    mean, std = float(net.mean()), float(net.std(ddof=1))
    return {
        "cycles": int(len(net)),
        "gross_annualized": float(ret.mean()) * CYCLES_PER_YEAR,
        "net_annualized": mean * CYCLES_PER_YEAR,
        "net_sharpe": (mean / std * math.sqrt(CYCLES_PER_YEAR)) if std > 0 else float("nan"),
        "net_vol": std * math.sqrt(CYCLES_PER_YEAR),
        "total_return": float(equity[-1] - 1.0),
        "max_drawdown": float(dd.min()),
        "dd_peak_cycle": peak_idx,
        "dd_trough_cycle": trough,
        "dd_peak_date": str(cycles[peak_idx]["start"]),
        "dd_trough_date": str(cycles[trough]["end"]),
        "mean_turnover": float(turn.mean()),
        "mean_names": float(np.mean([c["n"] for c in cycles])),
    }


def vol_target_scale(cycles: list[dict[str, Any]], target_annual: float, lookback: int) -> np.ndarray:
    """Trailing realised vol of the strategy's own cycle returns; capped at 1 (no leverage)."""
    ret = np.asarray([c["ret"] for c in cycles], dtype=float)
    scale = np.ones(len(ret))
    for i in range(lookback, len(ret)):
        realised = float(ret[i - lookback:i].std(ddof=1)) * math.sqrt(CYCLES_PER_YEAR)
        if realised > 0:
            scale[i] = min(1.0, target_annual / realised)
    return scale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--n-enter", type=int, default=30)
    parser.add_argument("--n-exit", type=int, default=60)
    parser.add_argument("--bp", type=float, default=BP)
    parser.add_argument("--vol-target", type=float, default=0.15)
    parser.add_argument("--vol-lookback", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "optimization_probe_results.json")
    args = parser.parse_args()

    ens, labs = ensemble_frames(args.artifacts)
    all_cycles: dict[str, list[dict[str, Any]]] = {}
    per_fold: dict[str, Any] = {}
    for fold in FOLDS:
        plain = to_cycles(build_portfolio(ens[fold], labs[fold], args.n_enter, None))
        buffered = to_cycles(build_portfolio(ens[fold], labs[fold], args.n_enter, args.n_exit))
        per_fold[fold] = {"year": FOLD_YEAR[fold], "plain_cycles": len(plain), "buffered_cycles": len(buffered)}
        all_cycles.setdefault("plain", []).extend(plain)
        all_cycles.setdefault("buffered", []).extend(buffered)

    out: dict[str, Any] = {
        "config": {
            "n_enter": args.n_enter, "n_exit": args.n_exit, "round_trip_bp": args.bp,
            "vol_target": args.vol_target, "vol_lookback_cycles": args.vol_lookback,
            "cycles_per_year": CYCLES_PER_YEAR,
        },
        "per_fold": per_fold,
        "variants": {},
    }
    for tag in ("plain", "buffered"):
        cycles = all_cycles[tag]
        base = metrics(cycles, args.bp)
        out["variants"][tag] = {"base": base}
        for target in (0.12, 0.15, 0.18):
            scale = vol_target_scale(cycles, target, args.vol_lookback)
            out["variants"][tag][f"vol_target_{target:.2f}"] = metrics(cycles, args.bp, scale)
        yearly = {}
        for fold in FOLDS:
            sub = [c for c in cycles if FOLD_YEAR[fold] == int(str(c["start"])[:4])]
            if sub:
                yearly[str(FOLD_YEAR[fold])] = metrics(sub, args.bp)
        out["variants"][tag]["yearly"] = yearly

    args.out.write_text(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    for tag in ("plain", "buffered"):
        v = out["variants"][tag]
        b = v["base"]
        print(f"\n=== {tag} (N_enter={args.n_enter}" + (f", N_exit={args.n_exit})" if tag == "buffered" else ")"))
        print(f"  base        年化 {b['net_annualized']*100:>6.2f}%  Sharpe {b['net_sharpe']:.3f}  "
              f"回撤 {b['max_drawdown']*100:>6.2f}%  换手 {b['mean_turnover']:.3f}  持股 {b['mean_names']:.1f}  累计 {b['total_return']*100:>6.2f}%")
        print(f"  回撤窗口: {b['dd_peak_date']} → {b['dd_trough_date']}")
        for target in ("0.12", "0.15", "0.18"):
            m = v[f"vol_target_{target}"]
            print(f"  波动目标{target} 年化 {m['net_annualized']*100:>6.2f}%  Sharpe {m['net_sharpe']:.3f}  "
                  f"回撤 {m['max_drawdown']*100:>6.2f}%  累计 {m['total_return']*100:>6.2f}%")
        for y in ("2022", "2023", "2024", "2025"):
            m = v["yearly"].get(y)
            if m: print(f"    {y}: 年化 {m['net_annualized']*100:>6.2f}%  回撤 {m['max_drawdown']*100:>6.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
