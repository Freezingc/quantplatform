#!/usr/bin/env python3
"""What does dropping M2 cost, and what does it buy back?

Every backtest number published so far in this study is an equal-weight ensemble of
M0R + M1 + M2. But M2 FAILED gates G4 and G6 (G4 by 4.08e-05, G6 = 905/969 = 0.9340
against a 0.95 floor) and the registry only carries M0R. The governance-clean choice
is therefore M0R + M1 -- and the owner has to pick between the two without a number.

This script produces that number. For each member set it reports:

  * COVERAGE -- rows and signal dates survived by the inner join. M2's
    minimum_valid_history_length=15 leaves 17,306 labelled rows unscored, so it is the
    narrowest member and it sets the width of every ensemble it is in. Dropping it
    should widen the sample; how much is a measurement, not a guess.
  * PERFORMANCE -- the same reference portfolio (top30, buffer N_exit=60), the same
    buffer sweep, and the same trend-filter stack, all on that member set's own rows.
  * LIKE-FOR-LIKE BASELINE -- best_simple re-scored on exactly those rows, so the
    ensemble is never compared against a baseline measured on a wider set.

Read-only over the signed artifacts plus index queries; writes only --out.
"""
from __future__ import annotations

import argparse
import json
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
from optimization_probes import (  # noqa: E402
    ARTIFACTS,
    BP,
    FOLD_YEAR,
    FOLDS,
    build_portfolio,
    metrics,
    to_cycles,
)
from run_backtest import load_labels, load_predictions  # noqa: E402

WORKSPACES = {"M0R": "m0r_v2", "M1": "m1_full_v1", "M2": "m2_v1"}
MEMBER_SETS = {
    "M0R+M1+M2": ("M0R", "M1", "M2"),
    "M0R+M1 (gate-clean)": ("M0R", "M1"),
    "M0R only": ("M0R",),
    "M1 only": ("M1",),
}
TREND_CODE = "932000.CSI"


def build_member_set(artifacts: Path, members: tuple[str, ...]) -> dict[str, Any]:
    """Ensemble score frames, labels and coverage for one member set."""
    cache = {m: load_predictions(artifacts / WORKSPACES[m]) for m in members}
    ens: dict[str, pd.DataFrame] = {}
    labs: dict[str, pd.DataFrame] = {}
    base: dict[str, pd.DataFrame] = {}
    coverage: dict[str, Any] = {}
    label_cache: dict[str, pd.DataFrame] = {}
    for fold in FOLDS:
        keys = None
        for m in members:
            piece = cache[m][(fold, "candidate")][0].loc[:, ["signal_date", "ts_code"]]
            keys = piece if keys is None else keys.merge(piece, on=["signal_date", "ts_code"])
        assert keys is not None
        scores = keys.copy()
        for m in members:
            piece = cache[m][(fold, "candidate")][0].loc[:, ["signal_date", "ts_code", "score"]]
            scores = scores.merge(piece.rename(columns={"score": m}), on=["signal_date", "ts_code"])
        scores["score"] = scores[list(members)].mean(axis=1)
        ens[fold] = scores.loc[:, ["signal_date", "ts_code", "score"]]

        # baseline (best_simple) re-scored on exactly these rows
        bl = cache["M0R"][(fold, "best_simple")][0] if "M0R" in cache else None
        if bl is None:
            bl = load_predictions(artifacts / WORKSPACES["M0R"])[(fold, "best_simple")][0]
        base[fold] = bl.loc[:, ["signal_date", "ts_code", "score"]].merge(
            keys, on=["signal_date", "ts_code"])

        dataset_id = None
        for manifest in sorted((artifacts / WORKSPACES["M0R"] / "predictions").glob("*/manifest.json")):
            meta = json.loads(manifest.read_text())["metadata"]
            if meta["fold_id"] == fold:
                dataset_id = meta["dataset_id"]
                break
        assert dataset_id is not None
        if dataset_id not in label_cache:
            label_cache[dataset_id] = load_labels(
                artifacts / WORKSPACES["M0R"] / "datasets", dataset_id)[0]
        labs[fold] = label_cache[dataset_id]

        eligible = ens[fold].merge(labs[fold], on=["signal_date", "ts_code"], how="inner")
        eligible = eligible.loc[eligible["label_valid"] & np.isfinite(eligible["raw_return_5d"])]
        coverage[fold] = {
            "year": FOLD_YEAR[fold],
            "ensemble_rows": int(len(eligible)),
            "ensemble_signal_dates": int(eligible["signal_date"].nunique()),
            "baseline_rows": int(len(base[fold].merge(labs[fold], on=["signal_date", "ts_code"],
                                                      how="inner").pipe(
                lambda f: f.loc[f["label_valid"] & np.isfinite(f["raw_return_5d"])]))),
            "mean_names_per_date": float(len(eligible) / max(1, eligible["signal_date"].nunique())),
        }
    return {"ens": ens, "labs": labs, "base": base, "coverage": coverage}


def portfolio_cycles(ens: dict[str, pd.DataFrame], labs: dict[str, pd.DataFrame],
                     n_enter: int, n_exit: int | None) -> list[dict[str, Any]]:
    """Per-fold grouping (192 = 4 x 48); a cycle never straddles a model-fold boundary."""
    out: list[dict[str, Any]] = []
    for fold in FOLDS:
        out.extend(cycles_with_days(build_portfolio(ens[fold], labs[fold], n_enter, n_exit)))
    return out


def summarise(cycles: list[dict[str, Any]], bp: float,
              exposure: dict[date, float] | None = None, floor: float = 0.0) -> dict[str, float]:
    if exposure is not None:
        scaled = []
        prev = None
        for c in cycles:
            scales = [exposure.get(d, 1.0) for d in c.get("days", [c["start"]])]
            s = floor + (1.0 - floor) * float(np.mean(scales))
            switch = abs(s - prev) if prev is not None else 0.0
            prev = s
            scaled.append({**c, "ret": c["ret"] * s, "turnover": c["turnover"] * s + switch})
        cycles = scaled
    return metrics(cycles, bp)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--bp", type=float, default=BP)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "gate_clean_ensemble_results.json")
    args = parser.parse_args()

    closes = load_index_closes()
    exposures = {ma: trend_exposure(closes[TREND_CODE], ma) for ma in (10, 20)}
    result: dict[str, Any] = {"config": {
        "round_trip_bp": args.bp, "trend_index": TREND_CODE, "trend_mas": [10, 20],
        "member_sets": {k: list(v) for k, v in MEMBER_SETS.items()},
        "gate_status": {"M0R": "six gates passed; registry status WALK_FORWARD_PASSED",
                        "M1": "six gates passed; NOT registered",
                        "M2": "FAILED G4 (+0.0049591733 vs 0.005) and G6 (905/969 = 0.9340 vs 0.95)"},
        "baseline_treatment": "best_simple re-scored on the member set's own intersection rows",
    }, "member_sets": {}}

    for tag, members in MEMBER_SETS.items():
        print(f"\n=== {tag} ===", flush=True)
        built = build_member_set(args.artifacts, members)
        cov = built["coverage"]
        total_rows = sum(cov[f]["ensemble_rows"] for f in FOLDS)
        total_dates = sum(cov[f]["ensemble_signal_dates"] for f in FOLDS)
        entry: dict[str, Any] = {"members": list(members), "coverage": cov,
                                 "total_eligible_rows": total_rows,
                                 "total_signal_dates": total_dates}
        print(f"      合格行 {total_rows:,}  信号日 {total_dates}  "
              f"日均候选 {total_rows/max(1,total_dates):.0f}", flush=True)

        variants: dict[str, Any] = {}
        for label, (n_enter, n_exit) in (("top30_exit60", (30, 60)),
                                        ("top30_exit150", (30, 150)),
                                        ("top100_exit60", (100, 60))):
            cyc = portfolio_cycles(built["ens"], built["labs"], n_enter, n_exit)
            variants[label] = {"none": summarise(cyc, args.bp)}
            for ma, exp in exposures.items():
                variants[label][f"ma{ma}"] = summarise(cyc, args.bp, exp)
            m0 = variants[label]["none"]
            print(f"      {label:<14} 无过滤 年化 {m0['net_annualized']*100:>6.2f}%  "
                  f"Sharpe {m0['net_sharpe']:.3f}  回撤 {m0['max_drawdown']*100:>6.2f}%  "
                  f"累计 {m0['total_return']*100:>6.2f}%  换手 {m0['mean_turnover']:.3f}", flush=True)
            for ma in exposures:
                m = variants[label][f"ma{ma}"]
                print(f"                     +MA{ma:<3} 年化 {m['net_annualized']*100:>6.2f}%  "
                      f"Sharpe {m['net_sharpe']:.3f}  回撤 {m['max_drawdown']*100:>6.2f}%  "
                      f"累计 {m['total_return']*100:>6.2f}%", flush=True)
        entry["variants"] = variants

        bl_cyc = portfolio_cycles(built["base"], built["labs"], 30, None)
        bl = summarise(bl_cyc, args.bp)
        entry["baseline_same_rows"] = bl
        print(f"      基线(同行,top30) 年化 {bl['net_annualized']*100:>6.2f}%  "
              f"Sharpe {bl['net_sharpe']:.3f}  回撤 {bl['max_drawdown']*100:>6.2f}%  "
              f"累计 {bl['total_return']*100:>6.2f}%", flush=True)
        result["member_sets"][tag] = entry

    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
