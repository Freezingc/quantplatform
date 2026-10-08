#!/usr/bin/env python3
"""How concentrated can the book be? Top-N sweep for the ensemble and the models.

Small capital changes two things at once: holding 10-30 names becomes feasible, and
market impact stops mattering (1-2M spread over 10-30 names is 33k-200k per name,
which is a rounding error against the daily turnover of a liquid A-share). So this
sweeps N and reports every cost level, letting the reader pick the line that matches
their size rather than inheriting one assumption.

The ensemble is the equal-weight mean of the three percentile scores on the inner join
of their coverages, and the baseline is re-run on those same rows, exactly as in
overlap_and_ensemble.py -- otherwise a concentrated ensemble would be compared against
a baseline scored on a wider set.

Reuses run_backtest's cycle machinery, so holding period, equal weighting, eligibility
and annualisation are identical to every other table in this study.

Read-only over the signed artifacts; writes only --out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_backtest import (  # noqa: E402
    CYCLES_PER_YEAR,
    backtest_series,
    load_labels,
    load_predictions,
)

ARTIFACTS = Path("/data/zhangyudong/qrv2_research_artifacts")
WORKSPACES = {"M0R": "m0r_v2", "M1": "m1_full_v1", "M2": "m2_v1"}
ENSEMBLE_MEMBERS = ("M0R", "M1", "M2")
FOLDS = ("F1", "F2", "F3", "F4")
FOLD_YEAR = {"F1": 2022, "F2": 2023, "F3": 2024, "F4": 2025}
SIZES = (5, 10, 15, 20, 30, 50, 100)
COST_BP = (0.0, 10.0, 15.0, 30.0, 50.0)


def _metric(returns: np.ndarray, turnovers: np.ndarray, bp: float) -> dict[str, float]:
    net = returns - turnovers * (bp / 10_000.0)
    equity = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(equity)
    mean, std = float(net.mean()), float(net.std(ddof=1))
    return {
        "annualized_return": mean * CYCLES_PER_YEAR,
        "annualized_vol": std * float(np.sqrt(CYCLES_PER_YEAR)),
        "sharpe": (mean / std * float(np.sqrt(CYCLES_PER_YEAR))) if std > 0 else float("nan"),
        "win_rate": float((net > 0).mean()),
        "max_drawdown": float((equity / peak - 1.0).min()),
        "total_return": float(equity[-1] - 1.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "concentration_results.json",
    )
    args = parser.parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    label_cache: dict[str, pd.DataFrame] = {}
    store: dict[str, dict[int, dict[str, list[float]]]] = {
        name: {n: {"returns": [], "turnover": [], "names": [], "yearly": []} for n in sizes}
        for name in ("ensemble", "baseline_same_rows", "M0R", "M1", "M2")
    }
    intersection_rows: dict[str, int] = {}
    realised_names: dict[str, dict[int, float]] = {k: {} for k in store}

    for fold in FOLDS:
        predictions = {m: load_predictions(args.artifacts / WORKSPACES[m]) for m in WORKSPACES}
        joined = None
        for member in ENSEMBLE_MEMBERS:
            frame = predictions[member][(fold, "candidate")][0]
            piece = frame.loc[:, ["signal_date", "ts_code", "score"]].rename(columns={"score": member})
            joined = piece if joined is None else joined.merge(piece, on=["signal_date", "ts_code"], how="inner")
        assert joined is not None
        intersection_rows[fold] = int(len(joined))
        joined["score"] = joined[list(ENSEMBLE_MEMBERS)].mean(axis=1)
        keys = joined.loc[:, ["signal_date", "ts_code"]]
        baseline_same = predictions["M0R"][(fold, "best_simple")][0].merge(
            keys, on=["signal_date", "ts_code"], how="inner"
        )
        sources = {
            "ensemble": joined.loc[:, ["signal_date", "ts_code", "score"]],
            "baseline_same_rows": baseline_same,
            "M0R": predictions["M0R"][(fold, "candidate")][0],
            "M1": predictions["M1"][(fold, "candidate")][0],
            "M2": predictions["M2"][(fold, "candidate")][0],
        }
        dataset_id = None
        for manifest in sorted((args.artifacts / WORKSPACES["M0R"] / "predictions").glob("*/manifest.json")):
            meta = json.loads(manifest.read_text())["metadata"]
            if meta["fold_id"] == fold:
                dataset_id = meta["dataset_id"]
                break
        assert dataset_id is not None
        if dataset_id not in label_cache:
            label_cache[dataset_id] = load_labels(
                args.artifacts / WORKSPACES["M0R"] / "datasets", dataset_id
            )[0]
        labels = label_cache[dataset_id]

        for name, frame in sources.items():
            for size in sizes:
                result = backtest_series(frame, labels, f"top{size}", 5)
                if not result.get("usable"):
                    continue
                bucket = store[name][size]
                bucket["returns"].extend(result["cycle_returns"])
                bucket["turnover"].extend(result["cycle_turnover"])
                bucket["names"].extend([result["mean_names"]] * len(result["cycle_returns"]))
                bucket["yearly"].append(
                    {
                        "fold": fold,
                        "year": FOLD_YEAR[fold],
                        "returns": result["cycle_returns"],
                        "turnover": result["cycle_turnover"],
                    }
                )

    output: dict[str, Any] = {
        "config": {
            "sizes": sizes,
            "cost_bp_reported": list(COST_BP),
            "cycles_per_year": CYCLES_PER_YEAR,
            "holding_days": 5,
            "ensemble_members": list(ENSEMBLE_MEMBERS),
            "weighting": "equal",
            "eligibility": "label_valid and finite raw_return_5d",
            "baseline_treatment": "scored on the ensemble's intersection rows",
        },
        "intersection_rows_by_fold": intersection_rows,
        "results": {},
    }
    for name, by_size in store.items():
        output["results"][name] = {}
        for size, bucket in by_size.items():
            ret = np.asarray(bucket["returns"], dtype=float)
            turn = np.asarray(bucket["turnover"], dtype=float)
            if not len(ret):
                continue
            entry: dict[str, Any] = {
                "cycles": int(len(ret)),
                "mean_names": float(np.mean(bucket["names"])),
                "mean_turnover": float(turn.mean()),
                "by_cost_bp": {str(bp): _metric(ret, turn, bp) for bp in COST_BP},
                "yearly_net_annualized_15bp": {},
                "yearly_net_annualized_30bp": {},
            }
            for row in bucket["yearly"]:
                r = np.asarray(row["returns"], dtype=float)
                t = np.asarray(row["turnover"], dtype=float)
                entry["yearly_net_annualized_15bp"][str(row["year"])] = float(_metric(r, t, 15.0)["annualized_return"])
                entry["yearly_net_annualized_30bp"][str(row["year"])] = float(_metric(r, t, 30.0)["annualized_return"])
            output["results"][name][str(size)] = entry

    args.out.write_text(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    for name in ("ensemble", "M2", "baseline_same_rows"):
        if name not in output["results"]:
            continue
        print(f"--- {name}")
        print(f"{'N':>4}{'names':>7}" + "".join(f"{bp:>9.0f}bp" for bp in COST_BP) + f"{'maxDD@15':>10}{'turn':>8}")
        for size in sizes:
            e = output["results"][name].get(str(size))
            if not e:
                continue
            cells = "".join(f"{e['by_cost_bp'][str(bp)]['sharpe']:>11.3f}" for bp in COST_BP)
            print(
                f"{size:>4}{e['mean_names']:>7.1f}{cells}"
                f"{e['by_cost_bp']['15.0']['max_drawdown']*100:>9.2f}%{e['mean_turnover']:>8.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
