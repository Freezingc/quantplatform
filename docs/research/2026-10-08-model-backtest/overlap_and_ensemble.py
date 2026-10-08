#!/usr/bin/env python3
"""Top-50 selection overlap between models, and an equal-weight score ensemble.

Reproduces sections 3.1 and 3.2 of backtest_report.md. Read-only over the signed
frozen predictions; reuses run_backtest's cycle machinery so the ensemble is
measured on exactly the same footing as the single models.

The ensemble is the equal-weight mean of the three percentile scores, evaluated on
the INNER JOIN of the three coverages. Because that changes the row set, the
baseline is re-run on the same intersection rows -- comparing an ensemble built on
M2's narrower coverage against a baseline scored on its own wider set would not be
a like-for-like difference.

Usage: overlap_and_ensemble.py [--out overlap_ensemble_results.json] [--round-trip-bp 30]
"""
from __future__ import annotations

import argparse
import itertools
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
ENSEMBLE_MEMBERS = ("M0R", "M1", "M2")
WORKSPACES = {"M0R": "m0r_v2", "M1": "m1_full_v1", "M2": "m2_v1", "M0-refined": "m0_v3"}
FOLDS = ("F1", "F2", "F3", "F4")
FOLD_YEAR = {"F1": 2022, "F2": 2023, "F3": 2024, "F4": 2025}
RULES = ("top50", "top100", "q5")
TOP_N = 50


def _candidate(ws_dir: Path, fold: str) -> pd.DataFrame:
    return load_predictions(ws_dir)[(fold, "candidate")][0]


def _baseline(ws_dir: Path, fold: str) -> pd.DataFrame:
    return load_predictions(ws_dir)[(fold, "best_simple")][0]


def _dataset_id(ws_dir: Path, fold: str) -> str:
    for manifest in sorted((ws_dir / "predictions").glob("*/manifest.json")):
        meta = json.loads(manifest.read_text())["metadata"]
        if meta["fold_id"] == fold:
            return meta["dataset_id"]
    raise KeyError(f"no prediction manifest for {fold}")


def top_sets(scores: pd.DataFrame, n: int = TOP_N) -> dict[Any, set[str]]:
    """signal_date -> the n highest-scoring securities that day."""
    out: dict[Any, set[str]] = {}
    for day, frame in scores.groupby("signal_date", sort=False):
        out[day] = set(frame.nlargest(n, "score")["ts_code"].astype(str))
    return out


def overlap(a: dict[Any, set[str]], b: dict[Any, set[str]]) -> dict[str, float]:
    common = sorted(set(a) & set(b))
    shared = [len(a[d] & b[d]) for d in common]
    jaccard = [len(a[d] & b[d]) / len(a[d] | b[d]) for d in common if a[d] or b[d]]
    return {
        "days": len(shared),
        "mean_shared_names": float(np.mean(shared)) if shared else float("nan"),
        "mean_jaccard": float(np.mean(jaccard)) if jaccard else float("nan"),
        "top_n": TOP_N,
    }


def stats(returns: list[float], turnovers: list[float], bp: float) -> dict[str, float]:
    ret = np.asarray(returns, dtype=float)
    net = ret - np.asarray(turnovers, dtype=float) * (bp / 10_000.0)
    equity = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(equity)
    mean, std = float(net.mean()), float(net.std(ddof=1))
    return {
        "cycles": int(len(net)),
        "gross_annualized": float(ret.mean()) * CYCLES_PER_YEAR,
        "net_annualized": mean * CYCLES_PER_YEAR,
        "net_sharpe": (mean / std * np.sqrt(CYCLES_PER_YEAR)) if std > 0 else float("nan"),
        "net_total_return": float(equity[-1] - 1.0),
        "net_max_drawdown": float((equity / peak - 1.0).min()),
        "mean_turnover": float(np.mean(turnovers)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--round-trip-bp", type=float, default=30.0)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "overlap_ensemble_results.json")
    args = parser.parse_args()
    bp = args.round_trip_bp

    sets: dict[str, dict[Any, set[str]]] = {}
    for name, ws in WORKSPACES.items():
        merged: dict[Any, set[str]] = {}
        for fold in FOLDS:
            merged.update(top_sets(_candidate(args.artifacts / ws, fold)))
        sets[name] = merged
    base_sets: dict[Any, set[str]] = {}
    for fold in FOLDS:
        base_sets.update(top_sets(_baseline(args.artifacts / WORKSPACES["M0R"], fold)))
    sets["BASELINE"] = base_sets

    pairs: dict[str, Any] = {}
    for a, b in itertools.combinations(WORKSPACES, 2):
        pairs[f"{a} vs {b}"] = overlap(sets[a], sets[b])
    for a in WORKSPACES:
        pairs[f"{a} vs BASELINE"] = overlap(sets[a], sets["BASELINE"])
    triple = sorted(set(sets["M0R"]) & set(sets["M1"]) & set(sets["M2"]))
    unions = [len(sets["M0R"][d] | sets["M1"][d] | sets["M2"][d]) for d in triple]
    inters = [len(sets["M0R"][d] & sets["M1"][d] & sets["M2"][d]) for d in triple]

    ens_pool = {r: {"ensemble": [], "baseline_same_rows": []} for r in RULES}
    ens_turn = {r: {"ensemble": [], "baseline_same_rows": []} for r in RULES}
    per_year: dict[str, dict[str, dict[str, float]]] = {r: {} for r in RULES}
    intersection_rows: dict[str, int] = {}
    for fold in FOLDS:
        joined = None
        for member in ENSEMBLE_MEMBERS:
            frame = _candidate(args.artifacts / WORKSPACES[member], fold)
            piece = frame.loc[:, ["signal_date", "ts_code", "score"]].rename(columns={"score": member})
            joined = piece if joined is None else joined.merge(piece, on=["signal_date", "ts_code"], how="inner")
        intersection_rows[fold] = int(len(joined))
        assert joined is not None
        joined["score"] = joined[list(ENSEMBLE_MEMBERS)].mean(axis=1)
        ensemble = joined.loc[:, ["signal_date", "ts_code", "score"]]
        keys = joined.loc[:, ["signal_date", "ts_code"]]
        baseline_same = _baseline(args.artifacts / WORKSPACES["M0R"], fold).merge(
            keys, on=["signal_date", "ts_code"], how="inner"
        )
        dsid = _dataset_id(args.artifacts / WORKSPACES["M0R"], fold)
        labels, _ = load_labels(args.artifacts / WORKSPACES["M0R"] / "datasets", dsid)
        for rule in RULES:
            year_returns = {}
            for tag, frame in (("ensemble", ensemble), ("baseline_same_rows", baseline_same)):
                res = backtest_series(frame, labels, rule, 5)
                if not res.get("usable"):
                    continue
                ens_pool[rule][tag].extend(res["cycle_returns"])
                ens_turn[rule][tag].extend(res["cycle_turnover"])
                year_returns[tag] = stats(res["cycle_returns"], res["cycle_turnover"], bp)
            per_year[rule][str(FOLD_YEAR[fold])] = year_returns

    result: dict[str, Any] = {
        "config": {
            "round_trip_bp_ASSUMPTION": bp,
            "top_n_for_overlap": TOP_N,
            "ensemble_members": list(ENSEMBLE_MEMBERS),
            "ensemble_rule": "equal-weight mean of percentile scores, on the inner join of the three coverages",
            "baseline_treatment": "re-run on the same intersection rows so the difference is like-for-like",
            "cycles_per_year": CYCLES_PER_YEAR,
        },
        "intersection_rows_by_fold": intersection_rows,
        "overlap": pairs,
        "triple": {
            "days": len(triple),
            "mean_union_names": float(np.mean(unions)),
            "mean_names_in_all_three": float(np.mean(inters)),
            "share_of_days_with_at_least_one_consensus_name": float(np.mean([x > 0 for x in inters])),
        },
        "ensemble": {
            rule: {
                tag: stats(ens_pool[rule][tag], ens_turn[rule][tag], bp)
                for tag in ("ensemble", "baseline_same_rows")
                if ens_pool[rule][tag]
            }
            for rule in RULES
        },
        "ensemble_per_year": per_year,
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
