#!/usr/bin/env python3
"""Read-only quantile-portfolio backtest over the signed frozen TEST predictions.

What it reads
    <artifacts>/<ws>/predictions/*/scores.parquet   candidate and best_simple scores
    <artifacts>/<ws>/predictions/*/manifest.json    fold / kind / dataset_id binding
    <artifacts>/<ws>/datasets/<dataset_id>/split=test/**.parquet   labels

What it never does
    no writes outside --out; no database access; no model weights or checkpoints;
    no re-inference -- every score is read from an artifact the signed evaluation
    already verified. Label semantics come from labels/forward_return.py:
    raw_return_5d = adjusted_open(T+6) / adjusted_open(T+1) - 1, i.e. buy at the
    open after the signal, hold five trading days, sell at the open.

Portfolio construction
    Signals are daily but each position is held five trading days, so consecutive
    signals overlap. Rather than average overlapping returns (which inflates the
    sample and hides autocorrelation), signals are grouped into cycles of five
    consecutive signal dates. One cycle is exactly one non-overlapping round trip
    for a staggered book that commits 1/5 of capital each day, so the cycle series
    is the strategy's true return series and annualises at 243/5 cycles per year.

    Selection is equal weight: the artifacts carry no market cap or price level,
    so value weighting is not computable from them and is not attempted.

Costs
    No cost data exists in the artifacts. --round-trip-bp is an ASSUMPTION applied
    to measured turnover, never a measurement; gross numbers are reported alongside.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HOLDING_DAYS = 5
TRADING_DAYS_PER_YEAR = 243
CYCLES_PER_YEAR = TRADING_DAYS_PER_YEAR / HOLDING_DAYS
FOLDS = ("F1", "F2", "F3", "F4")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_labels(dataset_dir: Path, dataset_id: str) -> pd.DataFrame:
    """signal_date/ts_code/raw_return_5d/label_valid from the frozen TEST split."""
    shards = sorted((dataset_dir / dataset_id / "split=test").rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no TEST shards under {dataset_dir / dataset_id}")
    frames = [pd.read_parquet(shard) for shard in shards]
    labels = pd.concat(frames, ignore_index=True).loc[
        :, ["signal_date", "ts_code", "raw_return_5d", "label_valid"]
    ]
    labels["signal_date"] = pd.to_datetime(labels["signal_date"], errors="raise").dt.date
    labels["label_valid"] = labels["label_valid"].fillna(False).astype(bool)
    labels["raw_return_5d"] = pd.to_numeric(labels["raw_return_5d"], errors="coerce")
    return labels, {str(shard): sha256_file(shard) for shard in shards}


def load_predictions(ws_dir: Path) -> dict[tuple[str, str], tuple[pd.DataFrame, str]]:
    """(fold, kind) -> (scores frame, prediction_id), from the signed artifacts."""
    out: dict[tuple[str, str], tuple[pd.DataFrame, str]] = {}
    for manifest_path in sorted((ws_dir / "predictions").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        meta = manifest["metadata"]
        scores = pd.read_parquet(manifest_path.parent / "scores.parquet")
        scores["signal_date"] = pd.to_datetime(scores["signal_date"], errors="raise").dt.date
        out[(meta["fold_id"], meta["kind"])] = (scores, manifest["prediction_id"])
    return out


def select_names(day: pd.DataFrame, rule: str, quantiles: int) -> pd.Series:
    """The bool mask of held names for one signal date, by score rank."""
    score = day["score"].to_numpy(dtype=float)
    if rule.startswith("top"):
        count = int(rule[3:])
        if len(score) <= count:
            return pd.Series(True, index=day.index)
        threshold = np.partition(score, -count)[-count]
        return pd.Series(score >= threshold, index=day.index)
    if rule.startswith("q"):
        index = int(rule[1:])
        rank = pd.Series(score, index=day.index).rank(method="first", pct=True)
        lower = (index - 1) / quantiles
        return (rank > lower) & (rank <= index / quantiles)
    raise ValueError(f"unknown selection rule {rule!r}")


def backtest_series(
    scores: pd.DataFrame, labels: pd.DataFrame, rule: str, quantiles: int
) -> dict[str, Any]:
    """Cycle-by-cycle returns and turnover for one selection rule."""
    joined = scores.merge(labels, on=["signal_date", "ts_code"], how="inner")
    joined = joined.loc[joined["label_valid"] & np.isfinite(joined["raw_return_5d"])]
    if joined.empty:
        return {"usable": False}
    dates = sorted(joined["signal_date"].unique())
    by_date = {day: frame for day, frame in joined.groupby("signal_date", sort=False)}

    period_rows: list[dict[str, float]] = []
    previous: set[str] | None = None
    for day in dates:
        frame = by_date[day]
        mask = select_names(frame, rule, quantiles)
        held = frame.loc[mask]
        if held.empty:
            previous = None
            continue
        names = set(held["ts_code"].astype(str))
        turnover = 1.0 if previous is None else 1.0 - len(names & previous) / max(len(names), 1)
        period_rows.append(
            {
                "signal_date": day,
                "ret": float(held["raw_return_5d"].mean()),
                "n": int(len(held)),
                "turnover": turnover,
            }
        )
        previous = names
    if not period_rows:
        return {"usable": False}

    periods = pd.DataFrame(period_rows)
    # Group consecutive signals into 5-day cycles: one cycle = one non-overlapping
    # round trip for the staggered book, so this series carries no overlap.
    periods["cycle"] = np.arange(len(periods)) // HOLDING_DAYS
    full = periods.groupby("cycle").filter(lambda g: len(g) == HOLDING_DAYS)
    if full.empty:
        return {"usable": False, "signal_dates": len(periods)}
    cycles = full.groupby("cycle").agg(
        ret=("ret", "mean"), turnover=("turnover", "mean"), n=("n", "mean")
    )
    return {
        "usable": True,
        "signal_dates": int(len(periods)),
        "cycles": int(len(cycles)),
        "mean_names": float(cycles["n"].mean()),
        "mean_turnover": float(cycles["turnover"].mean()),
        "cycle_returns": cycles["ret"].tolist(),
        "cycle_turnover": cycles["turnover"].tolist(),
    }


def summarise(
    result: dict[str, Any], round_trip_bp: float
) -> dict[str, float] | dict[str, Any]:
    if not result.get("usable") or not result.get("cycle_returns"):
        return {"usable": False}
    ret = np.asarray(result["cycle_returns"], dtype=float)
    turnover = np.asarray(result["cycle_turnover"], dtype=float)
    cost_per_cycle = turnover * (round_trip_bp / 10_000.0)
    net = ret - cost_per_cycle
    equity = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(equity)
    max_dd = float((equity / peak - 1.0).min())

    def stats(series: np.ndarray) -> dict[str, float]:
        mean = float(series.mean())
        std = float(series.std(ddof=1)) if len(series) > 1 else float("nan")
        return {
            "mean_per_cycle": mean,
            "annualized_return": mean * CYCLES_PER_YEAR,
            "annualized_vol": std * math.sqrt(CYCLES_PER_YEAR),
            "sharpe": (mean / std * math.sqrt(CYCLES_PER_YEAR)) if std > 0 else float("nan"),
            "win_rate": float((series > 0).mean()),
        }

    gross = stats(ret)
    netstats = stats(net)
    return {
        "usable": True,
        "cycles": result["cycles"],
        "signal_dates": result["signal_dates"],
        "mean_names": result["mean_names"],
        "mean_turnover": result["mean_turnover"],
        "gross": gross,
        "net": netstats,
        "max_drawdown_net": max_dd,
        "total_net_return": float(equity[-1] - 1.0),
        "cost_drag_annualized": float(cost_per_cycle.mean() * CYCLES_PER_YEAR),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts", type=Path, default=Path("/data/zhangyudong/qrv2_research_artifacts")
    )
    parser.add_argument(
        "--workspace",
        action="append",
        default=[],
        help="LABEL=DIRNAME, repeatable; defaults to the four authoritative candidate runs",
    )
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument(
        "--round-trip-bp",
        type=float,
        default=30.0,
        help="ASSUMED one-round-trip cost in basis points, applied to measured turnover",
    )
    parser.add_argument(
        "--rules",
        default="q1,q2,q3,q4,q5,top50,top100",
        help="comma-separated selection rules",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    workspaces = args.workspace or [
        "M0R=m0r_v2",
        "M1=m1_full_v1",
        "M2=m2_v1",
        "M0-refined=m0_v3",
    ]
    parsed = []
    for entry in workspaces:
        label, _, dirname = entry.partition("=")
        parsed.append((label or dirname, dirname))

    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    label_cache: dict[str, pd.DataFrame] = {}
    provenance: dict[str, str] = {}
    output: dict[str, Any] = {
        "config": {
            "holding_days": HOLDING_DAYS,
            "cycles_per_year": CYCLES_PER_YEAR,
            "quantiles": args.quantiles,
            "round_trip_bp_ASSUMPTION": args.round_trip_bp,
            "weighting": "equal (no market cap in the artifacts)",
            "eligibility": "label_valid and finite raw_return_5d only",
            "label_semantics": "adjusted_open(T+6)/adjusted_open(T+1)-1",
        },
        "by_workspace": {},
    }

    for label, dirname in parsed:
        ws_dir = args.artifacts / dirname
        predictions = load_predictions(ws_dir)
        per_fold: dict[str, Any] = {}
        pooled: dict[str, list[float]] = {rule: [] for rule in rules}
        pooled_turnover: dict[str, list[float]] = {rule: [] for rule in rules}
        pooled_names: dict[str, list[float]] = {rule: [] for rule in rules}
        base_pooled: dict[str, list[float]] = {rule: [] for rule in rules}
        base_turnover: dict[str, list[float]] = {rule: [] for rule in rules}
        base_names: dict[str, list[float]] = {rule: [] for rule in rules}
        for fold in FOLDS:
            if (fold, "candidate") not in predictions:
                continue
            dataset_id = json.loads(
                next(
                    p
                    for p in (ws_dir / "predictions").glob("*/manifest.json")
                    if json.loads(p.read_text())["metadata"]["fold_id"] == fold
                ).read_text()
            )["metadata"]["dataset_id"]
            if dataset_id not in label_cache:
                frames, hashes = load_labels(ws_dir / "datasets", dataset_id)
                label_cache[dataset_id] = frames
                provenance.update(hashes)
            labels = label_cache[dataset_id]
            cand, cand_id = predictions[(fold, "candidate")]
            base, base_id = predictions.get((fold, "best_simple"), (None, None))
            fold_out: dict[str, Any] = {"dataset_id": dataset_id, "candidate_prediction_id": cand_id}
            for rule in rules:
                res = backtest_series(cand, labels, rule, args.quantiles)
                fold_out[rule] = summarise(res, args.round_trip_bp)
                if res.get("usable"):
                    pooled[rule].extend(res["cycle_returns"])
                    pooled_turnover[rule].extend(res["cycle_turnover"])
                    pooled_names[rule].extend([res["mean_names"]] * len(res["cycle_returns"]))
            if base is not None:
                fold_out["baseline_prediction_id"] = base_id
                for rule in rules:
                    bres = backtest_series(base, labels, rule, args.quantiles)
                    fold_out[f"baseline_{rule}"] = summarise(bres, args.round_trip_bp)
                    if bres.get("usable"):
                        base_pooled[rule].extend(bres["cycle_returns"])
                        base_turnover[rule].extend(bres["cycle_turnover"])
                        base_names[rule].extend([bres["mean_names"]] * len(bres["cycle_returns"]))
            per_fold[fold] = fold_out

        def pool_block(
            returns: list[float], turnovers: list[float], names: list[float]
        ) -> dict[str, Any] | None:
            if not returns:
                return None
            ret = np.asarray(returns, dtype=float)
            turnover = np.asarray(turnovers, dtype=float)
            cost = turnover * (args.round_trip_bp / 10_000.0)
            net = ret - cost
            equity = np.cumprod(1.0 + net)
            peak = np.maximum.accumulate(equity)
            mean, std = float(net.mean()), float(net.std(ddof=1))
            return {
                "cycles": int(len(net)),
                "mean_names": float(np.asarray(names, dtype=float).mean()),
                "gross_mean_per_cycle": float(ret.mean()),
                "gross_annualized": float(ret.mean()) * CYCLES_PER_YEAR,
                "net_mean_per_cycle": mean,
                "net_annualized": mean * CYCLES_PER_YEAR,
                "net_annualized_vol": std * math.sqrt(CYCLES_PER_YEAR),
                "net_sharpe": (mean / std * math.sqrt(CYCLES_PER_YEAR)) if std > 0 else float("nan"),
                "net_win_rate": float((net > 0).mean()),
                "net_max_drawdown": float((equity / peak - 1.0).min()),
                "net_total_return": float(equity[-1] - 1.0),
                "mean_turnover": float(turnover.mean()),
                "cost_drag_annualized": float(cost.mean()) * CYCLES_PER_YEAR,
            }

        pooled_out = {
            rule: block
            for rule in rules
            if (block := pool_block(pooled[rule], pooled_turnover[rule], pooled_names[rule]))
        }
        baseline_out = {
            rule: block
            for rule in rules
            if (
                block := pool_block(
                    base_pooled[rule], base_turnover[rule], base_names[rule]
                )
            )
        }
        output["by_workspace"][label] = {
            "workspace": dirname,
            "per_fold": per_fold,
            "pooled_2022_2025": pooled_out,
            "baseline_pooled_2022_2025": baseline_out,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    output["label_shard_sha256"] = provenance
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
