"""Audit the Alpha206 median-volume repair with deterministic robustness checks.

The audit consumes the already materialized daily RankIC artifact and local
Tushare daily cache.  It is deliberately read-only and never evaluates 2026.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_alpha165_168_canonical_quality import LOCKED_TEST_START

CANDIDATE = "yeslab_206_median_abs_v2"
SOURCE_FACTOR_ID = "yeslab_206"
FROZEN_DIRECTION = -1
DEFAULT_RANK_IC = Path(
    "research_v2/artifacts/diagnostics/alpha206_volume_variants_2020_2025_v1/median_abs/daily_rank_ic.csv"
)
DEFAULT_CACHE_DIR = Path("research_v2/artifacts/tushare_cache")
SEED = 20260813
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
MIN_SIGNED_MEAN = 0.02
MIN_ICIR = 0.50


def _metrics(values: pd.Series | np.ndarray) -> dict[str, float | int]:
    values = pd.to_numeric(pd.Series(values), errors="coerce")
    values = values.loc[np.isfinite(values.to_numpy(float))].to_numpy(float)
    signed = values * FROZEN_DIRECTION
    std = float(signed.std(ddof=1)) if len(signed) > 1 else float("nan")
    return {
        "valid_day_count": int(len(signed)),
        "signed_mean_rank_ic": float(signed.mean()) if len(signed) else float("nan"),
        "signed_icir": float(signed.mean() / std) if np.isfinite(std) and std > 0 else float("nan"),
        "directional_day_share": float((signed > 0).mean()) if len(signed) else float("nan"),
    }


def _read_rank_ic(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"signal_date", "rank_ic"}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"RankIC artifact is missing columns: {missing}")
    frame = frame.loc[:, ["signal_date", "rank_ic"]].copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise")
    frame["rank_ic"] = pd.to_numeric(frame["rank_ic"], errors="coerce")
    frame = frame.loc[np.isfinite(frame["rank_ic"].to_numpy(float))]
    frame = frame.loc[frame["signal_date"] < pd.Timestamp(LOCKED_TEST_START)]
    if frame.empty:
        raise ValueError("RankIC artifact has no pre-2026 observations")
    return frame.sort_values("signal_date", kind="mergesort").reset_index(drop=True)


def _cache_paths(cache_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(cache_dir.glob("daily_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].json")):
        token = path.stem.removeprefix("daily_")
        try:
            current = pd.Timestamp(date.fromisoformat(f"{token[:4]}-{token[4:6]}-{token[6:8]}"))
        except ValueError:
            continue
        if start <= current <= end:
            paths.append(path)
    return paths


def load_market_states(cache_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Build an SH/SZ equal-weight market state proxy from cached Tushare daily data."""

    rows: list[dict[str, object]] = []
    for path in _cache_paths(cache_dir, start, end):
        frame = pd.read_json(path, orient="table")
        if not {"ts_code", "trade_date", "pct_chg"}.issubset(frame.columns):
            continue
        frame["ts_code"] = frame["ts_code"].astype(str)
        frame = frame.loc[frame["ts_code"].str.endswith((".SH", ".SZ"))].copy()
        frame["pct_chg"] = pd.to_numeric(frame["pct_chg"], errors="coerce") / 100.0
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["pct_chg"])
        if frame.empty:
            continue
        grouped = frame.groupby("trade_date", sort=False)["pct_chg"].agg(["mean", "count"])
        rows.extend(
            {
                "signal_date": pd.Timestamp(str(day)),
                "market_return": float(row["mean"]),
                "market_sample_count": int(row["count"]),
            }
            for day, row in grouped.iterrows()
        )
    if not rows:
        raise ValueError("Tushare cache has no SH/SZ daily market observations")
    result = (
        pd.DataFrame(rows)
        .groupby("signal_date", as_index=False)
        .agg(
            market_return=("market_return", "mean"),
            market_sample_count=("market_sample_count", "max"),
        )
    )
    result = result.sort_values("signal_date", kind="mergesort").reset_index(drop=True)
    result["trailing_return_20d"] = (1.0 + result["market_return"]).rolling(
        20, min_periods=20
    ).apply(np.prod, raw=True) - 1.0
    result["volatility_20d"] = result["market_return"].rolling(20, min_periods=20).std()
    train_vol = result.loc[result["signal_date"] < pd.Timestamp("2025-01-01"), "volatility_20d"]
    result["volatility_regime"] = np.where(
        result["volatility_20d"] > float(train_vol.median()), "high", "low"
    )
    result["trend_regime"] = np.where(
        result["trailing_return_20d"] > 0.0, "positive", "nonpositive"
    )
    return result


def leave_one_month(frame: pd.DataFrame) -> dict[str, object]:
    frame = frame.copy()
    frame["month"] = frame["signal_date"].dt.to_period("M").astype(str)
    baseline = _metrics(frame["rank_ic"])
    exclusions = []
    for month in sorted(frame["month"].unique()):
        exclusions.append(
            {"month": month, **_metrics(frame.loc[frame["month"] != month, "rank_ic"])}
        )
    worst_mean = min(exclusions, key=lambda row: float(row["signed_mean_rank_ic"]))
    worst_icir = min(exclusions, key=lambda row: float(row["signed_icir"]))
    status = (
        "PASS"
        if float(worst_mean["signed_mean_rank_ic"]) >= MIN_SIGNED_MEAN
        and float(worst_icir["signed_icir"]) >= MIN_ICIR
        else "PASS_PRIMARY_MEAN_MONITOR_STRICT_ICIR"
    )
    if float(worst_mean["signed_mean_rank_ic"]) < MIN_SIGNED_MEAN:
        status = "FAIL_SIGNED_MEAN"
    return {
        "baseline": baseline,
        "month_count": len(exclusions),
        "minimum_leave_one_month_signed_mean_rank_ic": float(worst_mean["signed_mean_rank_ic"]),
        "minimum_leave_one_month_signed_icir": float(worst_icir["signed_icir"]),
        "worst_signed_mean_exclusion": worst_mean["month"],
        "worst_icir_exclusion": worst_icir["month"],
        "status": status,
    }


def month_block_bootstrap(
    frame: pd.DataFrame, replicates: int, rng: np.random.Generator
) -> dict[str, object]:
    frame = frame.copy()
    frame["month"] = frame["signal_date"].dt.to_period("M").astype(str)
    blocks = [group["rank_ic"].to_numpy(float) for _, group in frame.groupby("month", sort=True)]
    if not blocks:
        raise ValueError("cannot bootstrap an empty period")
    samples = np.empty(replicates, dtype=float)
    for index in range(replicates):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        samples[index] = np.concatenate([blocks[item] for item in chosen]).mean() * FROZEN_DIRECTION
    ci = np.quantile(samples, [0.025, 0.975])
    return {
        **_metrics(frame["rank_ic"]),
        "month_block_count": len(blocks),
        "bootstrap_replicates": replicates,
        "ci95": [float(ci[0]), float(ci[1])],
        "probability_signed_mean_positive": float((samples > 0).mean()),
        "probability_signed_mean_at_least_0_02": float((samples >= MIN_SIGNED_MEAN).mean()),
    }


def regime_metrics(frame: pd.DataFrame, states: pd.DataFrame) -> dict[str, object]:
    joined = frame.merge(states, on="signal_date", how="left", validate="one_to_one")
    result: dict[str, object] = {"trend": {}, "volatility": {}}
    for dimension in ("trend_regime", "volatility_regime"):
        key = dimension.removesuffix("_regime")
        for regime, group in joined.dropna(subset=[dimension]).groupby(dimension, sort=True):
            result[key][str(regime)] = _metrics(group["rank_ic"])
    return result


def audit(
    rank_ic_path: Path, cache_dir: Path, replicates: int = DEFAULT_BOOTSTRAP_REPLICATES
) -> dict[str, object]:
    daily = _read_rank_ic(rank_ic_path)
    states = load_market_states(
        cache_dir, daily["signal_date"].min() - pd.Timedelta(days=45), daily["signal_date"].max()
    )
    train = daily.loc[daily["signal_date"].dt.year <= 2024].copy()
    validation = daily.loc[daily["signal_date"].dt.year == 2025].copy()
    rng = np.random.default_rng(SEED)
    train_regime = regime_metrics(train, states)
    validation_regime = regime_metrics(validation, states)
    regime_cells = [
        *train_regime["trend"].values(),
        *train_regime["volatility"].values(),
        *validation_regime["trend"].values(),
        *validation_regime["volatility"].values(),
    ]
    regime_pass = all(
        float(cell["signed_mean_rank_ic"]) >= MIN_SIGNED_MEAN
        and float(cell["signed_icir"]) >= MIN_ICIR
        for cell in regime_cells
    )
    bootstrap = {
        "train_2020_2024": month_block_bootstrap(train, replicates, rng),
        "validation_2025": month_block_bootstrap(validation, replicates, rng),
    }
    bootstrap_pass = all(float(row["ci95"][0]) >= MIN_SIGNED_MEAN for row in bootstrap.values())
    leave = {
        "train_2020_2024": leave_one_month(train),
        "validation_2025": leave_one_month(validation),
    }
    leave_pass = all(row["status"] == "PASS" for row in leave.values())
    return {
        "scope": "ALPHA206_MEDIAN_ABS_ROBUSTNESS_AUDIT_V1",
        "candidate_id": CANDIDATE,
        "source_factor_id": SOURCE_FACTOR_ID,
        "market_scope": "SH/SZ",
        "excluded_exchanges": ["BJ"],
        "label": "forward_open_5d:v1 raw five-trading-day forward return",
        "method": (
            "Leave-one-calendar-month stress, SH/SZ equal-weight market "
            "trend/volatility states, and calendar-month block bootstrap."
        ),
        "seed": SEED,
        "bootstrap_replicates": replicates,
        "thresholds": {
            "minimum_signed_mean_rank_ic": MIN_SIGNED_MEAN,
            "minimum_signed_icir": MIN_ICIR,
        },
        "train_2020_2024": {
            "leave_one_month": leave["train_2020_2024"],
            "regime": train_regime,
            "month_block_bootstrap": bootstrap["train_2020_2024"],
        },
        "validation_2025": {
            "leave_one_month": leave["validation_2025"],
            "regime": validation_regime,
            "month_block_bootstrap": bootstrap["validation_2025"],
        },
        "decision": "ROBUSTNESS_PASS_PENDING_EXPLICIT_ADMISSION"
        if leave_pass and regime_pass and bootstrap_pass
        else "ROBUSTNESS_MONITOR_OR_FAIL_PENDING_REVIEW",
        "checks": {
            "leave_one_month_pass": leave_pass,
            "regime_pass": regime_pass,
            "month_block_bootstrap_pass": bootstrap_pass,
        },
        "database_writes": False,
        "locked_test_read": False,
        "publication_status": "NOT_PUBLISHED",
        "formal_factor_set_unchanged": True,
        "model_training_started": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-ic", type=Path, default=DEFAULT_RANK_IC)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.replicates < 100:
        parser.error("replicates must be at least 100")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit(args.rank_ic, args.cache_dir, args.replicates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"decision": result["decision"], "output": str(args.output)}, ensure_ascii=False)
    )
    return 0 if result["decision"] == "ROBUSTNESS_PASS_PENDING_EXPLICIT_ADMISSION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
