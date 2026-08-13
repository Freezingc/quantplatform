"""Audit market-regime stability for the four formal monitored factors.

This read-only audit reuses canonical SH/SZ Tushare daily RankIC artifacts and
the local daily market cache. It never reads 2026 and does not write a factor
value or alter the formal factor set.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

LOCKED_TEST_START = pd.Timestamp("2026-01-01")
MIN_SIGNED_MEAN = 0.02
MIN_ICIR = 0.50
FORMAL_MONITORED = ("yeslab_380", "yeslab_431", "yeslab_437", "yeslab_94")
DEFAULT_CACHE_DIR = Path("research_v2/artifacts/tushare_cache")
DEFAULT_TRAIN_INPUTS = (
    Path(
        "research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_train_16_quality_v2/daily_rank_ic.csv"
    ),
    Path(
        "research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_train_canonical2_quality_v2/daily_rank_ic.csv"
    ),
    Path(
        "research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_train_legacy3_quality_v2/daily_rank_ic.csv"
    ),
)
DEFAULT_VALIDATION_INPUTS = (
    Path(
        "research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_validation_16_quality_v2/daily_rank_ic.csv"
    ),
    Path(
        "research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_validation_canonical2_quality_v2/daily_rank_ic.csv"
    ),
    Path(
        "research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_validation_legacy3_quality_v2/daily_rank_ic.csv"
    ),
)


def _metrics(values: pd.Series, direction: int = -1) -> dict[str, float | int]:
    values = pd.to_numeric(values, errors="coerce")
    values = values.loc[np.isfinite(values.to_numpy(float))].to_numpy(float)
    signed = values * direction
    std = float(signed.std(ddof=1)) if len(signed) > 1 else float("nan")
    return {
        "valid_day_count": int(len(signed)),
        "signed_mean_rank_ic": float(signed.mean()) if len(signed) else float("nan"),
        "signed_icir": float(signed.mean() / std) if np.isfinite(std) and std > 0 else float("nan"),
        "directional_day_share": float((signed > 0).mean()) if len(signed) else float("nan"),
    }


def load_rank_ic(paths: Sequence[Path], factor_ids: Sequence[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    wanted = set(factor_ids)
    for path in paths:
        frame = pd.read_csv(path)
        column = "factor_id" if "factor_id" in frame.columns else "factor_name"
        required = {column, "signal_date", "rank_ic"}
        if missing := sorted(required.difference(frame.columns)):
            raise ValueError(f"RankIC input {path} is missing columns: {missing}")
        frame = frame.loc[
            frame[column].astype(str).isin(wanted), [column, "signal_date", "rank_ic"]
        ].copy()
        frame = frame.rename(columns={column: "factor_id"})
        frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.normalize()
        frame["rank_ic"] = pd.to_numeric(frame["rank_ic"], errors="coerce")
        frames.append(frame)
    if not frames:
        raise ValueError("no RankIC inputs were provided")
    result = pd.concat(frames, ignore_index=True).dropna(subset=["rank_ic"])
    result = result.loc[result["signal_date"] < LOCKED_TEST_START]
    result = result.drop_duplicates(["factor_id", "signal_date"], keep="last")
    missing = sorted(wanted.difference(result["factor_id"].unique()))
    if missing:
        raise ValueError(f"RankIC inputs are missing factors: {missing}")
    return result.sort_values(["factor_id", "signal_date"], kind="mergesort").reset_index(drop=True)


def load_market_states(cache_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(cache_dir.glob("daily_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].json")):
        token = path.stem.removeprefix("daily_")
        try:
            day = pd.Timestamp(f"{token[:4]}-{token[4:6]}-{token[6:8]}")
        except ValueError:
            continue
        if not start <= day <= end:
            continue
        frame = pd.read_json(path, orient="table")
        if not {"ts_code", "trade_date", "pct_chg"}.issubset(frame.columns):
            continue
        frame["ts_code"] = frame["ts_code"].astype(str)
        frame = frame.loc[frame["ts_code"].str.endswith((".SH", ".SZ"))].copy()
        frame["pct_chg"] = pd.to_numeric(frame["pct_chg"], errors="coerce") / 100.0
        grouped = (
            frame.replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["pct_chg"])
            .groupby("trade_date")["pct_chg"]
            .mean()
        )
        if not grouped.empty:
            rows.append(grouped.rename("market_return").to_frame())
    if not rows:
        raise ValueError("market cache has no SH/SZ observations")
    result = pd.concat(rows).reset_index().rename(columns={"trade_date": "signal_date"})
    result["signal_date"] = pd.to_datetime(result["signal_date"], errors="raise").dt.normalize()
    result = (
        result.groupby("signal_date", as_index=False)["market_return"]
        .mean()
        .sort_values("signal_date")
    )
    result["trend_return_20d"] = (1 + result["market_return"]).rolling(20, min_periods=20).apply(
        np.prod, raw=True
    ) - 1
    result["volatility_20d"] = result["market_return"].rolling(20, min_periods=20).std()
    train_median = result.loc[
        result["signal_date"] < pd.Timestamp("2025-01-01"), "volatility_20d"
    ].median()
    result["trend_regime"] = np.where(result["trend_return_20d"] > 0, "positive", "nonpositive")
    result["volatility_regime"] = np.where(result["volatility_20d"] > train_median, "high", "low")
    return result


def audit(train: pd.DataFrame, validation: pd.DataFrame, states: pd.DataFrame) -> dict[str, object]:
    results: dict[str, object] = {}
    for factor_id in FORMAL_MONITORED:
        factor_result: dict[str, object] = {}
        for period, frame in (("train_2020_2024", train), ("validation_2025", validation)):
            current = frame.loc[frame["factor_id"] == factor_id].merge(
                states, on="signal_date", how="left", validate="one_to_one"
            )
            dimensions: dict[str, dict[str, dict[str, float | int]]] = {
                "trend": {},
                "volatility": {},
            }
            for source, key in (("trend_regime", "trend"), ("volatility_regime", "volatility")):
                for regime, group in current.dropna(subset=[source]).groupby(source, sort=True):
                    dimensions[key][str(regime)] = _metrics(group["rank_ic"])
            failed = [
                f"{dimension}:{regime}"
                for dimension, cells in dimensions.items()
                for regime, metric in cells.items()
                if float(metric["signed_mean_rank_ic"]) < MIN_SIGNED_MEAN
                or float(metric["signed_icir"]) < MIN_ICIR
            ]
            factor_result[period] = {"regime": dimensions, "failed_regimes": failed}
        results[factor_id] = factor_result
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--train-input", type=Path, action="append")
    parser.add_argument("--validation-input", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    train = load_rank_ic(tuple(args.train_input or DEFAULT_TRAIN_INPUTS), FORMAL_MONITORED)
    validation = load_rank_ic(
        tuple(args.validation_input or DEFAULT_VALIDATION_INPUTS), FORMAL_MONITORED
    )
    states = load_market_states(
        args.cache_dir,
        train["signal_date"].min() - pd.Timedelta(days=45),
        validation["signal_date"].max(),
    )
    result = {
        "scope": "FORMAL_MONITORED_FACTOR_REGIME_AUDIT_V1",
        "factor_ids": list(FORMAL_MONITORED),
        "market_scope": "SH/SZ",
        "excluded_exchanges": ["BJ"],
        "label": "forward_open_5d:v1 raw five-trading-day forward return",
        "thresholds": {
            "minimum_signed_mean_rank_ic": MIN_SIGNED_MEAN,
            "minimum_signed_icir": MIN_ICIR,
        },
        "results": audit(train, validation, states),
        "database_writes": False,
        "locked_test_read": False,
        "publication_status": "NOT_PUBLISHED",
        "formal_factor_set_unchanged": True,
        "model_training_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
