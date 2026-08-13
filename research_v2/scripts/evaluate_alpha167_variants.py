"""Evaluate one bounded Alpha167/Alpha99 repair against the frozen label.

The baseline is the canonical Alpha99 rolling covariance.  The only repair
variant is rolling correlation, which removes the rolling scale component while
keeping the five-day window, cross-sectional rank inputs, direction, universe,
and market source unchanged.  This is read-only and excludes 2026.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_alpha167_canonical_quality import (
    LOCKED_TEST_START,
    WARMUP_CALENDAR_DAYS,
    _connect,
    _periods,
    compute_daily_rank_ic,
    load_cached_market,
    load_period_inputs,
    summarize_daily,
    summarize_monthly,
)

FACTOR_ID = "yeslab_167"
FACTOR_NAME = "Alpha99"
FROZEN_DIRECTION = 1
WINDOW = 5
VARIANTS = ("raw_covariance", "rolling_correlation")
DEFAULT_CACHE_DIR = Path("research_v2/artifacts/tushare_cache")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("start date must not be after end date")
    if args.end >= LOCKED_TEST_START:
        parser.error("evaluation range must be before 2026 locked-test")
    return args


def compute_variants(market: pd.DataFrame, *, window: int = WINDOW) -> dict[str, pd.DataFrame]:
    required = {"trade_date", "ts_code", "close", "vol"}
    if missing := sorted(required.difference(market.columns)):
        raise ValueError(f"market frame is missing columns: {missing}")
    close = market.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    volume = market.pivot(index="trade_date", columns="ts_code", values="vol").sort_index()
    rank_close = close.rank(axis=1, pct=True)
    rank_volume = volume.rank(axis=1, pct=True)
    covariance = rank_close.rolling(window=window, min_periods=window).cov(rank_volume)
    close_std = rank_close.rolling(window=window, min_periods=window).std(ddof=1)
    volume_std = rank_volume.rolling(window=window, min_periods=window).std(ddof=1)
    correlation = covariance.div((close_std * volume_std).replace(0.0, np.nan))
    variants = {
        "raw_covariance": -covariance.rank(axis=1, pct=True),
        "rolling_correlation": -correlation.rank(axis=1, pct=True),
    }
    return {name: frame.replace([np.inf, -np.inf], np.nan) for name, frame in variants.items()}


def summarize_variant(daily: pd.DataFrame, variant: str) -> dict[str, object]:
    daily = daily.sort_values("signal_date", kind="mergesort")
    years = pd.to_datetime(daily["signal_date"]).dt.year
    train = daily.loc[years <= 2024]
    validation = daily.loc[years == 2025]
    annual = {
        str(year): summarize_daily(train.loc[pd.to_datetime(train["signal_date"]).dt.year == year])
        for year in range(2020, 2025)
    }
    train_monthly = summarize_monthly(train)
    validation_monthly = summarize_monthly(validation)
    return {
        "factor_id": FACTOR_ID,
        "factor_name": FACTOR_NAME,
        "variant": variant,
        "window": WINDOW,
        "frozen_direction": FROZEN_DIRECTION,
        "train_2020_2024": summarize_daily(train),
        "validation_2025": summarize_daily(validation),
        "annual": annual,
        "strict_annual_icir_pass_years": int(
            sum(abs(row["icir"]) >= 0.5 for row in annual.values())
        ),
        "train_monthly_signed_mean_pass_months": int(
            (train_monthly["signed_mean_rank_ic"] >= 0.02).sum()
        ),
        "validation_monthly_signed_mean_pass_months": int(
            (validation_monthly["signed_mean_rank_ic"] >= 0.02).sum()
        ),
        "train_monthly_directional_share_pass_months": int(
            (train_monthly["directional_day_share"] >= 0.55).sum()
        ),
        "validation_monthly_directional_share_pass_months": int(
            (validation_monthly["directional_day_share"] >= 0.55).sum()
        ),
    }


def run_evaluation(args: argparse.Namespace, *, client: object | None = None) -> dict[str, object]:
    own_client = client is None
    client = client or _connect()
    args.output.mkdir(parents=True, exist_ok=True)
    daily_parts: dict[str, list[pd.DataFrame]] = {variant: [] for variant in VARIANTS}
    market_rows = label_rows = 0
    for period in _periods(args.start, args.end):
        period_start = pd.Period(period, freq="M").start_time.date()
        period_end = pd.Period(period, freq="M").end_time.date()
        labels, codes = load_period_inputs(client, period)
        market = load_cached_market(
            args.cache_dir,
            period_start - timedelta(days=WARMUP_CALENDAR_DAYS),
            period_end,
            codes,
        )
        for variant, panel in compute_variants(market).items():
            daily_parts[variant].append(compute_daily_rank_ic(panel, labels))
        market_rows += len(market)
        label_rows += len(labels)
        del market, labels

    payload: dict[str, object] = {
        "scope": "ALPHA167_BOUNDED_VARIANTS_CANONICAL_TUSHARE_QUALITY_V1",
        "factor_id": FACTOR_ID,
        "factor_name": FACTOR_NAME,
        "market_scope": "SH/SZ",
        "excluded_exchanges": ["BJ"],
        "market_source": "local Tushare.daily cache",
        "label": "forward_open_5d:v1 raw five-trading-day forward return",
        "frozen_direction": FROZEN_DIRECTION,
        "window": WINDOW,
        "variants": list(VARIANTS),
        "repair_contract": (
            "Only covariance-to-correlation substitution; no window, rank, "
            "direction, universe, or label changes."
        ),
        "database_writes": False,
        "locked_test_read": False,
        "market_rows": int(market_rows),
        "label_rows": int(label_rows),
        "factors": {},
    }
    for variant, parts in daily_parts.items():
        daily = pd.concat(parts, ignore_index=True).sort_values("signal_date", kind="mergesort")
        output = args.output / variant
        output.mkdir(parents=True, exist_ok=True)
        daily.to_csv(output / "daily_rank_ic.csv", index=False)
        summarize_monthly(daily).to_csv(output / "monthly_rank_ic.csv", index=False)
        summary = summarize_variant(daily, variant)
        summary["artifact"] = f"{variant}/quality_summary.json"
        (output / "quality_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        payload["factors"][variant] = summary
    (args.output / "quality_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if own_client:
        client.close()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(run_evaluation(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
