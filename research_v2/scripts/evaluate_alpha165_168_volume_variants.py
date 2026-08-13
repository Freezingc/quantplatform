"""Evaluate bounded volume-scale repairs for Alpha97 and Alpha100.

This read-only experiment keeps each factor's lookback window and negative
direction fixed while comparing four volume representations: raw rolling
standard deviation, log-volume rolling standard deviation, rolling volume
coefficient of variation, and cross-sectional rank of rolling standard
deviation.  It uses the canonical V2 SH/SZ panel and raw five-trading-day
forward adjusted-return label, and rejects any range touching the 2026 locked
test set.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_alpha165_168_canonical_quality import (
    LOCKED_TEST_START,
    WARMUP_CALENDAR_DAYS,
    _periods,
    compute_daily_rank_ic,
    load_cached_market,
    load_period_inputs,
    summarize_daily,
    summarize_monthly,
    _connect,
)

DEFAULT_CACHE_DIR = Path("research_v2/artifacts/tushare_cache")
VARIANTS = ("raw_volume_std", "log_volume_std", "volume_cv", "rank_volume_std")
WINDOWS = {"yeslab_165": 10, "yeslab_168": 20}
FACTOR_NAMES = {"yeslab_165": "Alpha97", "yeslab_168": "Alpha100"}


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


def compute_variants(market: pd.DataFrame, window: int) -> dict[str, pd.DataFrame]:
    required = {"trade_date", "ts_code", "vol"}
    if missing := sorted(required.difference(market.columns)):
        raise ValueError(f"market frame is missing columns: {missing}")
    volume = market.pivot(index="trade_date", columns="ts_code", values="vol").sort_index()
    raw_std = volume.rolling(window=window, min_periods=window).std(ddof=1)
    log_std = np.log1p(volume).rolling(window=window, min_periods=window).std(ddof=1)
    mean_volume = volume.rolling(window=window, min_periods=window).mean()
    cv = raw_std.div(mean_volume.replace(0.0, np.nan))
    rank_std = raw_std.rank(axis=1, pct=True)
    variants = {
        "raw_volume_std": raw_std,
        "log_volume_std": log_std,
        "volume_cv": cv,
        "rank_volume_std": rank_std,
    }
    return {name: frame.replace([np.inf, -np.inf], np.nan) for name, frame in variants.items()}


def _summary(daily: pd.DataFrame, factor_id: str, variant: str, window: int) -> dict[str, object]:
    daily = daily.sort_values("signal_date", kind="mergesort")
    train = daily.loc[pd.to_datetime(daily["signal_date"]).dt.year <= 2024]
    validation = daily.loc[pd.to_datetime(daily["signal_date"]).dt.year == 2025]
    annual: dict[str, dict[str, object]] = {}
    for year in range(2020, 2025):
        annual[str(year)] = summarize_daily(train.loc[pd.to_datetime(train["signal_date"]).dt.year == year])
    train_month = summarize_monthly(train)
    validation_month = summarize_monthly(validation)
    return {
        "factor_id": factor_id,
        "factor_name": FACTOR_NAMES[factor_id],
        "variant": variant,
        "window": window,
        "train_2020_2024": summarize_daily(train),
        "validation_2025": summarize_daily(validation),
        "annual": annual,
        "strict_annual_icir_pass_years": int(sum(abs(row["icir"]) >= 0.5 for row in annual.values())),
        "train_monthly_signed_mean_pass_months": int((train_month["signed_mean_rank_ic"] >= 0.02).sum()),
        "validation_monthly_signed_mean_pass_months": int((validation_month["signed_mean_rank_ic"] >= 0.02).sum()),
        "train_monthly_directional_share_pass_months": int((train_month["directional_day_share"] >= 0.55).sum()),
        "validation_monthly_directional_share_pass_months": int((validation_month["directional_day_share"] >= 0.55).sum()),
    }


def run_evaluation(args: argparse.Namespace, *, client: object | None = None) -> dict[str, object]:
    own_client = client is None
    client = client or _connect()
    args.output.mkdir(parents=True, exist_ok=True)
    daily_parts: dict[tuple[str, str], list[pd.DataFrame]] = {
        (factor_id, variant): [] for factor_id in WINDOWS for variant in VARIANTS
    }
    market_rows = label_rows = 0
    for period in _periods(args.start, args.end):
        period_start = pd.Period(period, freq="M").start_time.date()
        period_end = pd.Period(period, freq="M").end_time.date()
        labels, panel_codes = load_period_inputs(client, period)
        market = load_cached_market(
            args.cache_dir,
            period_start - timedelta(days=WARMUP_CALENDAR_DAYS),
            period_end,
            panel_codes,
        )
        for factor_id, window in WINDOWS.items():
            for variant, panel in compute_variants(market, window).items():
                daily_parts[(factor_id, variant)].append(compute_daily_rank_ic(panel, labels))
        market_rows += len(market)
        label_rows += len(labels)
        del market, labels

    payload: dict[str, object] = {
        "scope": "ALPHA165_168_VOLUME_VARIANTS_CANONICAL_TUSHARE_QUALITY_V1",
        "factor_ids": list(WINDOWS),
        "factor_names": FACTOR_NAMES,
        "market_scope": "SH/SZ",
        "excluded_exchanges": ["BJ"],
        "market_source": "local Tushare.daily cache",
        "label": "forward_open_5d:v1 raw five-trading-day forward return",
        "frozen_direction": -1,
        "windows": WINDOWS,
        "variants": list(VARIANTS),
        "database_writes": False,
        "locked_test_read": False,
        "market_rows": int(market_rows),
        "label_rows": int(label_rows),
        "factors": {},
    }
    for (factor_id, variant), parts in daily_parts.items():
        daily = pd.concat(parts, ignore_index=True).sort_values("signal_date", kind="mergesort")
        output_dir = args.output / factor_id / variant
        output_dir.mkdir(parents=True, exist_ok=True)
        daily.to_csv(output_dir / "daily_rank_ic.csv", index=False)
        monthly = summarize_monthly(daily)
        monthly.to_csv(output_dir / "monthly_rank_ic.csv", index=False)
        summary = _summary(daily, factor_id, variant, WINDOWS[factor_id])
        summary["artifact"] = f"{factor_id}/{variant}/quality_summary.json"
        (output_dir / "quality_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        payload["factors"].setdefault(factor_id, {})[variant] = summary
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
