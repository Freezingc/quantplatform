"""Evaluate bounded repairs for Alpha206/60-day volume demean deviation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from audit_formula_replay_tushare import normalize_tushare_market
from evaluate_alpha165_168_canonical_quality import (
    LOCKED_TEST_START,
    WARMUP_CALENDAR_DAYS,
    _connect,
    _periods,
    compute_daily_rank_ic,
    load_period_inputs,
    summarize_daily,
    summarize_monthly,
)

DEFAULT_CACHE_DIR = Path("research_v2/artifacts/tushare_cache")
FACTOR_ID = "yeslab_206"
FACTOR_NAME = "60日成交量去均值相对偏离"
FROZEN_DIRECTION = -1
WINDOW = 60
VARIANTS = ("raw_sma_abs", "log_sma_abs", "median_abs", "raw_signed")


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


def _cache_dates(cache_dir: Path, start: date, end: date) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(cache_dir.glob("daily_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].json")):
        token = path.stem.removeprefix("daily_")
        try:
            current = date.fromisoformat(f"{token[:4]}-{token[4:6]}-{token[6:8]}")
        except ValueError:
            continue
        if start <= current <= end:
            paths.append(path)
    return paths


def load_cached_market(cache_dir: Path, start: date, end: date, codes: Sequence[str]) -> pd.DataFrame:
    wanted = {str(code) for code in codes}
    frames: list[pd.DataFrame] = []
    for path in _cache_dates(cache_dir, start, end):
        frame = pd.read_json(path, orient="table")
        frame["ts_code"] = frame["ts_code"].astype(str)
        frame = frame.loc[frame["ts_code"].isin(wanted)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError("Tushare cache has no overlap with V2 panel codes")
    return normalize_tushare_market(pd.concat(frames, ignore_index=True), codes=tuple(sorted(wanted)), include_daily_basic=False)


def compute_variants(market: pd.DataFrame) -> dict[str, pd.DataFrame]:
    required = {"trade_date", "ts_code", "vol"}
    if missing := sorted(required.difference(market.columns)):
        raise ValueError(f"market frame is missing columns: {missing}")
    volume = market.pivot(index="trade_date", columns="ts_code", values="vol").sort_index()
    mean = volume.rolling(WINDOW, min_periods=WINDOW).mean()
    raw_abs = (volume - mean).abs().div(mean.replace(0.0, np.nan))
    log_volume = np.log1p(volume.clip(lower=0.0))
    log_mean = log_volume.rolling(WINDOW, min_periods=WINDOW).mean()
    log_abs = (log_volume - log_mean).abs().div(log_mean.abs().replace(0.0, np.nan))
    median = volume.rolling(WINDOW, min_periods=WINDOW).median()
    median_abs = (volume - median).abs().div(median.replace(0.0, np.nan))
    raw_signed = (volume - mean).div(mean.replace(0.0, np.nan))
    variants = {"raw_sma_abs": raw_abs, "log_sma_abs": log_abs, "median_abs": median_abs, "raw_signed": raw_signed}
    return {name: frame.replace([np.inf, -np.inf], np.nan) for name, frame in variants.items()}


def summarize_variant(daily: pd.DataFrame, variant: str) -> dict[str, object]:
    daily = daily.sort_values("signal_date", kind="mergesort")
    train = daily.loc[pd.to_datetime(daily["signal_date"]).dt.year <= 2024]
    validation = daily.loc[pd.to_datetime(daily["signal_date"]).dt.year == 2025]
    annual = {str(year): summarize_daily(train.loc[pd.to_datetime(train["signal_date"]).dt.year == year]) for year in range(2020, 2025)}
    train_month = summarize_monthly(train)
    validation_month = summarize_monthly(validation)
    return {
        "factor_id": FACTOR_ID,
        "factor_name": FACTOR_NAME,
        "variant": variant,
        "window": WINDOW,
        "frozen_direction": FROZEN_DIRECTION,
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
    parts: dict[str, list[pd.DataFrame]] = {variant: [] for variant in VARIANTS}
    market_rows = label_rows = 0
    for period in _periods(args.start, args.end):
        period_start = pd.Period(period, freq="M").start_time.date()
        period_end = pd.Period(period, freq="M").end_time.date()
        labels, codes = load_period_inputs(client, period)
        market = load_cached_market(args.cache_dir, period_start - timedelta(days=WARMUP_CALENDAR_DAYS), period_end, codes)
        for variant, panel in compute_variants(market).items():
            parts[variant].append(compute_daily_rank_ic(panel, labels))
        market_rows += len(market)
        label_rows += len(labels)
        del market, labels
    payload: dict[str, object] = {
        "scope": "ALPHA206_VOLUME_VARIANTS_CANONICAL_TUSHARE_QUALITY_V1",
        "factor_id": FACTOR_ID,
        "factor_name": FACTOR_NAME,
        "market_scope": "SH/SZ",
        "excluded_exchanges": ["BJ"],
        "market_source": "local Tushare.daily cache",
        "label": "forward_open_5d:v1 raw five-trading-day forward return",
        "frozen_direction": FROZEN_DIRECTION,
        "window": WINDOW,
        "variants": list(VARIANTS),
        "database_writes": False,
        "locked_test_read": False,
        "market_rows": int(market_rows),
        "label_rows": int(label_rows),
        "factors": {},
    }
    for variant, daily_parts in parts.items():
        daily = pd.concat(daily_parts, ignore_index=True).sort_values("signal_date", kind="mergesort")
        out = args.output / variant
        out.mkdir(parents=True, exist_ok=True)
        daily.to_csv(out / "daily_rank_ic.csv", index=False)
        summarize_monthly(daily).to_csv(out / "monthly_rank_ic.csv", index=False)
        summary = summarize_variant(daily, variant)
        summary["artifact"] = f"{variant}/quality_summary.json"
        (out / "quality_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        payload["factors"][variant] = summary
    (args.output / "quality_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    if own_client:
        client.close()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(run_evaluation(parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
