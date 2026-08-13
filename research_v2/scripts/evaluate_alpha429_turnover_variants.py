"""Evaluate bounded turnover representations for Alpha429/GP_060.

The frozen calculator is ``STD_10(MAX_10(pct_chg)) + turnover_rate``.  This
read-only experiment keeps the return-volatility term, windows, SH/SZ V2
universe, negative direction, and raw five-trading-day label fixed.  It only
changes the turnover component among a small, predeclared set of scale and
stationarity transforms.  All market rows come from the local Tushare
``daily`` and ``daily_basic`` caches; the range cannot include 2026 locked
test data and no DolphinDB writes are performed.
"""

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
FROZEN_DIRECTION = -1
FACTOR_ID = "yeslab_429"
FACTOR_NAME = "GP_060"
VARIANTS = ("raw_turnover", "log_turnover", "demeaned_turnover", "zscore_turnover", "rank_turnover")
RETURN_WINDOW = 10
TURNOVER_WINDOW = 20


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
    if not paths:
        raise FileNotFoundError(f"no Tushare daily cache files in {start}..{end}")
    return paths


def load_cached_market(cache_dir: Path, start: date, end: date, codes: Sequence[str]) -> pd.DataFrame:
    """Load and join same-date daily and daily_basic cache rows."""

    wanted = {str(code) for code in codes}
    frames: list[pd.DataFrame] = []
    for daily_path in _cache_dates(cache_dir, start, end):
        basic_path = cache_dir / daily_path.name.replace("daily_", "daily_basic_", 1)
        if not basic_path.is_file():
            continue
        date_token = daily_path.stem.removeprefix("daily_")
        trade_date = f"{date_token[:4]}{date_token[4:6]}{date_token[6:8]}"
        daily = pd.read_json(daily_path, orient="table")
        basic = pd.read_json(basic_path, orient="table")
        daily["ts_code"] = daily["ts_code"].astype(str)
        basic["ts_code"] = basic["ts_code"].astype(str)
        if "trade_date" not in basic.columns:
            basic["trade_date"] = trade_date
        daily = daily.loc[daily["ts_code"].isin(wanted)]
        basic = basic.loc[basic["ts_code"].isin(wanted)]
        if daily.empty or basic.empty:
            continue
        basic = basic.loc[:, ["ts_code", "trade_date", "turnover_rate"]]
        frame = daily.merge(basic, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
        frames.append(frame)
    if not frames:
        raise ValueError("Tushare cache has no daily/daily_basic overlap with V2 panel codes")
    return normalize_tushare_market(
        pd.concat(frames, ignore_index=True),
        codes=tuple(sorted(wanted)),
        include_daily_basic=True,
    )


def compute_variants(market: pd.DataFrame) -> dict[str, pd.DataFrame]:
    required = {"trade_date", "ts_code", "pct_chg", "turnover_rate"}
    if missing := sorted(required.difference(market.columns)):
        raise ValueError(f"market frame is missing columns: {missing}")
    pct_chg = market.pivot(index="trade_date", columns="ts_code", values="pct_chg").sort_index()
    turnover = market.pivot(index="trade_date", columns="ts_code", values="turnover_rate").reindex(index=pct_chg.index, columns=pct_chg.columns)
    return_term = pct_chg.rolling(RETURN_WINDOW, min_periods=RETURN_WINDOW).max().rolling(RETURN_WINDOW, min_periods=RETURN_WINDOW).std(ddof=1)
    raw = turnover
    log = np.log1p(turnover.clip(lower=0.0))
    mean = turnover.rolling(TURNOVER_WINDOW, min_periods=TURNOVER_WINDOW).mean()
    std = turnover.rolling(TURNOVER_WINDOW, min_periods=TURNOVER_WINDOW).std(ddof=1)
    representations = {
        "raw_turnover": raw,
        "log_turnover": log,
        "demeaned_turnover": turnover - mean,
        "zscore_turnover": (turnover - mean).div(std.replace(0.0, np.nan)),
        "rank_turnover": turnover.rank(axis=1, pct=True),
    }
    factors = {name: return_term + value for name, value in representations.items()}
    return {name: frame.replace([np.inf, -np.inf], np.nan) for name, frame in factors.items()}


def summarize_variant(daily: pd.DataFrame, variant: str) -> dict[str, object]:
    daily = daily.sort_values("signal_date", kind="mergesort")
    train = daily.loc[pd.to_datetime(daily["signal_date"]).dt.year <= 2024]
    validation = daily.loc[pd.to_datetime(daily["signal_date"]).dt.year == 2025]
    annual: dict[str, dict[str, object]] = {}
    for year in range(2020, 2025):
        annual[str(year)] = summarize_daily(train.loc[pd.to_datetime(train["signal_date"]).dt.year == year])
    train_month = summarize_monthly(train)
    validation_month = summarize_monthly(validation)
    return {
        "factor_id": FACTOR_ID,
        "factor_name": FACTOR_NAME,
        "variant": variant,
        "return_window": RETURN_WINDOW,
        "turnover_window": TURNOVER_WINDOW,
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
    daily_parts: dict[str, list[pd.DataFrame]] = {variant: [] for variant in VARIANTS}
    market_rows = label_rows = 0
    for period in _periods(args.start, args.end):
        period_start = pd.Period(period, freq="M").start_time.date()
        period_end = pd.Period(period, freq="M").end_time.date()
        labels, panel_codes = load_period_inputs(client, period)
        market = load_cached_market(args.cache_dir, period_start - timedelta(days=WARMUP_CALENDAR_DAYS), period_end, panel_codes)
        for variant, panel in compute_variants(market).items():
            daily_parts[variant].append(compute_daily_rank_ic(panel, labels))
        market_rows += len(market)
        label_rows += len(labels)
        del market, labels
    payload: dict[str, object] = {
        "scope": "ALPHA429_TURNOVER_VARIANTS_CANONICAL_TUSHARE_QUALITY_V1",
        "factor_id": FACTOR_ID,
        "factor_name": FACTOR_NAME,
        "market_scope": "SH/SZ",
        "excluded_exchanges": ["BJ"],
        "market_source": "local Tushare.daily + daily_basic cache",
        "label": "forward_open_5d:v1 raw five-trading-day forward return",
        "frozen_direction": FROZEN_DIRECTION,
        "return_window": RETURN_WINDOW,
        "turnover_window": TURNOVER_WINDOW,
        "variants": list(VARIANTS),
        "database_writes": False,
        "locked_test_read": False,
        "market_rows": int(market_rows),
        "label_rows": int(label_rows),
        "factors": {},
    }
    for variant, parts in daily_parts.items():
        daily = pd.concat(parts, ignore_index=True).sort_values("signal_date", kind="mergesort")
        output_dir = args.output / variant
        output_dir.mkdir(parents=True, exist_ok=True)
        daily.to_csv(output_dir / "daily_rank_ic.csv", index=False)
        summarize_monthly(daily).to_csv(output_dir / "monthly_rank_ic.csv", index=False)
        summary = summarize_variant(daily, variant)
        summary["artifact"] = f"{variant}/quality_summary.json"
        (output_dir / "quality_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        payload["factors"][variant] = summary
    (args.output / "quality_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    if own_client:
        client.close()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(run_evaluation(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
