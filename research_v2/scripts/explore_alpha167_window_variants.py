"""Explore rolling-covariance window variants for Alpha167 (Alpha99).

The canonical window is 5; the annual strict gate fails in 2021 (0.349) and
2023 (0.470), both close to the 0.50 threshold. This read-only exploration
screens windows 3/4/5/6/7/10 over the 2020-2024 training window against the
frozen label to see whether any window passes the >= 4/5 annual gate. No
database writes, no locked-test reads, no admission decisions.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_alpha167_canonical_quality import (
    FROZEN_DIRECTION,
    LOCKED_TEST_START,
    WARMUP_CALENDAR_DAYS,
    _connect,
    _periods,
    compute_alpha99,
    compute_daily_rank_ic,
    load_cached_market,
    load_period_inputs,
    summarize_daily,
)

WINDOWS = (3, 4, 5, 6, 7, 10)
DEFAULT_CACHE_DIR = Path("research_v2/artifacts/tushare_cache")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2024, 12, 31))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("start date must be before end date")
    if args.end >= LOCKED_TEST_START:
        parser.error("evaluation range must be before 2026 locked-test")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    client = _connect()
    # cache market once per period, reuse across windows
    period_markets = []
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
        period_markets.append((market, labels))

    results = {}
    for window in WINDOWS:
        daily_parts = []
        for market, labels in period_markets:
            factor = compute_alpha99(market, window=window)
            daily_parts.append(compute_daily_rank_ic(factor, labels))
            del factor
        daily = pd.concat(daily_parts, ignore_index=True).sort_values("signal_date", kind="mergesort")
        train = daily.loc[pd.to_datetime(daily["signal_date"]).dt.year <= 2024]
        annual = {}
        for year in range(2020, 2025):
            year_daily = train.loc[pd.to_datetime(train["signal_date"]).dt.year == year]
            annual[str(year)] = summarize_daily(year_daily)
        summary = summarize_daily(train)
        signed_icir = summary["icir"] * FROZEN_DIRECTION
        strict_pass = int(sum(abs(row["icir"]) >= 0.5 for row in annual.values()))
        annual_icir = {yr: round(row["icir"], 4) for yr, row in annual.items()}
        results[f"window_{window}"] = {
            "window": window,
            "train_signed_mean_rank_ic": summary["signed_mean_rank_ic"],
            "train_signed_icir": signed_icir,
            "directional_day_share": summary["directional_day_share"],
            "strict_annual_icir_pass_years": strict_pass,
            "annual_icir": annual_icir,
        }
        print(
            f"window {window}: signed IC {summary['signed_mean_rank_ic']:.4f} "
            f"ICIR {signed_icir:.4f} annual {strict_pass}/5 "
            f"[{', '.join(f'{k[2:]}:{v:.2f}' for k, v in annual_icir.items())}]",
            flush=True,
        )

    payload = {
        "scope": "ALPHA167_WINDOW_VARIANT_EXPLORATION_V1",
        "factor_id": "yeslab_167",
        "factor_name": "Alpha99",
        "formula": "-cross-sectional rank of rolling covariance between rank(close) and rank(vol), window variant",
        "frozen_direction": FROZEN_DIRECTION,
        "market_scope": "SH/SZ",
        "excluded_exchanges": ["BJ"],
        "market_source": "local Tushare.daily cache",
        "label": "forward_open_5d:v1",
        "training_window": "2020-01-02/2024-12-31",
        "windows_screened": list(WINDOWS),
        "results": results,
        "database_writes": False,
        "locked_test_read": False,
        "formal_factor_set_changed": False,
        "model_training_started": False,
    }
    (args.output / "window_variant_exploration.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output / 'window_variant_exploration.json'}")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
