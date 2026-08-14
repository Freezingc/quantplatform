"""Evaluate the isolated Tushare SH/SZ canonical Alpha484 candidate."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

CANDIDATE_ID = "yeslab_484_tushare_shsz_v1"
SOURCE_FACTOR_ID = "yeslab_484"
TRAIN_START = date(2020, 1, 2)
TRAIN_END = date(2024, 12, 31)
VALIDATION_START = date(2025, 1, 2)
VALIDATION_END = date(2025, 12, 31)
LOCKED_TEST_START = date(2026, 1, 1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse a bounded Alpha484 canonical evaluation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=TRAIN_START)
    parser.add_argument("--end", type=date.fromisoformat, default=VALIDATION_END)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("research_v2/artifacts/tushare_cache")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.start > args.end or args.end >= LOCKED_TEST_START:
        parser.error("evaluation range must end before the 2026 locked test")
    return args


def validate_full_market_dates(
    market: pd.DataFrame, expected_dates: tuple[date, ...]
) -> None:
    """Require one complete canonical SH/SZ market result per open date."""
    if "trade_date" not in market.columns:
        raise ValueError("full SH/SZ market is missing trade_date")
    observed = set(pd.to_datetime(market["trade_date"], errors="raise").dt.date)
    missing = sorted(set(expected_dates).difference(observed))
    if missing:
        rendered = ", ".join(value.isoformat() for value in missing)
        raise ValueError(f"full SH/SZ market is missing open dates: {rendered}")
