"""Define the read-only canonical candidate-pair comparison contract."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

TRAIN_START = date(2020, 1, 2)
TRAIN_END = date(2024, 12, 31)
VALIDATION_START = date(2025, 1, 2)
VALIDATION_END = date(2025, 12, 31)
LOCKED_TEST_START = date(2026, 1, 1)
TRAIN_EXPECTED_DAYS = 1212
VALIDATION_EXPECTED_DAYS = 243


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    source_factor_id: str
    is_alpha206_median_repair: bool = False


CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec("yeslab_114_tushare_shsz_v2", "yeslab_114"),
    CandidateSpec("yeslab_381_tushare_shsz_v1", "yeslab_381"),
    CandidateSpec("yeslab_405_tushare_shsz_v1", "yeslab_405"),
    CandidateSpec("yeslab_425_tushare_shsz_v2", "yeslab_425"),
    CandidateSpec("yeslab_484_tushare_shsz_v1", "yeslab_484"),
    CandidateSpec("yeslab_206_median_abs_v2", "yeslab_206", is_alpha206_median_repair=True),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse one read-only comparison range that cannot reach the locked test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=TRAIN_START)
    parser.add_argument("--end", type=date.fromisoformat, default=VALIDATION_END)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("start date must not be after end date")
    if args.end >= LOCKED_TEST_START:
        parser.error("comparison range must end before the 2026 locked test")
    return args


def classify_pair(
    median_train: float,
    days_train: int,
    median_validation: float,
    days_validation: int,
) -> str:
    """Classify one pair only when both fixed comparison phases are complete."""
    if days_train != TRAIN_EXPECTED_DAYS or days_validation != VALIDATION_EXPECTED_DAYS:
        return "INCOMPLETE"
    maximum = max(median_train, median_validation)
    if maximum >= 0.90:
        return "REDUNDANT"
    if maximum >= 0.80:
        return "ELEVATED_PAIR_REDUNDANCY_MONITOR"
    return "PAIR_UNIQUE"
