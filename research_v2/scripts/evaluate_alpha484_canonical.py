"""Evaluate the isolated Tushare SH/SZ canonical Alpha484 candidate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
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
MIN_SIGNED_RANK_IC = 0.02
MIN_SIGNED_ICIR = 0.50
MIN_DIRECTIONAL_SHARE = 0.55
MIN_ANNUAL_GATE_YEARS = 4
MIN_MEDIAN_COVERAGE = 0.80
MIN_P05_COVERAGE = 0.70
MIN_MONTHLY_COVERAGE = 0.70


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


def freeze_training_direction(daily_rank_ic: pd.DataFrame) -> int:
    """Freeze factor direction from finite training RankIC values only."""
    if "rank_ic" not in daily_rank_ic.columns:
        raise ValueError("training RankIC is missing rank_ic")
    values = pd.to_numeric(daily_rank_ic["rank_ic"], errors="coerce").dropna()
    if values.empty:
        raise ValueError("training RankIC has no finite observations")
    return -1 if float(values.mean()) < 0.0 else 1


def summarize_coverage(
    replay_values: pd.DataFrame, labels: pd.DataFrame
) -> dict[str, object]:
    """Measure finite candidate coverage on canonical valid-label keys."""
    value_required = {"trade_date", "ts_code", "replay_value"}
    label_required = {"signal_date", "ts_code"}
    if missing := sorted(value_required.difference(replay_values.columns)):
        raise ValueError(f"replay values are missing columns: {missing}")
    if missing := sorted(label_required.difference(labels.columns)):
        raise ValueError(f"labels are missing columns: {missing}")
    value_frame = replay_values.loc[:, ["trade_date", "ts_code", "replay_value"]].copy()
    value_frame["signal_date"] = pd.to_datetime(
        value_frame.pop("trade_date"), errors="raise"
    ).dt.normalize()
    value_frame["ts_code"] = value_frame["ts_code"].astype(str)
    value_frame["replay_value"] = pd.to_numeric(
        value_frame["replay_value"], errors="coerce"
    )
    if value_frame.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("replay values contain duplicate signal-date keys")
    label_frame = labels.loc[:, ["signal_date", "ts_code"]].copy()
    label_frame["signal_date"] = pd.to_datetime(
        label_frame["signal_date"], errors="raise"
    ).dt.normalize()
    label_frame["ts_code"] = label_frame["ts_code"].astype(str)
    if label_frame.empty:
        raise ValueError("labels contain no canonical keys")
    if label_frame.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("labels contain duplicate signal-date keys")
    merged = label_frame.merge(
        value_frame, on=["signal_date", "ts_code"], how="left", validate="one_to_one"
    )
    merged["is_finite"] = merged["replay_value"].notna()
    daily = (
        merged.groupby("signal_date", sort=True)["is_finite"]
        .agg(expected_key_count="size", finite_key_count="sum")
        .reset_index()
    )
    daily["coverage_ratio"] = (
        daily["finite_key_count"] / daily["expected_key_count"]
    )
    daily["month"] = daily["signal_date"].dt.to_period("M").astype(str)
    monthly = (
        daily.groupby("month", sort=True)[["expected_key_count", "finite_key_count"]]
        .sum()
        .reset_index()
    )
    monthly["coverage_ratio"] = (
        monthly["finite_key_count"] / monthly["expected_key_count"]
    )
    median_ratio = float(daily["coverage_ratio"].median())
    p05_ratio = float(daily["coverage_ratio"].quantile(0.05))
    monthly_min_ratio = float(monthly["coverage_ratio"].min())
    status = (
        "PASS"
        if median_ratio >= MIN_MEDIAN_COVERAGE
        and p05_ratio >= MIN_P05_COVERAGE
        and monthly_min_ratio >= MIN_MONTHLY_COVERAGE
        else "FAIL"
    )
    return {
        "status": status,
        "valid_day_count": int(len(daily)),
        "label_key_count": int(len(label_frame)),
        "finite_key_count": int(merged["is_finite"].sum()),
        "median_ratio": median_ratio,
        "p05_ratio": p05_ratio,
        "monthly_min_ratio": monthly_min_ratio,
        "daily": [
            {
                "signal_date": row.signal_date.date().isoformat(),
                "expected_key_count": int(row.expected_key_count),
                "finite_key_count": int(row.finite_key_count),
                "coverage_ratio": float(row.coverage_ratio),
            }
            for row in daily.itertuples(index=False)
        ],
        "monthly": [
            {
                "month": str(row.month),
                "expected_key_count": int(row.expected_key_count),
                "finite_key_count": int(row.finite_key_count),
                "coverage_ratio": float(row.coverage_ratio),
            }
            for row in monthly.itertuples(index=False)
        ],
    }


def _passes_predictive_gate(metrics: Mapping[str, object]) -> bool:
    """Return whether one signed RankIC summary clears the fixed gate."""
    try:
        return (
            float(metrics["signed_mean_rank_ic"]) > MIN_SIGNED_RANK_IC
            and float(metrics["signed_icir"]) >= MIN_SIGNED_ICIR
            and float(metrics["directional_day_share"]) >= MIN_DIRECTIONAL_SHARE
        )
    except (KeyError, TypeError, ValueError):
        return False


def build_review(
    *,
    training: Mapping[str, object],
    validation: Mapping[str, object],
    coverage: Mapping[str, object],
    train_similarity: Mapping[str, object],
    validation_similarity: Mapping[str, object],
) -> dict[str, object]:
    """Build a candidate review that never performs an admission mutation."""
    try:
        annual_pass = int(training["annual_gate_years"]) >= MIN_ANNUAL_GATE_YEARS
    except (KeyError, TypeError, ValueError):
        annual_pass = False
    gates_pass = (
        _passes_predictive_gate(training)
        and _passes_predictive_gate(validation)
        and annual_pass
        and coverage.get("status") == "PASS"
        and train_similarity.get("status") == "PASS"
        and validation_similarity.get("status") == "PASS"
    )
    return {
        "candidate_id": CANDIDATE_ID,
        "source_factor_id": SOURCE_FACTOR_ID,
        "training": dict(training),
        "validation": dict(validation),
        "coverage": dict(coverage),
        "train_similarity": dict(train_similarity),
        "validation_similarity": dict(validation_similarity),
        "status": (
            "READY_FOR_EXPLICIT_ADMISSION_REVIEW"
            if gates_pass
            else "REJECT_CANONICAL_GATE"
        ),
        "database_writes": False,
        "formal_factor_set_changed": False,
        "model_eligible": False,
        "model_training_started": False,
        "locked_test_read": False,
    }
