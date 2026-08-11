"""Label-independent structural quality assessment for factor observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd


class QualityStatus(StrEnum):
    PASS = "PASS"
    ELIGIBILITY_FAIL = "ELIGIBILITY_FAIL"
    BLOCKER = "BLOCKER"


@dataclass(frozen=True)
class FactorQualitySnapshot:
    factor_id: str
    factor_version: int
    status: QualityStatus
    blockers: tuple[str, ...]
    eligibility_failures: tuple[str, ...]
    metrics: dict[str, float | int]


def assess_factor_quality(
    universe: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    factor_id: str,
    factor_version: int,
    minimum_coverage: float = 0.70,
    minimum_median_unique_ratio: float = 1.00,
    maximum_zero_dispersion_date_share: float = 0.20,
    maximum_staleness_calendar_days: int = 180,
) -> FactorQualitySnapshot:
    """Score clean-factor structure using only universe and factor observations."""
    required_universe = {"signal_date", "ts_code"}
    required_observations = {"signal_date", "ts_code", "value", "stale_days"}
    if missing := sorted(required_universe.difference(universe.columns)):
        raise ValueError(f"universe is missing columns: {missing}")
    if missing := sorted(required_observations.difference(observations.columns)):
        raise ValueError(f"observations are missing columns: {missing}")
    if not 0 <= minimum_coverage <= 1 or not 0 <= maximum_zero_dispersion_date_share <= 1:
        raise ValueError("quality thresholds must be proportions in [0, 1]")
    blockers: list[str] = []
    if universe.duplicated(["signal_date", "ts_code"]).any():
        blockers.append("DUPLICATE_UNIVERSE_KEY")
    if observations.duplicated(["signal_date", "ts_code"]).any():
        blockers.append("DUPLICATE_BUSINESS_KEY")
    numeric_values = pd.to_numeric(observations["value"], errors="coerce")
    finite_mask = numeric_values.map(lambda value: pd.notna(value) and math.isfinite(float(value)))
    if (~finite_mask & observations["value"].notna()).any():
        blockers.append("NONFINITE_CLEAN_VALUE")
    metric_observations = observations.drop_duplicates(
        ["signal_date", "ts_code"],
        keep="first",
    )
    universe_keys = universe.loc[:, ["signal_date", "ts_code"]].copy()
    merged = universe_keys.merge(
        metric_observations.loc[:, ["signal_date", "ts_code", "value", "stale_days"]],
        on=["signal_date", "ts_code"],
        how="left",
        validate="one_to_one",
    )
    merged["value"] = pd.to_numeric(merged["value"], errors="coerce")
    merged["stale_days"] = pd.to_numeric(merged["stale_days"], errors="coerce")
    merged["is_usable"] = (
        merged["value"].map(lambda value: pd.notna(value) and math.isfinite(float(value)))
        & merged["stale_days"].notna()
        & (merged["stale_days"] <= maximum_staleness_calendar_days)
    )
    daily_unique_ratios: list[float] = []
    zero_dispersion_days = 0
    total_dates = 0
    for _, group in merged.groupby("signal_date", sort=True):
        total_dates += 1
        values = group.loc[group["is_usable"], "value"].to_numpy(dtype=float)
        if len(values) == 0:
            daily_unique_ratios.append(0.0)
            continue
        daily_unique_ratios.append(float(len(np.unique(values)) / len(values)))
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        if mad == 0:
            zero_dispersion_days += 1
    universe_member_count = len(merged)
    finite_value_count = int(merged["is_usable"].sum())
    coverage = finite_value_count / universe_member_count if universe_member_count else 0.0
    median_unique_ratio = float(np.median(daily_unique_ratios)) if daily_unique_ratios else 0.0
    zero_dispersion_share = zero_dispersion_days / total_dates if total_dates else 0.0
    metrics: dict[str, float | int] = {
        "universe_member_count": universe_member_count,
        "finite_value_count": finite_value_count,
        "coverage": coverage,
        "median_daily_unique_ratio": median_unique_ratio,
        "zero_dispersion_date_share": zero_dispersion_share,
    }
    eligibility_failures: list[str] = []
    if coverage < minimum_coverage:
        eligibility_failures.append("LOW_COVERAGE")
    if median_unique_ratio < minimum_median_unique_ratio:
        eligibility_failures.append("LOW_UNIQUE_RATIO")
    if zero_dispersion_share > maximum_zero_dispersion_date_share:
        eligibility_failures.append("EXCESS_ZERO_DISPERSION")
    status = (
        QualityStatus.BLOCKER
        if blockers
        else QualityStatus.ELIGIBILITY_FAIL
        if eligibility_failures
        else QualityStatus.PASS
    )
    return FactorQualitySnapshot(
        factor_id=factor_id,
        factor_version=factor_version,
        status=status,
        blockers=tuple(blockers),
        eligibility_failures=tuple(eligibility_failures),
        metrics=metrics,
    )
