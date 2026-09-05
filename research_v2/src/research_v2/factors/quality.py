"""Label-independent structural quality assessment for factor observations."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
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
    scope_id: str
    as_of_date: date
    status: QualityStatus
    blockers: tuple[str, ...]
    eligibility_failures: tuple[str, ...]
    metrics: dict[str, float | int]
    # Blocker conditions that could not be evaluated because the caller did not supply the
    # input they need. Recorded rather than skipped silently: a gate that passes because it
    # never looked is worse than one that fails, and the omission belongs in the snapshot
    # that gets archived.
    unchecked_blockers: tuple[str, ...] = ()


def assess_factor_quality(
    universe: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    factor_id: str,
    factor_version: int,
    scope_id: str,
    as_of_date: date,
    minimum_coverage: float = 0.70,
    minimum_median_unique_ratio: float = 1.00,
    maximum_zero_dispersion_date_share: float = 0.20,
    maximum_staleness_calendar_days: int = 180,
    quarantined_keys: Iterable[tuple[date, str]] | None = None,
    known_factor_versions: Iterable[int] | None = None,
) -> FactorQualitySnapshot:
    """Score clean-factor structure using only universe and factor observations."""
    required_universe = {"signal_date", "ts_code"}
    required_observations = {"signal_date", "ts_code", "value", "stale_days"}
    if missing := sorted(required_universe.difference(universe.columns)):
        raise ValueError(f"universe is missing columns: {missing}")
    if missing := sorted(required_observations.difference(observations.columns)):
        raise ValueError(f"observations are missing columns: {missing}")
    # C6.1 keys a snapshot by (as_of_date, factor_id, factor_version, scope_id). Without the
    # scope and the as-of date a snapshot cannot be interpreted: the same factor scores
    # differently across training scopes and across vintages of the source, so a stale or
    # wrong-scope snapshot would be indistinguishable from a current one. The field name and
    # type match ScreeningRecord.scope_id in verify_admission_consistency.py, which is the
    # consumer these snapshots are recorded for.
    if not scope_id:
        raise ValueError("scope_id cannot be empty")
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
    universe_keys = universe_keys.drop_duplicates()
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
        # A negative stale_days means available_date is after signal_date, i.e. the value
        # did not exist yet. Counting it as usable would let a PIT violation inflate
        # coverage while the blocker below reported it, so the two must agree.
        & (merged["stale_days"] >= 0)
        & (merged["stale_days"] <= maximum_staleness_calendar_days)
    )
    finite_value_mask = merged["value"].map(
        lambda value: pd.notna(value) and math.isfinite(float(value))
    )
    stale_mask = (
        finite_value_mask
        & merged["stale_days"].notna()
        & (merged["stale_days"] > maximum_staleness_calendar_days)
    )
    daily_unique_ratios: list[float] = []
    daily_dispersions: list[float] = []
    usable_per_date: list[int] = []
    pooled_values: list[float] = []
    zero_value_count = 0
    dates_with_usable = 0
    securities_with_usable: set[str] = set()
    zero_dispersion_days = 0
    total_dates = 0
    for _, group in merged.groupby("signal_date", sort=True):
        total_dates += 1
        usable = group.loc[group["is_usable"]]
        values = usable["value"].to_numpy(dtype=float)
        usable_per_date.append(int(len(values)))
        if len(values) == 0:
            daily_unique_ratios.append(0.0)
            # An empty date contributes no dispersion sample. Counting it as zero would
            # drag the median dispersion down and make a factor with missing dates look
            # more stable than one that is merely flat.
            continue
        dates_with_usable += 1
        securities_with_usable.update(usable["ts_code"].astype(str))
        pooled_values.extend(float(value) for value in values)
        zero_value_count += int(np.count_nonzero(values == 0.0))
        daily_unique_ratios.append(float(len(np.unique(values)) / len(values)))
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        daily_dispersions.append(mad)
        if mad == 0:
            zero_dispersion_days += 1
    universe_member_count = len(merged)
    finite_value_count = int(merged["is_usable"].sum())
    coverage = finite_value_count / universe_member_count if universe_member_count else 0.0
    median_unique_ratio = float(np.median(daily_unique_ratios)) if daily_unique_ratios else 0.0
    zero_dispersion_share = zero_dispersion_days / total_dates if total_dates else 0.0
    pooled = np.asarray(pooled_values, dtype=float)
    pooled_median = float(np.median(pooled)) if pooled.size else 0.0
    observation_rows = len(observations)
    unique_observation_keys = len(observations.drop_duplicates(["signal_date", "ts_code"]))
    universe_security_count = int(merged["ts_code"].astype(str).nunique())
    metrics: dict[str, float | int] = {
        "universe_member_count": universe_member_count,
        "finite_value_count": finite_value_count,
        "coverage": coverage,
        "median_daily_unique_ratio": median_unique_ratio,
        "zero_dispersion_date_share": zero_dispersion_share,
        # C6.1's distribution metrics, pooled over every usable value in the scope.
        "zero_ratio": zero_value_count / pooled.size if pooled.size else 0.0,
        "value_median": pooled_median,
        "value_mad": (
            float(np.median(np.abs(pooled - pooled_median))) if pooled.size else 0.0
        ),
        "value_p01": float(np.percentile(pooled, 1)) if pooled.size else 0.0,
        "value_p50": float(np.percentile(pooled, 50)) if pooled.size else 0.0,
        "value_p99": float(np.percentile(pooled, 99)) if pooled.size else 0.0,
        "median_daily_dispersion": (
            float(np.median(daily_dispersions)) if daily_dispersions else 0.0
        ),
        # A value that is present but too old to use is missing for research purposes,
        # and how often that happens is a separate question from how often it is absent.
        "stale_share": (
            int(stale_mask.sum()) / universe_member_count if universe_member_count else 0.0
        ),
        "duplicate_observation_rate": (
            (observation_rows - unique_observation_keys) / observation_rows
            if observation_rows
            else 0.0
        ),
        "longest_missing_date_streak": _longest_zero_streak(usable_per_date),
        "date_coverage": dates_with_usable / total_dates if total_dates else 0.0,
        "security_coverage": (
            len(securities_with_usable) / universe_security_count
            if universe_security_count
            else 0.0
        ),
        # finite ratio describes what the source delivered, coverage what the universe could
        # use: a factor can be entirely finite and still cover almost nothing.
        "finite_ratio": _finite_ratio(observations),
    }
    if quarantined_keys is not None:
        # Emitted only when measured. A conflict rate of 0.0 and an unmeasured conflict rate
        # are different statements, and unchecked_blockers names the latter.
        metrics["conflict_key_rate"] = (
            len(set(quarantined_keys)) / observation_rows if observation_rows else 0.0
        )

    # The remaining three blockers. PIT violation needs no extra input: stale_days is
    # signal_date - available_date, so a negative value means the observation became
    # usable before it existed.
    if bool((pd.to_numeric(merged["stale_days"], errors="coerce") < 0).any()):
        blockers.append("PIT_VIOLATION")
    unchecked: list[str] = []
    if quarantined_keys is None:
        unchecked.append("CONFLICT_KEY_IN_CLEAN")
    elif _clean_key_overlap(observations, quarantined_keys):
        blockers.append("CONFLICT_KEY_IN_CLEAN")
    if known_factor_versions is None:
        unchecked.append("UNKNOWN_REGISTRY_VERSION")
    elif factor_version not in set(known_factor_versions):
        blockers.append("UNKNOWN_REGISTRY_VERSION")

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
        scope_id=scope_id,
        as_of_date=as_of_date,
        status=status,
        blockers=tuple(blockers),
        eligibility_failures=tuple(eligibility_failures),
        metrics=metrics,
        unchecked_blockers=tuple(unchecked),
    )


def _longest_zero_streak(counts: Sequence[int]) -> int:
    """Longest run of consecutive dates carrying no usable value at all.

    A streak rather than a count, because one dark day and a dark week are different
    failures: the first is a data hiccup, the second makes the factor unusable across a
    whole rebalance. ``counts`` must already be in date order, which the caller gets from
    ``groupby(sort=True)``.
    """
    longest = 0
    current = 0
    for count in counts:
        if count == 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _finite_ratio(observations: pd.DataFrame) -> float:
    """Share of delivered values that are finite, over the values that are not null."""
    values = pd.to_numeric(observations["value"], errors="coerce")
    present = int(values.notna().sum())
    if not present:
        return 0.0
    finite = int(
        values.map(lambda value: pd.notna(value) and math.isfinite(float(value))).sum()
    )
    return finite / present


def _clean_key_overlap(
    observations: pd.DataFrame, quarantined_keys: Iterable[tuple[date, str]]
) -> bool:
    """Whether any key deduplication quarantined nonetheless reached the clean input.

    Both sides are normalised to ``(date, str)`` because this codebase stores signal dates
    as ``datetime.date`` in assembled frames and as ``pandas.Timestamp`` in others, and a
    type mismatch here would read as "no overlap" -- the silent-pass direction.
    """
    clean = {
        (pd.Timestamp(day).date(), str(code))
        for day, code in zip(
            observations["signal_date"], observations["ts_code"], strict=True
        )
    }
    quarantined = {
        (pd.Timestamp(day).date(), str(code)) for day, code in quarantined_keys
    }
    return bool(clean & quarantined)
