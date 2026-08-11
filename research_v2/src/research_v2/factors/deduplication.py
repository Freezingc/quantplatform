"""Deterministic factor-observation deduplication and conflict quarantine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_BUSINESS_KEYS = ("observation_date", "ts_code", "factor_id", "factor_version")


@dataclass(frozen=True)
class DedupStatistics:
    identical_groups: int
    single_nonnull_groups: int
    conflict_groups: int
    invalid_groups: int


@dataclass(frozen=True)
class DedupResult:
    clean: pd.DataFrame
    quarantine: pd.DataFrame
    statistics: DedupStatistics


def deduplicate_factor_observations(observations: pd.DataFrame) -> DedupResult:
    """Resolve duplicate groups without ever selecting a conflicting value.

    Group classification and representative selection are vectorized. This is
    important for YesLab, where a normal daily factor extract may contain an
    identical duplicate row for every security/date key.
    """
    required = set(_BUSINESS_KEYS) | {"raw_value", "source_row_id"}
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise ValueError(f"factor observations are missing columns: {missing}")
    if observations.loc[:, list(_BUSINESS_KEYS)].isna().any().any():
        raise ValueError("factor observation business keys cannot be null")
    frame = observations.copy(deep=True)
    frame["raw_value"] = pd.to_numeric(frame["raw_value"], errors="coerce")
    frame["_finite_value"] = np.isfinite(frame["raw_value"].to_numpy(dtype=float))
    duplicate_rows = frame.duplicated(list(_BUSINESS_KEYS), keep=False)
    unique = frame.loc[~duplicate_rows].copy()
    finite_unique = unique["_finite_value"]
    clean_unique = unique.loc[finite_unique].copy()
    clean_unique["dedup_resolution"] = "UNIQUE"
    invalid_unique = unique.loc[~finite_unique].copy()
    if not invalid_unique.empty:
        invalid_unique["reason"] = "NO_FINITE_VALUE"
    duplicate_frame = frame.loc[duplicate_rows].copy()
    grouped = duplicate_frame.groupby(list(_BUSINESS_KEYS), sort=True, dropna=False)
    grouped_summary = grouped["_finite_value"].agg(group_size="size", finite_count="sum")
    summary = pd.DataFrame(grouped_summary)
    finite_values = duplicate_frame["raw_value"].where(duplicate_frame["_finite_value"])
    duplicate_frame["_finite_numeric"] = finite_values
    summary["distinct_finite"] = duplicate_frame.groupby(
        list(_BUSINESS_KEYS), sort=True, dropna=False
    )["_finite_numeric"].nunique(dropna=True)
    summary["reason"] = pd.NA
    summary.loc[summary["distinct_finite"] > 1, "reason"] = "CONFLICT_QUARANTINE"
    summary.loc[summary["finite_count"] == 0, "reason"] = "NO_FINITE_VALUE"
    summary["dedup_resolution"] = np.where(
        summary["finite_count"] == summary["group_size"],
        "IDENTICAL_COLLAPSE",
        "SINGLE_NONNULL",
    )
    summary = summary.reset_index()

    ordered_finite = duplicate_frame.loc[duplicate_frame["_finite_value"]].sort_values(
        list(_BUSINESS_KEYS) + ["source_row_id"], kind="mergesort"
    )
    representatives = ordered_finite.drop_duplicates(list(_BUSINESS_KEYS), keep="first")
    representatives = representatives.merge(
        summary.loc[:, list(_BUSINESS_KEYS) + ["reason", "dedup_resolution"]],
        on=list(_BUSINESS_KEYS),
        how="left",
        validate="one_to_one",
        suffixes=("", "_summary"),
    )
    representatives = representatives.loc[representatives["reason"].isna()].copy()
    representatives = representatives.drop(columns=["reason"])
    quarantine_duplicate = duplicate_frame.merge(
        summary.loc[:, list(_BUSINESS_KEYS) + ["reason"]],
        on=list(_BUSINESS_KEYS),
        how="left",
        validate="many_to_one",
    )
    quarantine_duplicate = quarantine_duplicate.loc[quarantine_duplicate["reason"].notna()].copy()

    clean = pd.concat([clean_unique, representatives], ignore_index=True)
    if not clean.empty:
        clean = clean.sort_values(list(_BUSINESS_KEYS), ignore_index=True)
    quarantine = pd.concat([invalid_unique, quarantine_duplicate], ignore_index=True)
    if not quarantine.empty:
        quarantine = quarantine.sort_values(
            list(_BUSINESS_KEYS) + ["source_row_id"],
            ignore_index=True,
        )
    clean = clean.drop(columns=["_finite_value", "_finite_numeric"], errors="ignore")
    quarantine = quarantine.drop(columns=["_finite_value", "_finite_numeric"], errors="ignore")
    identical_groups = int(
        ((summary["reason"].isna()) & (summary["finite_count"] == summary["group_size"])).sum()
    )
    single_nonnull_groups = int(
        ((summary["reason"].isna()) & (summary["finite_count"] < summary["group_size"])).sum()
    )
    conflict_groups = int((summary["reason"] == "CONFLICT_QUARANTINE").sum())
    invalid_groups = int(len(invalid_unique) + (summary["reason"] == "NO_FINITE_VALUE").sum())
    return DedupResult(
        clean=clean,
        quarantine=quarantine,
        statistics=DedupStatistics(
            identical_groups=identical_groups,
            single_nonnull_groups=single_nonnull_groups,
            conflict_groups=conflict_groups,
            invalid_groups=invalid_groups,
        ),
    )
