"""Scope-bounded cross-sectional factor similarity calculations."""

from __future__ import annotations

import math
from datetime import date
from itertools import combinations
from typing import Any, cast

import pandas as pd


def _as_date(value: Any) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def _is_finite(value: Any) -> bool:
    return pd.notna(value) and math.isfinite(float(value))


def compute_factor_similarity(
    observations: pd.DataFrame,
    *,
    start_date: date,
    end_date: date,
    minimum_common_members: int = 30,
    minimum_valid_days: int = 60,
) -> pd.DataFrame:
    """Compute median daily absolute Spearman only within the explicit training scope."""
    required = {"signal_date", "ts_code", "factor_id", "factor_version", "value"}
    if missing := sorted(required.difference(observations.columns)):
        raise ValueError(f"factor observations are missing columns: {missing}")
    if start_date > end_date or minimum_common_members < 2 or minimum_valid_days < 1:
        raise ValueError("invalid similarity scope or thresholds")
    if observations.duplicated(
        ["signal_date", "ts_code", "factor_id", "factor_version"]
    ).any():
        raise ValueError("factor observations contain duplicate business keys")
    scoped = observations.copy(deep=True)
    scoped["signal_date"] = scoped["signal_date"].map(_as_date)
    scoped = scoped.loc[
        (scoped["signal_date"] >= start_date) & (scoped["signal_date"] <= end_date)
    ].copy()
    scoped["value"] = pd.to_numeric(scoped["value"], errors="coerce")
    factors = sorted(
        {
            (str(row["factor_id"]), int(row["factor_version"]))
            for row in cast(list[dict[str, Any]], scoped.to_dict(orient="records"))
        }
    )
    rows: list[dict[str, object]] = []
    for (left_id, left_version), (right_id, right_version) in combinations(factors, 2):
        left = scoped.loc[
            (scoped["factor_id"].astype(str) == left_id)
            & (scoped["factor_version"] == left_version),
            ["signal_date", "ts_code", "value"],
        ].rename(columns={"value": "left_value"})
        right = scoped.loc[
            (scoped["factor_id"].astype(str) == right_id)
            & (scoped["factor_version"] == right_version),
            ["signal_date", "ts_code", "value"],
        ].rename(columns={"value": "right_value"})
        merged = left.merge(
            right,
            on=["signal_date", "ts_code"],
            how="inner",
            validate="one_to_one",
        )
        correlations: list[float] = []
        for _, daily in merged.groupby("signal_date", sort=True):
            finite = daily.loc[
                daily["left_value"].map(_is_finite) & daily["right_value"].map(_is_finite),
                ["left_value", "right_value"],
            ]
            if len(finite) < minimum_common_members:
                continue
            rho = finite["left_value"].rank(method="average").corr(
                finite["right_value"].rank(method="average"),
                method="pearson",
            )
            if pd.notna(rho) and math.isfinite(float(rho)):
                correlations.append(abs(float(rho)))
        valid_days = len(correlations)
        rows.append(
            {
                "factor_id_left": left_id,
                "factor_version_left": left_version,
                "factor_id_right": right_id,
                "factor_version_right": right_version,
                "median_daily_abs_spearman": float(pd.Series(correlations).median())
                if correlations
                else float("nan"),
                "valid_days": valid_days,
                "status": "PASS" if valid_days >= minimum_valid_days else "INSUFFICIENT",
            }
        )
    return pd.DataFrame(rows)
