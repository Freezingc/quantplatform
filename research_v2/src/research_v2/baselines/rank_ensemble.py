"""Scale-free cross-sectional rank ensemble for frozen factor directions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd


def directional_rank_scores(
    features: pd.DataFrame,
    factor_order: tuple[str, ...],
    directions: Sequence[int],
    *,
    minimum_factor_coverage: float = 0.60,
) -> pd.Series:
    """Rank daily equal-weight scores after applying frozen directions.

    Missing values are omitted from a row's equal-weight score, but a row must
    retain the configured minimum fraction of valid factors. The combined score
    is then percentile-ranked within each signal-date cross-section.
    """
    if not factor_order or len(set(factor_order)) != len(factor_order):
        raise ValueError("factor_order must be non-empty and unique")
    if not 0 < minimum_factor_coverage <= 1:
        raise ValueError("minimum_factor_coverage must be in (0, 1]")
    if len(directions) != len(factor_order) or any(
        direction not in {-1, 1} for direction in directions
    ):
        raise ValueError("directions must align with factor_order and contain only -1 or 1")
    expected_columns = ("signal_date", "ts_code", *factor_order)
    if tuple(features.columns) != expected_columns:
        raise ValueError(
            "feature columns must exactly match signal_date, ts_code, and factor_order"
        )
    if features.empty:
        raise ValueError("features cannot be empty")
    signal_dates = pd.to_datetime(features["signal_date"], errors="raise").dt.date
    if signal_dates.isna().any():
        raise ValueError("signal_date cannot be null")
    if features.assign(signal_date=signal_dates).duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("features must have unique signal-date/security keys")

    numeric = features.loc[:, list(factor_order)].replace([np.inf, -np.inf], np.nan)
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    for factor_name, direction in zip(factor_order, directions, strict=True):
        numeric[factor_name] = numeric[factor_name] * direction
    valid_factor_count = numeric.notna().sum(axis=1)
    minimum_valid_factor_count = int(np.ceil(minimum_factor_coverage * len(factor_order)))
    combined = numeric.mean(axis=1, skipna=True).where(
        valid_factor_count >= minimum_valid_factor_count
    )
    ranks = combined.groupby(signal_dates, sort=False).rank(method="average", pct=True)
    return cast("pd.Series[Any]", ranks.rename("score"))
