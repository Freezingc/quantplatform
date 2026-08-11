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
) -> pd.Series:
    """Average per-date factor percentiles after applying frozen directions.

    Ranking is performed independently inside each signal-date cross-section, so
    factors with different units or outliers cannot dominate the ensemble. Missing
    values are omitted from that row's average; a row with no finite factors stays
    ``NaN`` and is therefore excluded by the downstream RankIC evaluator.
    """
    if not factor_order or len(set(factor_order)) != len(factor_order):
        raise ValueError("factor_order must be non-empty and unique")
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
    if features.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("features must have unique signal-date/security keys")

    numeric = features.loc[:, list(factor_order)].replace([np.inf, -np.inf], np.nan)
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    by_date = numeric.groupby(features["signal_date"], sort=False)
    ranks = by_date.rank(method="average")
    finite_counts = numeric.notna().groupby(features["signal_date"], sort=False).transform("sum")
    denominators = finite_counts - 1.0
    ranks = (ranks - 1.0).div(denominators.where(denominators > 0))
    ranks = ranks.where(denominators > 0)
    ranks = ranks.mask(finite_counts.eq(1) & numeric.notna(), 0.5)
    for factor_name, direction in zip(factor_order, directions, strict=True):
        if direction == -1:
            ranks[factor_name] = 1.0 - ranks[factor_name]
    return cast("pd.Series[Any]", ranks.mean(axis=1, skipna=True).rename("score"))
