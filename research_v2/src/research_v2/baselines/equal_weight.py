"""Equal-weight factor-score baseline."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd


def equal_weight_scores(features: pd.DataFrame, factor_order: tuple[str, ...]) -> pd.Series:
    """Average finite feature values while preserving all-missing rows as NaN."""
    if tuple(features.columns) != factor_order:
        raise ValueError("feature columns must exactly match factor_order")
    numeric = features.replace([float("inf"), float("-inf")], float("nan")).apply(
        pd.to_numeric,
        errors="coerce",
    )
    return cast("pd.Series[Any]", numeric.mean(axis=1, skipna=True))
