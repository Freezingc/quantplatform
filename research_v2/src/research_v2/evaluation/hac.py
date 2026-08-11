"""HAC inference for overlapping-label daily performance series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm  # type: ignore[import-untyped]


@dataclass(frozen=True)
class HACMeanTest:
    observation_count: int
    max_lags: int
    mean: float
    standard_error: float
    t_statistic: float
    p_value: float


def hac_mean_test(values: pd.Series, *, max_lags: int = 5) -> HACMeanTest:
    """Fit an intercept-only HAC OLS model to a daily IC or portfolio-return series."""
    series = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) <= max_lags or max_lags < 0:
        raise ValueError("HAC series must contain more observations than max_lags")
    design = np.ones((len(series), 1))
    fitted = sm.OLS(series.to_numpy(dtype=float), design).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": max_lags},
    )
    return HACMeanTest(
        observation_count=len(series),
        max_lags=max_lags,
        mean=float(fitted.params[0]),
        standard_error=float(fitted.bse[0]),
        t_statistic=float(fitted.tvalues[0]),
        p_value=float(fitted.pvalues[0]),
    )
