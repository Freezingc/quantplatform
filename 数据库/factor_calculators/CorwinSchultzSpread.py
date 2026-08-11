"""Corwin-Schultz (CS) high-low spread estimator research prototype.

This module deliberately contains no ``FactorCalculator`` subclass and no
database registration.  It accepts daily wide matrices (dates x stocks), so
the formula can be reviewed and validated before promotion to production.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CorwinSchultzResult:
    """Daily CS estimate and the two 21-trading-day research composites."""

    daily: pd.DataFrame
    mean_21: pd.DataFrame
    std_21: pd.DataFrame


def _validate_prices(high: pd.DataFrame, low: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(high, pd.DataFrame) or not isinstance(low, pd.DataFrame):
        raise TypeError("high 和 low 必须都是 pandas.DataFrame")
    if not high.index.equals(low.index) or not high.columns.equals(low.columns):
        raise ValueError("high 和 low 必须具有完全相同的索引与列")

    high = high.astype(np.float64, copy=False).sort_index()
    low = low.astype(np.float64, copy=False).reindex(index=high.index)
    valid = (high > 0.0) & (low > 0.0) & (high >= low)
    return high.where(valid), low.where(valid)


def corwin_schultz_daily(
    high: pd.DataFrame,
    low: pd.DataFrame,
    *,
    clip_negative: bool = False,
) -> pd.DataFrame:
    """Estimate the two-day Corwin-Schultz spread from daily highs and lows.

    The default follows the linked article exactly and therefore permits
    negative estimates.  Set ``clip_negative=True`` to apply the common
    empirical convention ``alpha=max(alpha, 0)``.
    """
    high, low = _validate_prices(high, low)

    log_high_low_sq = np.log(high / low).pow(2)
    beta = log_high_low_sq.rolling(window=2, min_periods=2).mean()

    two_day_high = high.rolling(window=2, min_periods=2).max()
    two_day_low = low.rolling(window=2, min_periods=2).min()
    gamma = np.log(two_day_high / two_day_low).pow(2)

    denominator = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (
        (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denominator
        - np.sqrt(gamma / denominator)
    )
    if clip_negative:
        alpha = alpha.clip(lower=0.0)

    # Algebraically identical to 2*(exp(alpha)-1)/(1+exp(alpha)), while
    # avoiding overflow for extreme input values.
    spread = 2.0 * np.tanh(alpha / 2.0)
    return spread.replace([np.inf, -np.inf], np.nan)


def corwin_schultz_factors(
    high: pd.DataFrame,
    low: pd.DataFrame,
    *,
    composite_window: int = 21,
    clip_negative: bool = False,
) -> CorwinSchultzResult:
    """Return daily CS spread plus its rolling mean and sample std composites."""
    if composite_window < 2:
        raise ValueError("composite_window 必须至少为 2")

    daily = corwin_schultz_daily(high, low, clip_negative=clip_negative)
    mean = daily.rolling(
        window=composite_window, min_periods=composite_window
    ).mean()
    std = daily.rolling(
        window=composite_window, min_periods=composite_window
    ).std(ddof=1)
    return CorwinSchultzResult(daily=daily, mean_21=mean, std_21=std)

