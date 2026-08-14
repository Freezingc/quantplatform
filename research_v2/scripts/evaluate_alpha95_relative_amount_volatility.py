"""Evaluate isolated scale-normalized Alpha95 research candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd

CANDIDATE_WINDOWS = (10, 20, 40)


def compute_relative_amount_volatility(market: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return trailing amount standard deviation relative to trailing mean."""
    required = {"trade_date", "ts_code", "amount"}
    if missing := sorted(required.difference(market.columns)):
        raise ValueError(f"market frame is missing columns: {missing}")
    amount = market.pivot(index="trade_date", columns="ts_code", values="amount").sort_index()
    deviation = amount.rolling(window=window, min_periods=window).std(ddof=1)
    baseline = amount.rolling(window=window, min_periods=window).mean()
    return deviation.div(baseline.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
