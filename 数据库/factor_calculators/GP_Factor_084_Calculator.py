from .base import FactorCalculator
import numpy as np
import pandas as pd

def _slope(arr):
    if not np.all(np.isfinite(arr)) or len(arr) < 2:
        return np.nan
    t = np.arange(1, len(arr) + 1, dtype=float)
    cov = np.mean(t * arr) - np.mean(t) * np.mean(arr)
    var_t = np.var(t)
    return np.nan if var_t <= 0 else cov / var_t



class GP_Factor_084_Calculator(FactorCalculator):
    """遗传规划因子 #84，Rank-IC=0.050606
    表达式: add(cs_zscore(amount), ts_regression_slope_10(ts_delta_ratio_30(high)))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["high", "amount"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        high     = data.xs('high',          level=0)
        amount   = data.xs('amount',        level=0)
        _v1 = amount.sub(amount.mean(axis=1), axis=0).div(amount.std(axis=1) + 1e-10, axis=0)
        _v2 = (high - high.shift(30)) / high.shift(30).replace(0, np.nan)
        _v3 = _v2.rolling(10).apply(_slope, raw=True)
        _v4 = _v1 + _v3
        factor = _v4.replace([np.inf, -np.inf], np.nan)
        return factor
