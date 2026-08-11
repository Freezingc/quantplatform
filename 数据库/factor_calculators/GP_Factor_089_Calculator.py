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



class GP_Factor_089_Calculator(FactorCalculator):
    """遗传规划因子 #89，Rank-IC=0.05055
    表达式: add(ts_std_5(amount), ts_regression_slope_60(shift_10(high)))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["high", "amount"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        high     = data.xs('high',          level=0)
        amount   = data.xs('amount',        level=0)
        _v1 = amount.rolling(5).std()
        _v2 = high.shift(10)
        _v3 = _v2.rolling(60).apply(_slope, raw=True)
        _v4 = _v1 + _v3
        factor = _v4.replace([np.inf, -np.inf], np.nan)
        return factor
