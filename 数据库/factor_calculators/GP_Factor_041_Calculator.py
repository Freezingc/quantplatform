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



class GP_Factor_041_Calculator(FactorCalculator):
    """遗传规划因子 #41，Rank-IC=0.051801
    表达式: max(amount, ts_delta_10(cs_rank(ts_sum_20(ts_regression_slope_30(ts_regression_residual_20(ts_median...
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["amount"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        amount   = data.xs('amount',        level=0)
        _v1 = amount.rolling(60).kurt()
        _v2 = _v1.rolling(10).median()
        _v4 = _v2.rolling(20).mean()
        _v5 = amount.rolling(20).mean()
        _v6 = _v2.rolling(20).cov(amount) / _v2.rolling(20).var().replace(0, np.nan)
        _v3 = (amount - _v5) - _v6 * (_v2 - _v4)
        _v7 = _v3.rolling(30).apply(_slope, raw=True)
        _v8 = _v7.rolling(20).sum()
        _v9 = _v8.rank(axis=1, pct=True)
        _v10 = _v9 - _v9.shift(10)
        _v11 = amount.where(amount >= _v10, _v10)
        factor = _v11.replace([np.inf, -np.inf], np.nan)
        return factor
