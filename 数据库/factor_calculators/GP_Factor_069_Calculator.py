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



class GP_Factor_069_Calculator(FactorCalculator):
    """遗传规划因子 #69，Rank-IC=0.050808
    表达式: add(amount, square(ts_regression_residual_5(ts_std_10(ts_regression_residual_20(low, ts_regression_s...
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low", "amount", "pct_chg"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        low      = data.xs('low',           level=0)
        amount   = data.xs('amount',        level=0)
        pct_chg  = data.xs('pct_chg',       level=0)
        _v1 = amount.rolling(60).std()
        _v2 = _v1.rolling(20).apply(_slope, raw=True)
        _v4 = low.rolling(20).mean()
        _v5 = _v2.rolling(20).mean()
        _v6 = low.rolling(20).cov(_v2) / low.rolling(20).var().replace(0, np.nan)
        _v3 = (_v2 - _v5) - _v6 * (low - _v4)
        _v7 = _v3.rolling(10).std()
        _v9 = _v7.rolling(5).mean()
        _v10 = pct_chg.rolling(5).mean()
        _v11 = _v7.rolling(5).cov(pct_chg) / _v7.rolling(5).var().replace(0, np.nan)
        _v8 = (pct_chg - _v10) - _v11 * (_v7 - _v9)
        _v12 = _v8 ** 2
        _v13 = amount + _v12
        factor = _v13.replace([np.inf, -np.inf], np.nan)
        return factor
