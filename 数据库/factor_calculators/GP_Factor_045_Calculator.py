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



class GP_Factor_045_Calculator(FactorCalculator):
    """遗传规划因子 #45，Rank-IC=0.051726
    表达式: add(amount, ts_regression_slope_60(ts_corr_30(turnover, ts_delta_20(ts_skewness_10(pct_chg)))))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["amount", "pct_chg", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        amount   = data.xs('amount',        level=0)
        pct_chg  = data.xs('pct_chg',       level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = pct_chg.rolling(10).skew()
        _v2 = _v1 - _v1.shift(20)
        _v3 = turnover.rolling(30).corr(_v2)
        _v4 = _v3.rolling(60).apply(_slope, raw=True)
        _v5 = amount + _v4
        factor = _v5.replace([np.inf, -np.inf], np.nan)
        return factor
