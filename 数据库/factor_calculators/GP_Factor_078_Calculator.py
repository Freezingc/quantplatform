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



class GP_Factor_078_Calculator(FactorCalculator):
    """遗传规划因子 #78，Rank-IC=0.050662
    表达式: max(sub(turnover, cs_zscore(ts_regression_slope_10(ts_rank_30(vol)))), amount)
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol", "amount", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        vol      = data.xs('vol',           level=0)
        amount   = data.xs('amount',        level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = vol.rolling(30).rank(pct=True)
        _v2 = _v1.rolling(10).apply(_slope, raw=True)
        _v3 = _v2.sub(_v2.mean(axis=1), axis=0).div(_v2.std(axis=1) + 1e-10, axis=0)
        _v4 = turnover - _v3
        _v5 = _v4.where(_v4 >= amount, amount)
        factor = _v5.replace([np.inf, -np.inf], np.nan)
        return factor
