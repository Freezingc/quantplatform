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



class GP_Factor_002_Calculator(FactorCalculator):
    """遗传规划因子 #2，Rank-IC=0.059955
    表达式: pow(ts_regression_slope_60(vol), turnover)
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        vol      = data.xs('vol',           level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = vol.rolling(60).apply(_slope, raw=True)
        _v2 = (_v1 ** turnover).replace([np.inf, -np.inf], np.nan)
        factor = _v2.replace([np.inf, -np.inf], np.nan)
        return factor
