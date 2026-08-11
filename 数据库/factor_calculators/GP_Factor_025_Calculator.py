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


class GP_Factor_025_Calculator(FactorCalculator):
    """遗传规划因子 #25，Rank-IC=0.053482
    表达式: max(turnover, mul(amount, ts_regression_slope_5(cs_std(
               ts_regression_10(shift_1(pct_chg), cs_zscore(ts_sum_20(cs_zscore(open))))))))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["open", "amount", "pct_chg", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        open_    = data.xs('open',          level=0)
        amount   = data.xs('amount',        level=0)
        pct_chg  = data.xs('pct_chg',       level=0)
        turnover = data.xs('turnover_rate', level=0)

        _v1 = pct_chg.shift(1)
        _v2 = open_.sub(open_.mean(axis=1), axis=0).div(open_.std(axis=1) + 1e-10, axis=0)
        _v3 = _v2.rolling(20).sum()
        _v4 = _v3.sub(_v3.mean(axis=1), axis=0).div(_v3.std(axis=1) + 1e-10, axis=0)
        _v5 = _v1.rolling(10).cov(_v4) / _v1.rolling(10).var().replace(0, np.nan)
        _v6 = _v5.std(axis=1)
        _v7 = _v6.rolling(5).apply(_slope, raw=True)
        _v8 = amount.mul(_v7, axis=0)
        factor = turnover.where(turnover >= _v8, _v8).replace([np.inf, -np.inf], np.nan)
        return factor
