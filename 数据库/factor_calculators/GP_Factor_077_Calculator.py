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



class GP_Factor_077_Calculator(FactorCalculator):
    """遗传规划因子 #77，Rank-IC=0.05067
    表达式: mul(turnover, cs_max(ts_corr_60(ts_regression_slope_10(ts_rank_60(ts_kurtosis_10(ts_delta_5(relu(ts_...
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
        _v1 = amount.rank(axis=1, pct=True)
        _v2 = _v1.rolling(10).skew()
        _v3 = _v2.clip(lower=0)
        _v4 = _v3 - _v3.shift(5)
        _v5 = _v4.rolling(10).kurt()
        _v6 = _v5.rolling(60).rank(pct=True)
        _v7 = _v6.rolling(10).apply(_slope, raw=True)
        _v8 = pct_chg.rank(axis=1, pct=True)
        _v9 = _v8 - _v8.shift(60)
        _v10 = pct_chg * _v9
        _v11 = _v10.rolling(30).apply(_slope, raw=True)
        _v12 = _v11.mean(axis=1)
        _v14 = _v12.rolling(60).mean()
        _v15 = turnover.rolling(60).mean()
        _v16_var = _v12.rolling(60).var().replace(0, np.nan)
        _v16 = _v12.rolling(60).cov(turnover).div(_v16_var, axis=0)
        _v13 = (turnover - _v15) - _v16.mul(_v12 - _v14, axis=0)
        _v17 = _v13.rolling(60).median()
        _v18 = _v7.rolling(60).corr(_v17)
        _v19 = _v18.max(axis=1)
        _v20 = turnover.mul(_v19, axis=0)
        factor = _v20.replace([np.inf, -np.inf], np.nan)
        return factor
