from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_099_Calculator(FactorCalculator):
    """遗传规划因子 #99，Rank-IC=0.050413
    表达式: mul(turnover, cs_mean(ts_regression_5(sub(amount, ts_max_5(ts_delta_60(cs_zscore(relu(ts_delta_60(lo...
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low", "vol", "amount", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        low      = data.xs('low',           level=0)
        vol      = data.xs('vol',           level=0)
        amount   = data.xs('amount',        level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = low - low.shift(60)
        _v2 = _v1.clip(lower=0)
        _v3 = _v2.sub(_v2.mean(axis=1), axis=0).div(_v2.std(axis=1) + 1e-10, axis=0)
        _v4 = _v3 - _v3.shift(60)
        _v5 = _v4.rolling(5).max()
        _v6 = amount - _v5
        _v7 = vol.rank(axis=1, pct=True)
        _v8 = _v6.rolling(5).cov(_v7) / _v6.rolling(5).var().replace(0, np.nan)
        _v9 = _v8.mean(axis=1)
        _v10 = turnover.mul(_v9, axis=0)
        factor = _v10.replace([np.inf, -np.inf], np.nan)
        return factor
