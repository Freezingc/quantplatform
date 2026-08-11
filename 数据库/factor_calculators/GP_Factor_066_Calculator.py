from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_066_Calculator(FactorCalculator):
    """遗传规划因子 #66，Rank-IC=0.05085
    表达式: mul(turnover, cs_mean(ts_regression_5(turnover, ts_regression_residual_30(ts_min_5(ts_mean_5(cs_medi...
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["amount", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        amount   = data.xs('amount',        level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = turnover.median(axis=1)
        _v2 = _v1.rolling(5).mean()
        _v3 = _v2.rolling(5).min()
        _v5 = _v3.rolling(30).mean()
        _v6 = amount.rolling(30).mean()
        # cov 为面板、var 为日频 Series：除法与后面乘法须 axis=0，否则按列对齐会得到全 NaN
        _v7_var = _v3.rolling(30).var().replace(0, np.nan)
        _v7 = _v3.rolling(30).cov(amount).div(_v7_var, axis=0)
        _v4 = (amount - _v6) - _v7.mul(_v3 - _v5, axis=0)
        _v8 = turnover.rolling(5).cov(_v4) / turnover.rolling(5).var().replace(0, np.nan)
        _v9 = _v8.mean(axis=1)
        _v10 = turnover.mul(_v9, axis=0)
        factor = _v10.replace([np.inf, -np.inf], np.nan)
        return factor
