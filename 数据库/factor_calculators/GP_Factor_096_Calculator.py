from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_096_Calculator(FactorCalculator):
    """遗传规划因子 #96，Rank-IC=0.050453
    表达式: mul(amount, cs_median(ts_std_10(add(abs(ts_regression_residual_5(low, turnover)), amount))))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low", "amount", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        low      = data.xs('low',           level=0)
        amount   = data.xs('amount',        level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v2 = low.rolling(5).mean()
        _v3 = turnover.rolling(5).mean()
        _v4 = low.rolling(5).cov(turnover) / low.rolling(5).var().replace(0, np.nan)
        _v1 = (turnover - _v3) - _v4 * (low - _v2)
        _v5 = _v1.abs()
        _v6 = _v5 + amount
        _v7 = _v6.rolling(10).std()
        _v8 = _v7.median(axis=1)
        _v9 = amount.mul(_v8, axis=0)
        factor = _v9.replace([np.inf, -np.inf], np.nan)
        return factor
