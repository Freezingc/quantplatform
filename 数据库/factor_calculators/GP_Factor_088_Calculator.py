from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_088_Calculator(FactorCalculator):
    """遗传规划因子 #88，Rank-IC=0.050553
    表达式: add(amount, ts_sum_10(sigmoid(ts_kurtosis_20(high))))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["high", "amount"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        high     = data.xs('high',          level=0)
        amount   = data.xs('amount',        level=0)
        _v1 = high.rolling(20).kurt()
        _v2 = 1 / (1 + np.exp(-_v1))
        _v3 = _v2.rolling(10).sum()
        _v4 = amount + _v3
        factor = _v4.replace([np.inf, -np.inf], np.nan)
        return factor
