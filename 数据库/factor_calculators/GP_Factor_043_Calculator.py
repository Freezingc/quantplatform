from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_043_Calculator(FactorCalculator):
    """遗传规划因子 #43，Rank-IC=0.051798
    表达式: add(amount, ts_delta_10(ts_sum_60(ts_delta_5(ts_kurtosis_60(close)))))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "amount"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        close    = data.xs('close',         level=0)
        amount   = data.xs('amount',        level=0)
        _v1 = close.rolling(60).kurt()
        _v2 = _v1 - _v1.shift(5)
        _v3 = _v2.rolling(60).sum()
        _v4 = _v3 - _v3.shift(10)
        _v5 = amount + _v4
        factor = _v5.replace([np.inf, -np.inf], np.nan)
        return factor
