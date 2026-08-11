from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_093_Calculator(FactorCalculator):
    """遗传规划因子 #93，Rank-IC=0.050498
    表达式: mul(amount, cs_median(ts_std_10(add(ts_kurtosis_60(open), amount))))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["open", "amount"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        open_    = data.xs('open',          level=0)
        amount   = data.xs('amount',        level=0)
        open = open_
        _v1 = open.rolling(60).kurt()
        _v2 = _v1 + amount
        _v3 = _v2.rolling(10).std()
        _v4 = _v3.median(axis=1)
        _v5 = amount.mul(_v4, axis=0)
        factor = _v5.replace([np.inf, -np.inf], np.nan)
        return factor
