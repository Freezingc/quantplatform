from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_058_Calculator(FactorCalculator):
    """遗传规划因子 #58，Rank-IC=0.051074
    表达式: mul(amount, cs_mean(ts_std_20(close)))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "amount"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        close    = data.xs('close',         level=0)
        amount   = data.xs('amount',        level=0)
        _v1 = close.rolling(20).std()
        _v2 = _v1.mean(axis=1)
        _v3 = amount.mul(_v2, axis=0)
        factor = _v3.replace([np.inf, -np.inf], np.nan)
        return factor
