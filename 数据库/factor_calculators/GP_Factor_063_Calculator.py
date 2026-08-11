from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_063_Calculator(FactorCalculator):
    """遗传规划因子 #63，Rank-IC=0.050882
    表达式: ts_std_5(cs_zscore(amount))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["amount"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        amount   = data.xs('amount',        level=0)
        _v1 = amount.sub(amount.mean(axis=1), axis=0).div(amount.std(axis=1) + 1e-10, axis=0)
        _v2 = _v1.rolling(5).std()
        factor = _v2.replace([np.inf, -np.inf], np.nan)
        return factor
