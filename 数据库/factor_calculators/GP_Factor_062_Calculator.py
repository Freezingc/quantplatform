from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_062_Calculator(FactorCalculator):
    """遗传规划因子 #62，Rank-IC=0.050898
    表达式: add(ts_std_5(amount), mul(ts_kurtosis_20(ts_skewness_10(close)), ts_rank_20(amount)))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "amount"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        close    = data.xs('close',         level=0)
        amount   = data.xs('amount',        level=0)
        _v1 = amount.rolling(5).std()
        _v2 = close.rolling(10).skew()
        _v3 = _v2.rolling(20).kurt()
        _v4 = amount.rolling(20).rank(pct=True)
        _v5 = _v3 * _v4
        _v6 = _v1 + _v5
        factor = _v6.replace([np.inf, -np.inf], np.nan)
        return factor
