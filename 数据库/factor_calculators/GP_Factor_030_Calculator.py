from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_030_Calculator(FactorCalculator):
    """遗传规划因子 #30，Rank-IC=0.052887
    表达式: mul(ts_std_30(ts_max_5(ts_delta_60(ts_min_5(close)))), signed_power(amount, 2.6315789473684212))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "amount"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        close    = data.xs('close',         level=0)
        amount   = data.xs('amount',        level=0)
        _v1 = close.rolling(5).min()
        _v2 = _v1 - _v1.shift(60)
        _v3 = _v2.rolling(5).max()
        _v4 = _v3.rolling(30).std()
        _v5 = (np.sign(amount) * amount.abs() ** 2.6315789473684212).replace([np.inf, -np.inf], np.nan)
        _v6 = _v4 * _v5
        factor = _v6.replace([np.inf, -np.inf], np.nan)
        return factor
