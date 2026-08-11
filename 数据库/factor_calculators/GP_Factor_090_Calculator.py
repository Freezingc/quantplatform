from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_090_Calculator(FactorCalculator):
    """遗传规划因子 #90，Rank-IC=0.05054
    表达式: add(amount, ts_delta_10(ts_sum_60(turnover)))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["amount", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        amount   = data.xs('amount',        level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = turnover.rolling(60).sum()
        _v2 = _v1 - _v1.shift(10)
        _v3 = amount + _v2
        factor = _v3.replace([np.inf, -np.inf], np.nan)
        return factor
