from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_019_Calculator(FactorCalculator):
    """遗传规划因子 #19，Rank-IC=0.054852
    表达式: mul(ts_sum_10(turnover), abs(amount))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["amount", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        amount   = data.xs('amount',        level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = turnover.rolling(10).sum()
        _v2 = amount.abs()
        _v3 = _v1 * _v2
        factor = _v3.replace([np.inf, -np.inf], np.nan)
        return factor
