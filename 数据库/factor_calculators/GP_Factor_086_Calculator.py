from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_086_Calculator(FactorCalculator):
    """遗传规划因子 #86，Rank-IC=0.050574
    表达式: add(ts_std_10(ts_regression_60(amount, inv(turnover))), amount)
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["amount", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        amount   = data.xs('amount',        level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = (1 / turnover.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        _v2 = amount.rolling(60).cov(_v1) / amount.rolling(60).var().replace(0, np.nan)
        _v3 = _v2.rolling(10).std()
        _v4 = _v3 + amount
        factor = _v4.replace([np.inf, -np.inf], np.nan)
        return factor
