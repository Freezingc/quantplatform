from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_013_Calculator(FactorCalculator):
    """遗传规划因子 #13，Rank-IC=0.056007
    表达式: mul(turnover, cs_rank(cs_zscore(shift_1(amount))))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["amount", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        amount   = data.xs('amount',        level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = amount.shift(1)
        _v2 = _v1.sub(_v1.mean(axis=1), axis=0).div(_v1.std(axis=1) + 1e-10, axis=0)
        _v3 = _v2.rank(axis=1, pct=True)
        _v4 = turnover * _v3
        factor = _v4.replace([np.inf, -np.inf], np.nan)
        return factor
