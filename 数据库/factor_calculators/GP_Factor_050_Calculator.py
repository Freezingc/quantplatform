from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_050_Calculator(FactorCalculator):
    """遗传规划因子 #50，Rank-IC=0.051498
    表达式: add(cs_rank(ts_sum_10(high)), turnover)
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["high", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        high     = data.xs('high',          level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = high.rolling(10).sum()
        _v2 = _v1.rank(axis=1, pct=True)
        _v3 = _v2 + turnover
        factor = _v3.replace([np.inf, -np.inf], np.nan)
        return factor
