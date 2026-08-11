from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_060_Calculator(FactorCalculator):
    """遗传规划因子 #60，Rank-IC=0.051031
    表达式: add(ts_std_10(ts_max_10(pct_chg)), turnover)
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["pct_chg", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        pct_chg  = data.xs('pct_chg',       level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = pct_chg.rolling(10).max()
        _v2 = _v1.rolling(10).std()
        _v3 = _v2 + turnover
        factor = _v3.replace([np.inf, -np.inf], np.nan)
        return factor
