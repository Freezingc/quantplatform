from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_023_Calculator(FactorCalculator):
    """遗传规划因子 #23，Rank-IC=0.054067
    表达式: mul(turnover, ts_max_20(pct_chg))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["pct_chg", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        pct_chg  = data.xs('pct_chg',       level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = pct_chg.rolling(20).max()
        _v2 = turnover * _v1
        factor = _v2.replace([np.inf, -np.inf], np.nan)
        return factor
