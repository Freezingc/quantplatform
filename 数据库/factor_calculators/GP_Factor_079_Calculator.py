from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_079_Calculator(FactorCalculator):
    """遗传规划因子 #79，Rank-IC=0.05063
    表达式: add(turnover, ts_mean_5(pct_chg))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["pct_chg", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        pct_chg  = data.xs('pct_chg',       level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = pct_chg.rolling(5).mean()
        _v2 = turnover + _v1
        factor = _v2.replace([np.inf, -np.inf], np.nan)
        return factor
