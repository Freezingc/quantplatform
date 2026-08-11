from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_020_Calculator(FactorCalculator):
    """遗传规划因子 #20，Rank-IC=0.054554
    表达式: add(pow(pct_chg, ts_sum_10(turnover)), pct_chg)
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["pct_chg", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        pct_chg  = data.xs('pct_chg',       level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = turnover.rolling(10).sum()
        _v2 = (pct_chg ** _v1).replace([np.inf, -np.inf], np.nan)
        _v3 = _v2 + pct_chg
        factor = _v3.replace([np.inf, -np.inf], np.nan)
        return factor
