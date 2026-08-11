from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_001_Calculator(FactorCalculator):
    """遗传规划因子 #1，Rank-IC=0.063069
    表达式: mul(turnover, cs_rank(pow(pct_chg, ts_sum_10(turnover))))
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
        _v3 = _v2.rank(axis=1, pct=True)
        _v4 = turnover * _v3
        factor = _v4.replace([np.inf, -np.inf], np.nan)
        return factor
