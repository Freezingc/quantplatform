from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_044_Calculator(FactorCalculator):
    """遗传规划因子 #44，Rank-IC=0.05178
    表达式: ts_max_5(mul(turnover, pct_chg))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["pct_chg", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        pct_chg  = data.xs('pct_chg',       level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = turnover * pct_chg
        _v2 = _v1.rolling(5).max()
        factor = _v2.replace([np.inf, -np.inf], np.nan)
        return factor
