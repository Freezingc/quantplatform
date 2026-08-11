from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_040_Calculator(FactorCalculator):
    """遗传规划因子 #40，Rank-IC=0.051853
    表达式: add(cs_rank(close), turnover)
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        close    = data.xs('close',         level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = close.rank(axis=1, pct=True)
        _v2 = _v1 + turnover
        factor = _v2.replace([np.inf, -np.inf], np.nan)
        return factor
