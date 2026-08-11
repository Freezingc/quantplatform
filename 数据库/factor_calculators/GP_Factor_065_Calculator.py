from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_065_Calculator(FactorCalculator):
    """遗传规划因子 #65，Rank-IC=0.050852
    表达式: max(turnover, ts_std_10(low))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        low      = data.xs('low',           level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = low.rolling(10).std()
        _v2 = turnover.where(turnover >= _v1, _v1)
        factor = _v2.replace([np.inf, -np.inf], np.nan)
        return factor
