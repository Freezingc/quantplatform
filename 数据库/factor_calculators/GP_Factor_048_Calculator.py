from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_048_Calculator(FactorCalculator):
    """遗传规划因子 #48，Rank-IC=0.051656
    表达式: max(div(amount, vol), turnover)
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol", "amount", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        vol      = data.xs('vol',           level=0)
        amount   = data.xs('amount',        level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = amount / vol.replace(0, np.nan)
        _v2 = _v1.where(_v1 >= turnover, turnover)
        factor = _v2.replace([np.inf, -np.inf], np.nan)
        return factor
