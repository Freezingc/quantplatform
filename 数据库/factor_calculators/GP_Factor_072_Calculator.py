from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_072_Calculator(FactorCalculator):
    """遗传规划因子 #72，Rank-IC=0.050745
    表达式: mul(mul(close, turnover), turnover)
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        close    = data.xs('close',         level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = close * turnover
        _v2 = _v1 * turnover
        factor = _v2.replace([np.inf, -np.inf], np.nan)
        return factor
