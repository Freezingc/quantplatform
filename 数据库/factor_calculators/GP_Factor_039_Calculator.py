from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_039_Calculator(FactorCalculator):
    """遗传规划因子 #39，Rank-IC=0.051967
    表达式: mul(turnover, cs_mean(ts_kurtosis_10(turnover)))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = turnover.rolling(10).kurt()
        _v2 = _v1.mean(axis=1)
        _v3 = turnover.mul(_v2, axis=0)
        factor = _v3.replace([np.inf, -np.inf], np.nan)
        return factor
