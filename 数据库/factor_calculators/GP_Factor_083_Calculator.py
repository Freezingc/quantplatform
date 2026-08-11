from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_083_Calculator(FactorCalculator):
    """遗传规划因子 #83，Rank-IC=0.050612
    表达式: mul(turnover, cs_std(ts_regression_10(close, ts_delta_ratio_20(turnover))))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        close    = data.xs('close',         level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = (turnover - turnover.shift(20)) / turnover.shift(20).replace(0, np.nan)
        _v2 = close.rolling(10).cov(_v1) / close.rolling(10).var().replace(0, np.nan)
        _v3 = _v2.std(axis=1)
        _v4 = turnover.mul(_v3, axis=0)
        factor = _v4.replace([np.inf, -np.inf], np.nan)
        return factor
