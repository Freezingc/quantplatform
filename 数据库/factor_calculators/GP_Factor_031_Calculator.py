from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_031_Calculator(FactorCalculator):
    """遗传规划因子 #31，Rank-IC=0.052732
    表达式: mul(turnover, ts_rank_60(cs_zscore(high)))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["high", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        high     = data.xs('high',          level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = high.sub(high.mean(axis=1), axis=0).div(high.std(axis=1) + 1e-10, axis=0)
        _v2 = _v1.rolling(60).rank(pct=True)
        _v3 = turnover * _v2
        factor = _v3.replace([np.inf, -np.inf], np.nan)
        return factor
