from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_087_Calculator(FactorCalculator):
    """遗传规划因子 #87，Rank-IC=0.050563
    表达式: mul(turnover, ts_rank_60(cs_zscore(open)))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["open", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        open_    = data.xs('open',          level=0)
        turnover = data.xs('turnover_rate', level=0)
        open = open_
        _v1 = open.sub(open.mean(axis=1), axis=0).div(open.std(axis=1) + 1e-10, axis=0)
        _v2 = _v1.rolling(60).rank(pct=True)
        _v3 = turnover * _v2
        factor = _v3.replace([np.inf, -np.inf], np.nan)
        return factor
