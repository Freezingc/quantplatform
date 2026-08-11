from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_037_Calculator(FactorCalculator):
    """遗传规划因子 #37，Rank-IC=0.052164
    表达式: mul(turnover, cs_rank(pow(tanh(ts_delta_ratio_20(turnover)), ts_sum_10(turnover))))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = (turnover - turnover.shift(20)) / turnover.shift(20).replace(0, np.nan)
        _v2 = np.tanh(_v1)
        _v3 = turnover.rolling(10).sum()
        _v4 = (_v2 ** _v3).replace([np.inf, -np.inf], np.nan)
        _v5 = _v4.rank(axis=1, pct=True)
        _v6 = turnover * _v5
        factor = _v6.replace([np.inf, -np.inf], np.nan)
        return factor
