from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_015_Calculator(FactorCalculator):
    """遗传规划因子 #15，Rank-IC=0.055733
    表达式: mul(turnover, cs_rank(pow(tanh(ts_delta_ratio_20(vol)), low)))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low", "vol", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        low      = data.xs('low',           level=0)
        vol      = data.xs('vol',           level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = (vol - vol.shift(20)) / vol.shift(20).replace(0, np.nan)
        _v2 = np.tanh(_v1)
        _v3 = (_v2 ** low).replace([np.inf, -np.inf], np.nan)
        _v4 = _v3.rank(axis=1, pct=True)
        _v5 = turnover * _v4
        factor = _v5.replace([np.inf, -np.inf], np.nan)
        return factor
