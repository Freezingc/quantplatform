from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_097_Calculator(FactorCalculator):
    """遗传规划因子 #97，Rank-IC=0.050442
    表达式: mul(turnover, cs_rank(ts_delta_60(cs_min(ts_mean_20(inv(ts_kurtosis_10(vol)))))))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        vol      = data.xs('vol',           level=0)
        turnover = data.xs('turnover_rate', level=0)

        _v1 = vol.rolling(10).kurt()
        _v2 = (1 / _v1.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        _v3 = _v2.rolling(20).mean()
        _v4 = _v3.min(axis=1)              # cs_min → Series
        _v5 = _v4 - _v4.shift(60)          # ts_delta_60(Series) → Series
        # cs_rank(Series) → 时间维度排名
        _v6 = _v5.rank(pct=True)
        factor = turnover.mul(_v6, axis=0).replace([np.inf, -np.inf], np.nan)
        return factor
