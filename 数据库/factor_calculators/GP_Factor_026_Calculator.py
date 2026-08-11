from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_026_Calculator(FactorCalculator):
    """遗传规划因子 #26，Rank-IC=0.053305
    表达式: mul(mul(vol, ts_rank_60(cs_rank(close))), sub(turnover, ts_corr_10(open, sub(close, ts_rank_20(inv(t...
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "open", "vol", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        close    = data.xs('close',         level=0)
        open_    = data.xs('open',          level=0)
        vol      = data.xs('vol',           level=0)
        turnover = data.xs('turnover_rate', level=0)
        open = open_
        _v1 = close.rank(axis=1, pct=True)
        _v2 = _v1.rolling(60).rank(pct=True)
        _v3 = vol * _v2
        _v4 = close.rolling(5).mean()
        _v5 = (1 / _v4.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        _v6 = _v5.rolling(20).rank(pct=True)
        _v7 = close - _v6
        _v8 = open.rolling(10).corr(_v7)
        _v9 = turnover - _v8
        _v10 = _v3 * _v9
        factor = _v10.replace([np.inf, -np.inf], np.nan)
        return factor
