from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_055_Calculator(FactorCalculator):
    """遗传规划因子 #55，Rank-IC=0.05114
    表达式: mul(turnover, cs_mean(ts_regression_5(ts_min_5(ts_rank_20(turnover)), cs_zscore(ts_regression_residu...
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "high", "amount", "pct_chg", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        close    = data.xs('close',         level=0)
        high     = data.xs('high',          level=0)
        amount   = data.xs('amount',        level=0)
        pct_chg  = data.xs('pct_chg',       level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = turnover.rolling(20).rank(pct=True)
        _v2 = _v1.rolling(5).min()
        _v3 = high.max(axis=1)
        _v4 = _v3.rolling(60).corr(amount)
        _v6 = _v4.rolling(10).mean()
        _v7 = pct_chg.rolling(10).mean()
        _v8 = _v4.rolling(10).cov(pct_chg) / _v4.rolling(10).var().replace(0, np.nan)
        _v5 = (pct_chg - _v7) - _v8 * (_v4 - _v6)
        _v9 = close * _v5
        _v10 = turnover.sum(axis=1)
        _v11 = _v10.rolling(20).std()
        _v12 = _v11.abs()
        _v13 = _v12.rolling(60).kurt()
        _v15 = _v13.rolling(60).mean()
        _v16 = high.rolling(60).mean()
        # cov 为面板、var 为日频 Series：须 .div(..., axis=0)，与 _v14 中 Series 广播同理
        _v17_var = _v13.rolling(60).var().replace(0, np.nan)
        _v17 = _v13.rolling(60).cov(high).div(_v17_var, axis=0)
        # Series(_v13-_v15) 按日期索引；* 默认按列对齐会得到全 NaN，须 axis=0 按行广播
        _v14 = (high - _v16) - _v17.mul(_v13 - _v15, axis=0)
        _v18 = _v14 - _v14.shift(5)
        _v20 = _v9.rolling(5).mean()
        _v21 = _v18.rolling(5).mean()
        _v22 = _v9.rolling(5).cov(_v18) / _v9.rolling(5).var().replace(0, np.nan)
        _v19 = (_v18 - _v21) - _v22 * (_v9 - _v20)
        _v23 = _v19 - _v19.shift(5)
        _v25 = high.rolling(60).mean()
        _v26 = amount.rolling(60).mean()
        _v27 = high.rolling(60).cov(amount) / high.rolling(60).var().replace(0, np.nan)
        _v24 = (amount - _v26) - _v27 * (high - _v25)
        _v28 = _v24.sub(_v24.mean(axis=1), axis=0).div(_v24.std(axis=1) + 1e-10, axis=0)
        _v29 = _v28.rolling(5).mean()
        _v30 = turnover.rolling(60).rank(pct=True)
        _v31 = (amount ** _v30).replace([np.inf, -np.inf], np.nan)
        _v33 = _v29.rolling(20).mean()
        _v34 = _v31.rolling(20).mean()
        _v35 = _v29.rolling(20).cov(_v31) / _v29.rolling(20).var().replace(0, np.nan)
        _v32 = (_v31 - _v34) - _v35 * (_v29 - _v33)
        _v36 = _v32.rank(axis=1, pct=True)
        _v38 = _v23.rolling(60).mean()
        _v39 = _v36.rolling(60).mean()
        _v40 = _v23.rolling(60).cov(_v36) / _v23.rolling(60).var().replace(0, np.nan)
        _v37 = (_v36 - _v39) - _v40 * (_v23 - _v38)
        _v41 = _v37.sub(_v37.mean(axis=1), axis=0).div(_v37.std(axis=1) + 1e-10, axis=0)
        _v42 = _v2.rolling(5).cov(_v41) / _v2.rolling(5).var().replace(0, np.nan)
        _v43 = _v42.mean(axis=1)
        _v44 = turnover.mul(_v43, axis=0)
        factor = _v44.replace([np.inf, -np.inf], np.nan)
        return factor
