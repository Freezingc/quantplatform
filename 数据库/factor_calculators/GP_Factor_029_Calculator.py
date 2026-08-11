from .base import FactorCalculator
import numpy as np
import pandas as pd

def _slope(arr):
    if not np.all(np.isfinite(arr)) or len(arr) < 2:
        return np.nan
    t = np.arange(1, len(arr) + 1, dtype=float)
    cov = np.mean(t * arr) - np.mean(t) * np.mean(arr)
    var_t = np.var(t)
    return np.nan if var_t <= 0 else cov / var_t


class GP_Factor_029_Calculator(FactorCalculator):
    """遗传规划因子 #29，Rank-IC=0.053017
    表达式: ts_max_5(ts_regression_residual_10(cs_max(ts_corr_60(pct_chg,
               ts_regression_slope_30(ts_corr_20(open, ts_regression_slope_10(low))))),
               mul(cs_zscore(pct_chg), amount)))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["open", "low", "amount", "pct_chg"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        open_   = data.xs('open',    level=0)
        low     = data.xs('low',     level=0)
        amount  = data.xs('amount',  level=0)
        pct_chg = data.xs('pct_chg', level=0)

        # ts_regression_slope_10(low) → DF
        _v1 = low.rolling(10).apply(_slope, raw=True)
        # ts_corr_20(open, DF) → DF；ts_regression_slope_30(DF) → DF
        _v2 = open_.rolling(20).corr(_v1)
        _v3 = _v2.rolling(30).apply(_slope, raw=True)
        # ts_corr_60(pct_chg_DF, DF) → DF；cs_max → Series
        _v4 = pct_chg.rolling(60).corr(_v3)
        _v5 = _v4.max(axis=1)                  # cs_max → Series

        # mul(cs_zscore(pct_chg), amount) → DF
        _v6 = pct_chg.sub(pct_chg.mean(axis=1), axis=0).div(pct_chg.std(axis=1) + 1e-10, axis=0)
        _v7 = _v6 * amount                      # y = DF

        # ts_regression_residual_10(x=Series _v5, y=DF _v7) → 逐列循环，避免列名混杂
        _mx  = _v5.rolling(10).mean()
        _var = _v5.rolling(10).var().replace(0, np.nan)
        _v8  = pd.DataFrame(index=_v7.index, columns=_v7.columns, dtype=float)
        for _c in _v7.columns:
            _my   = _v7[_c].rolling(10).mean()
            _cov  = _v5.rolling(10).cov(_v7[_c])   # Series × Series → OK
            _beta = _cov / _var
            _v8[_c] = ((_v7[_c] - _my) - _beta * (_v5 - _mx)).replace([np.inf, -np.inf], np.nan)

        # ts_max_5(DF) → DF
        factor = _v8.rolling(5).max().replace([np.inf, -np.inf], np.nan)
        return factor
