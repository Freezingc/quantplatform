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


class GP_Factor_061_Calculator(FactorCalculator):
    """遗传规划因子 #61，Rank-IC=0.050907
    表达式: add(amount, square(ts_regression_residual_5(
               ts_std_10(ts_regression_residual_20(low,
                   ts_regression_slope_20(cs_std(ts_regression_residual_5(
                       ts_std_10(ts_regression_residual_20(low,
                           ts_regression_slope_20(ts_std_60(amount)))),
                       pct_chg))))),
               pct_chg)))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low", "amount", "pct_chg"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        low     = data.xs('low',    level=0)
        amount  = data.xs('amount', level=0)
        pct_chg = data.xs('pct_chg', level=0)

        # ── 内层 ──────────────────────────────────────────────────────────
        # ts_std_60(amount) → ts_regression_slope_20 → DF
        _v1 = amount.rolling(60).std()
        _v2 = _v1.rolling(20).apply(_slope, raw=True)   # DF

        # ts_regression_residual_20(x=low_DF, y=_v2_DF) → DF（同形状，OK）
        _v4  = low.rolling(20).mean()
        _v5  = _v2.rolling(20).mean()
        _v6  = low.rolling(20).cov(_v2) / low.rolling(20).var().replace(0, np.nan)
        _v3  = (_v2 - _v5) - _v6 * (low - _v4)         # DF

        # ts_std_10(DF) → ts_regression_residual_5(x=DF, y=pct_chg_DF) → DF
        _v7  = _v3.rolling(10).std()                    # DF
        _v9  = _v7.rolling(5).mean()
        _v10 = pct_chg.rolling(5).mean()
        _v11 = _v7.rolling(5).cov(pct_chg) / _v7.rolling(5).var().replace(0, np.nan)
        _v8  = (pct_chg - _v10) - _v11 * (_v7 - _v9)   # DF

        # cs_std(DF) → Series；ts_regression_slope_20(Series) → Series
        _v12 = _v8.std(axis=1)                          # Series
        _v13 = _v12.rolling(20).apply(_slope, raw=True) # Series

        # ── 外层：ts_regression_residual_20(x=low_DF, y=_v13_Series) ──────
        # y 是 Series，须逐列循环，否则 Series-DF 列对齐出错
        _v14 = pd.DataFrame(index=low.index, columns=low.columns, dtype=float)
        for _c in low.columns:
            _cov     = low[_c].rolling(20).cov(_v13)
            _varx    = low[_c].rolling(20).var().replace(0, np.nan)
            _beta    = _cov / _varx
            _v14[_c] = (_v13 - _beta * low[_c]).replace([np.inf, -np.inf], np.nan)

        # ts_std_10(DF) → ts_regression_residual_5(x=DF, y=pct_chg_DF) → DF
        _v18 = _v14.rolling(10).std()                   # DF
        _v20 = _v18.rolling(5).mean()
        _v21 = pct_chg.rolling(5).mean()
        _v22 = _v18.rolling(5).cov(pct_chg) / _v18.rolling(5).var().replace(0, np.nan)
        _v19 = (pct_chg - _v21) - _v22 * (_v18 - _v20) # DF

        # square + add(amount, DF)
        factor = (amount + _v19 ** 2).replace([np.inf, -np.inf], np.nan)
        return factor
