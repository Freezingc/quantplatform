from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_049_Calculator(FactorCalculator):
    """遗传规划因子 #49，Rank-IC=0.051545
    表达式: mul(turnover, ts_min_20(ts_regression_residual_10(
               cs_max(ts_regression_20(ts_mean_20(turnover),
                   ts_regression_residual_60(sqrt(amount), ts_delta_30(turnover)))),
               cs_zscore(amount))))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["amount", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        amount   = data.xs('amount',        level=0)
        turnover = data.xs('turnover_rate', level=0)

        # ts_mean_20(turnover) → DF
        _v1 = turnover.rolling(20).mean()

        # ts_regression_residual_60(sqrt(amount), ts_delta_30(turnover)) → DF（两者同为DF）
        _v2  = amount.clip(lower=0) ** 0.5          # sqrt(amount)
        _v3  = turnover - turnover.shift(30)         # ts_delta_30(turnover)
        _v5  = _v2.rolling(60).mean()
        _v6  = _v3.rolling(60).mean()
        _v7  = _v2.rolling(60).cov(_v3) / _v2.rolling(60).var().replace(0, np.nan)
        _v4  = (_v3 - _v6) - _v7 * (_v2 - _v5)     # residual DF

        # ts_regression_20(DF _v1, DF _v4) → DF（元素级 beta）
        _v8  = _v1.rolling(20).cov(_v4) / _v1.rolling(20).var().replace(0, np.nan)

        # cs_max(DF) → Series  ← x
        _v9  = _v8.max(axis=1)

        # cs_zscore(amount) → DF  ← y
        _v10 = amount.sub(amount.mean(axis=1), axis=0).div(amount.std(axis=1) + 1e-10, axis=0)

        # ts_regression_residual_10(x=Series _v9, y=DF _v10) → 逐列循环
        _mx   = _v9.rolling(10).mean()
        _varx = _v9.rolling(10).var().replace(0, np.nan)
        _v11  = pd.DataFrame(index=_v10.index, columns=_v10.columns, dtype=float)
        for _c in _v10.columns:
            _my      = _v10[_c].rolling(10).mean()
            _cov     = _v9.rolling(10).cov(_v10[_c])
            _beta    = _cov / _varx
            _v11[_c] = ((_v10[_c] - _my) - _beta * (_v9 - _mx)).replace([np.inf, -np.inf], np.nan)

        # ts_min_20(DF) → mul(turnover, DF) → DF
        _v15   = _v11.rolling(20).min()
        factor = (turnover * _v15).replace([np.inf, -np.inf], np.nan)
        return factor
