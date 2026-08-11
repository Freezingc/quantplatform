from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_075_Calculator(FactorCalculator):
    """遗传规划因子 #75，Rank-IC=0.050702
    表达式: max(turnover, mul(cs_zscore(ts_regression_residual_60(
               ts_sum_5(cs_std(ts_delta_20(ts_kurtosis_60(ts_kurtosis_20(ts_regression_20(low, open)))))), high)), low))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["open", "high", "low", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        open_    = data.xs('open',          level=0)
        high     = data.xs('high',          level=0)
        low      = data.xs('low',           level=0)
        turnover = data.xs('turnover_rate', level=0)

        # ts_regression_20(low, open) → DF.rolling.cov(DF)/DF.rolling.var() → DF
        _v1 = low.rolling(20).cov(open_) / low.rolling(20).var().replace(0, np.nan)
        # ts_kurtosis_20 → DF；ts_kurtosis_60 → DF；ts_delta_20 → DF
        _v2 = _v1.rolling(20).kurt()
        _v3 = _v2.rolling(60).kurt()
        _v4 = _v3 - _v3.shift(20)
        # cs_std(DF) → Series；ts_sum_5(Series) → Series
        _v5 = _v4.std(axis=1)
        _v6 = _v5.rolling(5).sum()

        # ts_regression_residual_60(Series _v6, DF_high) → 逐列
        _mx = _v6.rolling(60).mean()
        _var6 = _v6.rolling(60).var().replace(0, np.nan)
        _res = pd.DataFrame(index=high.index, columns=high.columns, dtype=float)
        for _c in high.columns:
            _my = high[_c].rolling(60).mean()
            _cov = _v6.rolling(60).cov(high[_c])
            _beta = _cov / _var6
            _res[_c] = ((high[_c] - _my) - _beta * (_v6 - _mx)).replace([np.inf, -np.inf], np.nan)

        # cs_zscore(DF) → DF；乘 low → DF
        _v7 = _res.sub(_res.mean(axis=1), axis=0).div(_res.std(axis=1) + 1e-10, axis=0)
        _v8 = _v7 * low

        # max(turnover_DF, DF)
        factor = turnover.where(turnover >= _v8, _v8).replace([np.inf, -np.inf], np.nan)
        return factor
