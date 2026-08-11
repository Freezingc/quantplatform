from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_054_Calculator(FactorCalculator):
    """遗传规划因子 #54，Rank-IC=0.051216
    表达式: add(pow(ts_delta_20(ts_sum_20(cs_rank(ts_median_5(open)))),
               ts_sum_10(mul(shift_1(open), cs_max(ts_corr_60(turnover,
               ts_rank_60(cs_std(ts_regression_10(ts_skewness_20(cs_rank(cs_max(abs(turnover)))), amount)))))))), turnover)
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["open", "amount", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        open_    = data.xs('open',          level=0)
        amount   = data.xs('amount',        level=0)
        turnover = data.xs('turnover_rate', level=0)

        # 左侧: ts_delta_20(ts_sum_20(cs_rank(ts_median_5(open))))
        _lhs1 = open_.rolling(5).median().rank(axis=1, pct=True)
        _lhs2 = _lhs1.rolling(20).sum()
        _lhs  = _lhs2 - _lhs2.shift(20)

        # cs_rank(cs_max(abs(turnover))) → cs_max 返回 Series，cs_rank(Series) 按时间排名
        _v1 = turnover.abs().max(axis=1)                   # cs_max → Series
        _v2 = _v1.rank(pct=True)                           # cs_rank(Series) → 时间维排名

        # ts_skewness_20(Series) → Series
        _v3 = _v2.rolling(20).skew()

        # ts_regression_10(Series, DF_amount) → 逐列计算
        _v3var = _v3.rolling(10).var().replace(0, np.nan)
        _v4 = pd.DataFrame(index=amount.index, columns=amount.columns, dtype=float)
        for _c in amount.columns:
            _v4[_c] = (_v3.rolling(10).cov(amount[_c]) / _v3var).replace([np.inf, -np.inf], np.nan)

        # cs_std(DF) → Series；ts_rank_60(Series) → Series
        _v5 = _v4.std(axis=1)
        _v6 = _v5.rolling(60).rank(pct=True)

        # ts_corr_60(DF_turnover, Series) → 逐列
        _v7 = pd.DataFrame(index=turnover.index, columns=turnover.columns, dtype=float)
        for _c in turnover.columns:
            _v7[_c] = turnover[_c].rolling(60).corr(_v6)

        # cs_max(DF) → Series
        _v8 = _v7.max(axis=1)

        # mul(shift_1(open), Series) → DF.mul(Series, axis=0)；ts_sum_10 → DF
        _rhs = open_.shift(1).mul(_v8, axis=0).rolling(10).sum()

        # pow(lhs, rhs) + turnover
        _pow = (_lhs ** _rhs).replace([np.inf, -np.inf], np.nan)
        factor = (_pow + turnover).replace([np.inf, -np.inf], np.nan)
        return factor
