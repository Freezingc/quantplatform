from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_074_Calculator(FactorCalculator):
    """遗传规划因子 #74，Rank-IC=0.050708
    表达式: ts_max_5(ts_regression_residual_30(cs_mean(cs_rank(amount)), mul(amount, pct_chg)))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["amount", "pct_chg"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        amount  = data.xs('amount',  level=0)
        pct_chg = data.xs('pct_chg', level=0)

        # cs_rank(amount) → cs_mean → Series (x)
        _v2 = amount.rank(axis=1, pct=True).mean(axis=1)   # Series

        # mul(amount, pct_chg) → DF (y)
        _v3 = amount * pct_chg                              # DF

        # ts_regression_residual_30(x=Series _v2, y=DF _v3) → 逐列循环
        _mx   = _v2.rolling(30).mean()
        _varx = _v2.rolling(30).var().replace(0, np.nan)
        _v4   = pd.DataFrame(index=_v3.index, columns=_v3.columns, dtype=float)
        for _c in _v3.columns:
            _my      = _v3[_c].rolling(30).mean()
            _cov     = _v2.rolling(30).cov(_v3[_c])
            _beta    = _cov / _varx
            _v4[_c]  = ((_v3[_c] - _my) - _beta * (_v2 - _mx)).replace([np.inf, -np.inf], np.nan)

        factor = _v4.rolling(5).max().replace([np.inf, -np.inf], np.nan)
        return factor
