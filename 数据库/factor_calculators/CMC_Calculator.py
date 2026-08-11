from .base import FactorCalculator
from .IndustryJointMomentum_Calculator import (
    compute_industry_returns, rolling_cmc,
)
import numpy as np
import pandas as pd


class CMC_Calculator(FactorCalculator):
    """行业联合动量因子 CMC = ICM - ICR

    在21日窗口内，取个股收益率最高5日对应的行业收益率加权和（ICM），
    减去最低15日对应的行业收益率加权和（ICR）。

    基于国信证券研报《行业联合动量因子》（2024-01-09）。
    """

    def __init__(self, s, lookback_days=21):
        super().__init__(s, lookback_days)
        self.required_factors = ["close_hfq", "total_mv"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        close = data.xs("close_hfq", level=0)
        rtn = close.pct_change(fill_method=None)
        total_mv = data.xs("total_mv", level=0)

        ind_info = self.fetcher.get_stock_industry_info()
        ts_to_l1 = dict(zip(ind_info["ts_code"], ind_info["l1_code"]))

        print("计算行业收益率 …")
        ind_returns = compute_industry_returns(rtn, total_mv, ts_to_l1)

        print("计算 CMC …")
        factor = pd.DataFrame(np.nan, index=rtn.index, columns=rtn.columns)
        window = self.lookback_days
        done = 0
        for col in rtn.columns:
            l1 = ts_to_l1.get(col)
            if l1 is None or l1 not in ind_returns.columns:
                continue
            factor[col] = rolling_cmc(
                rtn[col].values, ind_returns[l1].values, window,
            )
            done += 1
            if done % 500 == 0:
                print(f"  进度: {done}/{len(rtn.columns)}")
        print(f"  完成: {done}/{len(rtn.columns)} 只股票")
        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "行业联合动量_CMC",
    "creator": "zhangyudong",
    "create_date": "2026.06.03",
    "formula": "CMC = ICM - ICR，21日窗口内个股收益Top5/Bottom15对应日期的行业收益率加权差值",
    "description": "基于国信证券研报（2024-01-09）。ICM捕捉个股与行业的动量共振，ICR捕捉反转，CMC度量净动量效应",
    "lookback_days": 21,
    "method": "CMC_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "动量因子"
}
'''
