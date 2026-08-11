from .base import FactorCalculator
from .IndustryJointMomentum_Calculator import (
    compute_industry_returns, rolling_icm,
)
import numpy as np
import pandas as pd


class ICM_Calculator(FactorCalculator):
    """行业捕捉动量因子（ICM）

    在21日窗口内取个股收益率最高的5个交易日，对应行业（申万一级、市值加权）
    收益率加权求和。权重 w_5 = [0.5, 0.59, 0.7, 0.84, 1.0]。

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

        print("计算 ICM …")
        factor = pd.DataFrame(np.nan, index=rtn.index, columns=rtn.columns)
        window = self.lookback_days
        done = 0
        for col in rtn.columns:
            l1 = ts_to_l1.get(col)
            if l1 is None or l1 not in ind_returns.columns:
                continue
            factor[col] = rolling_icm(
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
    "factor_name": "行业捕捉动量_ICM",
    "creator": "zhangyudong",
    "create_date": "2026.06.03",
    "formula": "ICM = w_5 · ind_ret[top5_dates]，21日窗口内个股收益Top5对应日期的行业收益率加权和",
    "description": "基于国信证券研报（2024-01-09）。取个股21日收益率最高的5天，对应申万一级行业市值加权收益率加权求和，度量个股与行业共振的动量效应",
    "lookback_days": 21,
    "method": "ICM_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "动量因子"
}
'''
