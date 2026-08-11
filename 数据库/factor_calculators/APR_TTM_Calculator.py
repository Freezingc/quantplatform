from .base import FactorCalculator, cumulative_to_single_quarter, rolling_sum_quarters
import numpy as np
import pandas as pd


class APR_TTM_Calculator(FactorCalculator):
    """
    应计利润占比（TTM）：
    APR_TTM = 应计利润TTM / 营业利润TTM
    其中应计利润 = 营业利润 - 经营现金流
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_income_factors = ["operate_profit"]
        self.required_cashflow_factors = ["n_cashflow_act"]

    def _calculate(self, start_time, end_time):
        # 1) 取数：累计营业利润、累计经营现金流
        income_data = self.fetch_income_data(start_time, end_time)
        cashflow_data = self.fetch_cashflow_data(start_time, end_time)

        op_cum = income_data.xs("operate_profit", level=0)
        ocf_cum = cashflow_data.xs("n_cashflow_act", level=0)

        # 2) 单季度值（日频近似）
        op_quarter = cumulative_to_single_quarter(op_cum)
        ocf_quarter = cumulative_to_single_quarter(ocf_cum)

        # 3) TTM 滚动4期求和（日频近似）
        op_ttm = rolling_sum_quarters(op_quarter, n_quarters=4)
        ocf_ttm = rolling_sum_quarters(ocf_quarter, n_quarters=4)

        # 4) 应计利润TTM 与占比
        accrual_ttm = op_ttm - ocf_ttm
        apr_ttm = accrual_ttm / op_ttm.replace(0, np.nan)
        apr_ttm = apr_ttm.replace([np.inf, -np.inf], np.nan)
        return apr_ttm


# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "APR_TTM",
    "creator": "zhangyudong",
    "create_date": "2026.01.22",
    "formula": "应计利润TTM / 营业利润TTM",
    "description": "应计利润占比（TTM），衡量利润中应计成分相对经营现金流的占比",
    "lookback_days": 750,
    "method": "APR_TTM_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子",
}


'''
# 说明：
# 1. 数据来源：
#    - operate_profit: 利润表 company_income_table，营业利润（累计值）
#    - n_cashflow_act: 现金流量表 cashflow_table，经营活动产生的现金流量净额（累计值）
#
# 2. 计算逻辑：
#    - 单季度营业利润/经营现金流 = 累计值差分（日频近似）
#    - 应计利润TTM = 营业利润TTM - 经营现金流TTM
#    - APR_TTM = 应计利润TTM / 营业利润TTM
#
# 3. 注意事项：
#    - 营业利润TTM 接近0时会出现异常值；inf/-inf 已转为 NaN
#    - 若需改为严格报告期口径，可替换为报告期维度计算
'''
