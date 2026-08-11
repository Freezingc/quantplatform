from .base import FactorCalculator, cumulative_to_single_quarter, rolling_sum_quarters, lag_by_quarters
import numpy as np
import pandas as pd


class RAT_Calculator(FactorCalculator):
    """
    应收周转率（TTM）：
    RAT = 营业收入TTM / (平均应收账款 + 平均应收票据)
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_income_factors = ["revenue"]
        self.required_balance_sheet_factors = ["accounts_receiv", "accounts_receiv_bill"]

    def _calculate(self, start_time, end_time):
        # 1) 取数：累计营业收入、应收账款、应收票据
        income_data = self.fetch_income_data(start_time, end_time)
        balance_data = self.fetch_balance_sheet_data(start_time, end_time)

        revenue_cum = income_data.xs("revenue", level=0)
        accounts_receiv = balance_data.xs("accounts_receiv", level=0)
        accounts_receiv_bill = balance_data.xs("accounts_receiv_bill", level=0)

        # 2) 营业收入TTM（日频近似）
        revenue_quarter = cumulative_to_single_quarter(revenue_cum)
        revenue_ttm = rolling_sum_quarters(revenue_quarter, n_quarters=4)

        # 3) 平均应收账款 + 平均应收票据（交易日偏移法）
        receivable = accounts_receiv + accounts_receiv_bill
        receivable_lag = lag_by_quarters(receivable, 4)
        avg_receivable = (receivable + receivable_lag) / 2

        # 4) 应收周转率（TTM）
        rat = revenue_ttm / avg_receivable.replace(0, np.nan)
        rat = rat.replace([np.inf, -np.inf], np.nan)
        return rat


# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "RAT",
    "creator": "zhangyudong",
    "create_date": "2026.01.22",
    "formula": "营业收入TTM / (平均应收账款 + 平均应收票据)",
    "description": "应收周转率（TTM），衡量收入回款效率与应收占用水平",
    "lookback_days": 750,
    "method": "RAT_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子",
}


'''
# 说明：
# 1. 数据来源：
#    - revenue: 利润表 company_income_table，营业收入（累计值）
#    - accounts_receiv: 资产负债表 balance_sheet，应收账款
#    - accounts_receiv_bill: 资产负债表 balance_sheet，应收票据
#
# 2. 计算逻辑：
#    - 营业收入TTM = 单季度营业收入滚动4期求和（日频近似）
#    - 平均应收 = (当期应收 + 去年同期应收) / 2
#    - RAT = 营业收入TTM / 平均应收
#
# 3. 注意事项：
#    - 平均应收接近0时会出现异常值；inf/-inf 已转为 NaN
#    - 若票据口径需调整，请告知对应字段
'''
