from .base import FactorCalculator, cumulative_to_single_quarter, rolling_sum_quarters
import numpy as np
import pandas as pd


class OPtoGR_TTM_Calculator(FactorCalculator):
    """
    营业利润比毛利润（TTM）：
    OPtoGR_TTM = 营业利润TTM / 毛利润TTM
    其中毛利润TTM = 营业收入TTM - 营业成本TTM
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_income_factors = ["operate_profit", "revenue", "total_cogs"]

    def _calculate(self, start_time, end_time):
        # 1) 取数：累计营业利润、累计营业收入、累计营业成本
        income_data = self.fetch_income_data(start_time, end_time)
        op_cum = income_data.xs("operate_profit", level=0)
        revenue_cum = income_data.xs("revenue", level=0)
        cogs_cum = income_data.xs("total_cogs", level=0)

        # 2) 单季度值（日频近似）
        op_quarter = cumulative_to_single_quarter(op_cum)
        revenue_quarter = cumulative_to_single_quarter(revenue_cum)
        cogs_quarter = cumulative_to_single_quarter(cogs_cum)

        # 3) TTM 滚动4期求和（日频近似）
        op_ttm = rolling_sum_quarters(op_quarter, n_quarters=4)
        revenue_ttm = rolling_sum_quarters(revenue_quarter, n_quarters=4)
        cogs_ttm = rolling_sum_quarters(cogs_quarter, n_quarters=4)

        # 4) 毛利润TTM
        gross_profit_ttm = revenue_ttm - cogs_ttm

        # 5) 营业利润比毛利润（TTM）
        op_to_gr_ttm = op_ttm / gross_profit_ttm.replace(0, np.nan)
        op_to_gr_ttm = op_to_gr_ttm.replace([np.inf, -np.inf], np.nan)
        return op_to_gr_ttm


# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "OPtoGR_TTM",
    "creator": "zhangyudong",
    "create_date": "2026.01.22",
    "formula": "营业利润TTM / 毛利润TTM",
    "description": "营业利润比毛利润（TTM），衡量毛利润转化为营业利润的效率",
    "lookback_days": 750,
    "method": "OPtoGR_TTM_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子",
}


'''
# 说明：
# 1. 数据来源：
#    - operate_profit: 利润表 company_income_table，营业利润（累计值）
#    - revenue: 利润表 company_income_table，营业收入（累计值）
#    - total_cogs: 利润表 company_income_table，营业总成本/营业成本（累计值）
#
# 2. 计算逻辑：
#    - 单季度营业利润/收入/成本 = 累计值差分（日频近似）
#    - 毛利润TTM = 营业收入TTM - 营业成本TTM
#    - OPtoGR_TTM = 营业利润TTM / 毛利润TTM
#
# 3. 注意事项：
#    - 毛利润TTM 接近0时会出现异常值；inf/-inf 已转为 NaN
#    - 若“营业成本”口径需调整，请告知对应字段
'''
