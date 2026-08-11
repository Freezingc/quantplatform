from .base import FactorCalculator, cumulative_to_single_quarter, lag_by_quarters, rolling_sum_quarters
import numpy as np
import pandas as pd


class GPMD_Calculator(FactorCalculator):
    """
    毛利率变动（TTM）：
    GPMD = 当期毛利率TTM - 上期毛利率TTM
    其中毛利率TTM = (营业收入TTM - 营业成本TTM) / 营业收入TTM
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_income_factors = ["revenue", "total_cogs"]

    def _calculate(self, start_time, end_time):
        # 1) 取数：累计营业收入、累计营业成本
        income_data = self.fetch_income_data(start_time, end_time)
        revenue_cumsum = income_data.xs("revenue", level=0)
        cogs_cumsum = income_data.xs("total_cogs", level=0)

        # 2) 单季度营业收入/成本（日频近似）
        revenue_quarter = cumulative_to_single_quarter(revenue_cumsum)
        cogs_quarter = cumulative_to_single_quarter(cogs_cumsum)

        # 3) TTM 滚动4期求和（日频近似）
        revenue_ttm = rolling_sum_quarters(revenue_quarter, n_quarters=4)
        cogs_ttm = rolling_sum_quarters(cogs_quarter, n_quarters=4)

        # 4) 毛利率TTM
        gpm_ttm = (revenue_ttm - cogs_ttm) / revenue_ttm

        # 5) 毛利率变动（交易日偏移法）
        gpm_ttm_lag = lag_by_quarters(gpm_ttm, 1)
        gpmd = gpm_ttm - gpm_ttm_lag

        gpmd = gpmd.replace([np.inf, -np.inf], np.nan)
        return gpmd


# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "GPMD",
    "creator": "zhangyudong",
    "create_date": "2026.01.08",
    "formula": "当期毛利率TTM - 上期毛利率TTM",
    "description": "毛利率变动（TTM），衡量企业毛利率的边际改善或恶化",
    "lookback_days": 750,
    "method": "GPMD_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子",
}


'''
# 说明：
# 1. 数据来源：
#    - revenue: 利润表 company_income_table，营业收入（累计值）
#    - total_cogs: 利润表 company_income_table，营业总成本/营业成本（累计值）
#
# 2. 计算逻辑：
#    - 单季度收入/成本 = 累计值差分（Q1 直接用累计）
#    - 收入TTM/成本TTM = 单季度滚动4期求和
#    - 毛利率TTM = (收入TTM - 成本TTM) / 收入TTM
#    - GPMD = 当期毛利率TTM - 上期毛利率TTM
#
# 3. 注意事项：
#    - 至少需要8个季度数据才能稳定计算TTM与差分
#    - 收入TTM 接近0时会出现异常值；inf/-inf 已转为 NaN
#    - 若“总成本”口径需调整为“营业成本”，请告知对应字段
'''
