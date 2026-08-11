from .base import FactorCalculator, cumulative_to_single_quarter, lag_by_quarters, rolling_sum_quarters
import numpy as np
import pandas as pd


class ATD_Calculator(FactorCalculator):
    """
    资产周转率变动（TTM）：
    ATD = 当期资产周转率TTM - 上期资产周转率TTM
    其中 AT = 营业收入TTM / 平均总资产
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_income_factors = ["revenue"]
        self.required_balance_sheet_factors = ["total_cur_assets", "total_nca"]

    def _calculate(self, start_time, end_time):
        # 1) 取数
        income_data = self.fetch_income_data(start_time, end_time)
        balance_data = self.fetch_balance_sheet_data(start_time, end_time)

        revenue_cumsum = income_data.xs("revenue", level=0)
        total_cur_assets = balance_data.xs("total_cur_assets", level=0)
        total_nca = balance_data.xs("total_nca", level=0)

        # 2) 总资产（日频展示值）
        total_assets = total_cur_assets + total_nca

        # 3) 单季度营业收入（日频近似）
        revenue_quarter = cumulative_to_single_quarter(revenue_cumsum)

        # 4) 营业收入TTM（日频近似）
        revenue_ttm = rolling_sum_quarters(revenue_quarter, n_quarters=4)

        # 5) 平均总资产（交易日偏移法）
        total_assets_lag = lag_by_quarters(total_assets, 4)
        avg_total_assets = (total_assets + total_assets_lag) / 2

        # 6) 资产周转率（TTM）
        at = revenue_ttm / avg_total_assets

        # 7) 资产周转率变动（交易日偏移法）
        at_lag = lag_by_quarters(at, 1)
        atd = at - at_lag

        atd = atd.replace([np.inf, -np.inf], np.nan)
        return atd


# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "ATD",
    "creator": "zhangyudong",
    "create_date": "2026.01.08",
    "formula": "当期资产周转率TTM - 上期资产周转率TTM",
    "description": "资产周转率变动（TTM），衡量资产使用效率的边际改善或恶化",
    "lookback_days": 750,
    "method": "ATD_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子",
}


'''
# 说明：
# 1. 数据来源：
#    - revenue: 利润表 company_income_table，营业收入（累计值）
#    - total_cur_assets: 资产负债表 balance_sheet，流动资产合计
#    - total_nca: 资产负债表 balance_sheet，非流动资产合计
#
# 2. 计算逻辑：
#    - 单季度营业收入 = 累计值差分（Q1 直接用累计）
#    - 营业收入TTM = 单季度营业收入滚动4期求和
#    - 总资产 = 流动资产 + 非流动资产
#    - 平均总资产 = (当期总资产 + 上年同期总资产) / 2
#    - 资产周转率 = 营业收入TTM / 平均总资产
#    - ATD = 当期资产周转率TTM - 上期资产周转率TTM（报告期差分）
#
# 3. 注意事项：
#    - 至少需要8个季度数据才能稳定计算TTM与差分
#    - 平均总资产为空或接近0时，结果会出现异常值
'''
