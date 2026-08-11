from .base import FactorCalculator, cumulative_to_single_quarter, rolling_sum_quarters, lag_by_quarters
import numpy as np
import pandas as pd


class OPMD_Calculator(FactorCalculator):
    """
    营业利润率变动：
    OPMD = 当期营业利润率 - 上期营业利润率
    其中营业利润率 = 营业利润TTM / 营业收入TTM
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_income_factors = ["operate_profit", "revenue"]

    def _calculate(self, start_time, end_time):
        # 1) 取数：累计营业利润、累计营业收入
        income_data = self.fetch_income_data(start_time, end_time)
        op_cum = income_data.xs("operate_profit", level=0)
        revenue_cum = income_data.xs("revenue", level=0)

        # 2) 单季度值（日频近似）
        op_quarter = cumulative_to_single_quarter(op_cum)
        revenue_quarter = cumulative_to_single_quarter(revenue_cum)

        # 3) TTM 滚动4期求和（日频近似）
        op_ttm = rolling_sum_quarters(op_quarter, n_quarters=4)
        revenue_ttm = rolling_sum_quarters(revenue_quarter, n_quarters=4)

        # 4) 营业利润率（TTM）
        opm_ttm = op_ttm / revenue_ttm.replace(0, np.nan)

        # 5) 营业利润率变动（交易日偏移法）
        opm_ttm_lag = lag_by_quarters(opm_ttm, 1)
        opmd = opm_ttm - opm_ttm_lag

        opmd = opmd.replace([np.inf, -np.inf], np.nan)
        return opmd


# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "OPMD",
    "creator": "zhangyudong",
    "create_date": "2026.01.22",
    "formula": "当期营业利润率TTM - 上期营业利润率TTM",
    "description": "营业利润率变动，衡量主营盈利能力的边际改善或恶化",
    "lookback_days": 750,
    "method": "OPMD_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子",
}


'''
# 说明：
# 1. 数据来源：
#    - operate_profit: 利润表 company_income_table，营业利润（累计值）
#    - revenue: 利润表 company_income_table，营业收入（累计值）
#
# 2. 计算逻辑：
#    - 单季度营业利润/收入 = 累计值差分（日频近似）
#    - 营业利润率TTM = 营业利润TTM / 营业收入TTM
#    - OPMD = 当期营业利润率TTM - 上期营业利润率TTM（交易日偏移）
#
# 3. 注意事项：
#    - 营业收入TTM 接近0时会出现异常值；inf/-inf 已转为 NaN
#    - 若需改为严格报告期口径，可替换为报告期维度计算
'''
