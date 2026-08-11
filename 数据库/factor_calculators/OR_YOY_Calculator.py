from .base import FactorCalculator, cumulative_to_single_quarter, lag_by_quarters, rolling_sum_quarters
import numpy as np
import pandas as pd


class OR_YOY_Calculator(FactorCalculator):
    """
    营业收入 TTM 同比：
    (OR_TTM_t - OR_TTM_{t-4}) / OR_TTM_{t-4}
    其中 OR_TTM 为单季度营业收入的滚动4期和。
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_income_factors = ["revenue"]

    def _calculate(self, start_time, end_time):
        # 1) 取数：累计营业收入
        income_data = self.fetch_income_data(start_time, end_time)
        revenue_cumsum = income_data.xs("revenue", level=0)

        # 2) 单季度营业收入（日频近似）
        revenue_quarter = cumulative_to_single_quarter(revenue_cumsum)

        # 3) 营业收入 TTM（日频近似）
        revenue_ttm = rolling_sum_quarters(revenue_quarter, n_quarters=4)

        # 4) 去年同期 TTM（交易日偏移法）
        revenue_ttm_lag = lag_by_quarters(revenue_ttm, 4)

        # 5) TTM 同比（日频近似）
        or_yoy = (revenue_ttm - revenue_ttm_lag) / revenue_ttm_lag.replace(0, np.nan)

        or_yoy = or_yoy.replace([np.inf, -np.inf], np.nan)
        return or_yoy


# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "OR_YOY",
    "creator": "zhangyudong",
    "create_date": "2026.01.08",
    "formula": "(当期营业收入TTM - 去年同期营业收入TTM) / 去年同期营业收入TTM",
    "description": "营业收入TTM同比增速，平滑季节性波动，衡量收入端年度变化趋势",
    "lookback_days": 750,
    "method": "OR_YOY_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子",
}


'''
# 说明：
# 1. 数据来源：
#    - revenue: 利润表 company_income_table，营业收入（累计值）
#
# 2. 计算逻辑：
#    - 单季度营业收入 = 累计值差分（Q1 直接用累计）
#    - OR_TTM = 单季度营业收入滚动4期求和
#    - OR_YOY = (OR_TTM_t - OR_TTM_{t-4}) / OR_TTM_{t-4}
#
# 3. 注意事项：
#    - 至少需要8个季度数据才能稳定计算同比
#    - 去年同期 TTM 接近0或为负时，增速可能异常；inf/-inf 已转为 NaN
#    - 若需改为环比或单季同比，可调整滚动或差分逻辑
'''
