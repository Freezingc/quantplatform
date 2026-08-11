from .base import FactorCalculator, cumulative_to_single_quarter, lag_by_quarters
import numpy as np
import pandas as pd


class OR_Q_YOY_Calculator(FactorCalculator):
    """
    单季度营业收入同比增速：
    (本季度营业收入 - 去年同季度营业收入) / 去年同季度营业收入
    其中营业收入取财报累计口径 revenue 差分得到单季度。
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

        # 3) 去年同季度（交易日偏移法）
        revenue_quarter_lag = lag_by_quarters(revenue_quarter, 4)

        # 4) 单季度同比（日频近似）
        or_q_yoy = (
            revenue_quarter - revenue_quarter_lag
        ) / revenue_quarter_lag.replace(0, np.nan)

        or_q_yoy = or_q_yoy.replace([np.inf, -np.inf], np.nan)
        return or_q_yoy


# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "OR_Q_YOY",
    "creator": "zhangyudong",
    "create_date": "2026.01.08",
    "formula": "(本季度营业收入 - 去年同季度营业收入) / 去年同季度营业收入",
    "description": "单季度营业收入同比增速，衡量收入端的年度增长情况，反映主营业务扩张速度",
    "lookback_days": 400,
    "method": "OR_Q_YOY_Calculator",
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
#    - 去年同季度 = 单季度营业收入_{t-4}
#    - OR_Q_YOY = (单季度营业收入_t - 单季度营业收入_{t-4}) / 单季度营业收入_{t-4}
#
# 3. 注意事项：
#    - 去年同季度接近0或为负时，增速可能异常；inf/-inf 已转为 NaN
#    - 至少需要5个季度数据才能稳定计算
#    - 若需改为 TTM 或环比，可调整滚动或差分逻辑
'''
