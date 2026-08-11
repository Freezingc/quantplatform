from .base import FactorCalculator, cumulative_to_single_quarter, lag_by_quarters, rolling_sum_quarters
import numpy as np
import pandas as pd


class OP_YOY_Calculator(FactorCalculator):
    """
    营业利润 TTM 同比：
    (OP_TTM_t - OP_TTM_{t-4}) / OP_TTM_{t-4}
    其中 OP_TTM 为单季度营业利润的滚动4期和。
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_income_factors = ["operate_profit"]

    def _calculate(self, start_time, end_time):
        # 1) 取数：累计营业利润
        income_data = self.fetch_income_data(start_time, end_time)
        op_cumsum = income_data.xs("operate_profit", level=0)

        # 2) 单季度营业利润（日频近似）
        op_quarter = cumulative_to_single_quarter(op_cumsum)

        # 3) 营业利润 TTM（日频近似）
        op_ttm = rolling_sum_quarters(op_quarter, n_quarters=4)

        # 4) 去年同期 TTM（交易日偏移法）
        op_ttm_lag = lag_by_quarters(op_ttm, 4)

        # 5) TTM 同比（日频近似）
        op_yoy = (op_ttm - op_ttm_lag) / op_ttm_lag.replace(0, np.nan)

        op_yoy = op_yoy.replace([np.inf, -np.inf], np.nan)
        return op_yoy


# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "OP_YOY",
    "creator": "zhangyudong",
    "create_date": "2026.01.08",
    "formula": "(当期营业利润TTM - 去年同期营业利润TTM) / 去年同期营业利润TTM",
    "description": "营业利润TTM同比增速，平滑季节性波动，刻画主营盈利能力的年度变化趋势",
    "lookback_days": 750,
    "method": "OP_YOY_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子",
}


'''
# 说明：
# 1. 数据来源：
#    - operate_profit: 利润表 company_income_table，营业利润（累计值）
#
# 2. 计算逻辑：
#    - 单季度营业利润 = 累计值差分（Q1 直接用累计）
#    - OP_TTM = 单季度营业利润滚动4期求和
#    - OP_YOY = (OP_TTM_t - OP_TTM_{t-4}) / OP_TTM_{t-4}
#
# 3. 注意事项：
#    - 至少需要8个季度数据才能稳定计算同比
#    - 去年同期 TTM 接近0或为负时，增速可能异常；inf/-inf 已转为 NaN
#    - 若需改为环比或单季同比，可调整 rolling 或差分逻辑
'''

