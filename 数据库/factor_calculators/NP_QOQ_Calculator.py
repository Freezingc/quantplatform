from .base import FactorCalculator, cumulative_to_single_quarter, lag_by_quarters
import numpy as np
import pandas as pd


class NP_QOQ_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_income_factors = ["n_income_attr_p"]

    def _calculate(self, start_time, end_time):
        # 1. 取数（累计净利润）
        income_data = self.fetch_income_data(start_time, end_time)
        n_income_cumsum = income_data.xs("n_income_attr_p", level=0)

        # 2. 单季度净利润（日频近似）
        n_income_quarter = cumulative_to_single_quarter(n_income_cumsum)

        # 3. 上季度净利润（交易日偏移法）
        n_income_quarter_lag = lag_by_quarters(n_income_quarter, 1)

        # 4. 单季度环比增长率（日频近似）
        np_qoq = (
            n_income_quarter - n_income_quarter_lag
        ) / n_income_quarter_lag.replace(0, np.nan)

        np_qoq = np_qoq.replace([np.inf, -np.inf], np.nan)
        return np_qoq


# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "NP_QOQ",
    "creator": "zhangyudong",
    "create_date": "2026.01.08",
    "formula": "(本季度净利润 - 上季度净利润) / 上季度净利润",
    "description": "净利润单季度环比增速，刻画最新季度盈利的环比变化，敏感捕捉业绩边际改善或走弱",
    "lookback_days": 400,
    "method": "NP_QOQ_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子",
}


'''
# 说明：
# 1. 数据来源：
#    - n_income_attr_p: 利润表 company_income_table，归母净利润（累计值）
#
# 2. 计算逻辑：
#    - 单季度净利润 = 当期累计净利润 - 上期累计净利润
#      * Q1: 直接用累计值
#    - 上季度净利润 = 单季度净利润_{t-1}
#    - NP_QOQ = (单季度净利润_t - 单季度净利润_{t-1}) / 单季度净利润_{t-1}
#
# 3. 注意事项：
#    - 需要至少2个连续季度数据才能计算
#    - 若上季度净利润接近0或为负，环比增速可能出现极端值或不可比
#    - 建议与同比类因子（如 NP_Q_YOY）结合查看，区分季节性与真实拐点
'''

