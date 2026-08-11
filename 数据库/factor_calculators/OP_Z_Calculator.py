from .base import FactorCalculator, cumulative_to_single_quarter, lag_by_quarters, rolling_mean_std_quarters
import numpy as np
import pandas as pd


class OP_Z_Calculator(FactorCalculator):
    """
    营业利润增速的时序标准分数：
    z_t = (g_t - mean(g_{t-w+1:t})) / std(g_{t-w+1:t})
    其中 g_t 为单季度营业利润同比增速，窗口默认 8 个报告期（约 2 年）。
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_income_factors = ["operate_profit"]
        self.window = 8  # 报告期滚动窗口

    def _calculate(self, start_time, end_time):
        # 1) 取数：累计营业利润
        income_data = self.fetch_income_data(start_time, end_time)
        op_cumsum = income_data.xs("operate_profit", level=0)

        # 2) 单季度营业利润（日频近似）
        op_quarter = cumulative_to_single_quarter(op_cumsum)

        # 3) 去年同季度（交易日偏移法）
        op_quarter_lag = lag_by_quarters(op_quarter, 4)

        # 4) 单季度营业利润同比增速 g_t（日频近似）
        op_growth = (
            op_quarter - op_quarter_lag
        ) / op_quarter_lag.replace(0, np.nan)

        # 5) 时序标准分数（滚动窗口，按交易日近似报告期）
        rolling_mean, rolling_std = rolling_mean_std_quarters(
            op_growth,
            window_quarters=self.window,
            min_quarters=4,
        )
        op_z = (op_growth - rolling_mean) / rolling_std

        op_z = op_z.replace([np.inf, -np.inf], np.nan)
        return op_z


# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "OP_Z",
    "creator": "zhangyudong",
    "create_date": "2026.01.08",
    "formula": "(营业利润增速 - 均值) / 标准差；增速取单季度同比，均值/标准差为过去8个报告期滚动",
    "description": "营业利润增速的时序标准分数，衡量当前主营盈利增速相对自身历史的高低位置，识别业绩拐点与异常加速/放缓",
    "lookback_days": 400,
    "method": "OP_Z_Calculator",
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
#    - 单季度营业利润：累计值按报告期差分
#    - 增速 g_t：单季度营业利润同比 = (利润_t - 利润_{t-4}) / 利润_{t-4}
#    - OP_Z：对 g_t 做时序 z-score，窗口 8 报告期（min_periods=4）
#
# 3. 注意事项：
#    - 去年同期为负或接近0时，增速可能异常；std 为 0 时会产生 inf，已转为 NaN
#    - 至少需要 5 个季度数据才能开始计算增速，8 个季度后 z-score 更稳定
#    - 如需改窗口或增速口径（环比/TTM），调整 self.window 或增速定义即可
'''

