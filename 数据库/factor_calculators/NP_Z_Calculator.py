from .base import FactorCalculator, cumulative_to_single_quarter, lag_by_quarters, rolling_mean_std_quarters
import numpy as np
import pandas as pd


class NP_Z_Calculator(FactorCalculator):
    """
    净利润增速的时序标准分数：
    z_t = (g_t - mean(g_{t-w+1:t})) / std(g_{t-w+1:t})
    其中 g_t 为单季度净利润同比增速，窗口默认 8 个报告期（约 2 年）。
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_income_factors = ["n_income_attr_p"]
        self.window = 8  # 报告期滚动窗口

    def _calculate(self, start_time, end_time):
        # 1) 取数：累计净利润
        income_data = self.fetch_income_data(start_time, end_time)
        n_income_cumsum = income_data.xs("n_income_attr_p", level=0)

        # 2) 单季度净利润（日频近似）
        n_income_quarter = cumulative_to_single_quarter(n_income_cumsum)

        # 3) 去年同季度净利润（交易日偏移法）
        n_income_quarter_lag = lag_by_quarters(n_income_quarter, 4)

        # 4) 单季度净利润同比增速 g_t（日频近似）
        np_growth = (
            n_income_quarter - n_income_quarter_lag
        ) / n_income_quarter_lag.replace(0, np.nan)

        # 5) 时序标准分数（滚动窗口，按交易日近似报告期）
        rolling_mean, rolling_std = rolling_mean_std_quarters(
            np_growth,
            window_quarters=self.window,
            min_quarters=4,
        )
        np_z = (np_growth - rolling_mean) / rolling_std

        np_z = np_z.replace([np.inf, -np.inf], np.nan)
        return np_z


# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "NP_Z",
    "creator": "zhangyudong",
    "create_date": "2026.01.08",
    "formula": "(净利润增速 - 均值) / 标准差；净利润增速取单季度同比，均值/标准差为过去8个报告期滚动",
    "description": "净利润增速的时序标准分数，衡量当前盈利增速相对自身历史的高低位置，捕捉业绩拐点和异常加速/放缓",
    "lookback_days": 400,
    "method": "NP_Z_Calculator",
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
#    - 单季度净利润：累计净利润按报告期差分
#    - 净利润增速 g_t：单季度净利润同比增速 = (净利_t - 净利_{t-4}) / 净利_{t-4}
#    - NP_Z：对 g_t 在窗口内做时序 z-score，窗口 8 个报告期（约 2 年），min_periods=4
#
# 3. 注意事项：
#    - 上年同期为负或接近0时，增速可能异常；z-score 会弱化部分极值但仍需结合业务判断
#    - std 为 0 时结果会产生 inf，已转为 NaN
#    - 如需改窗口大小或增速口径（改为环比/TTM），可调整 self.window 或增速定义
'''

