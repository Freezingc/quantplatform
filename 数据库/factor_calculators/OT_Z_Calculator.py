from .base import (
    FactorCalculator,
    cumulative_to_single_quarter,
    lag_by_quarters,
    rolling_sum_quarters,
    rolling_mean_std_quarters,
)
import numpy as np
import pandas as pd


class OT_Z_Calculator(FactorCalculator):
    """
    经营税 TTM 环比增速的时序标准分数：
    z_t = (g_t - mean(g_{t-w+1:t})) / std(g_{t-w+1:t})
    其中 g_t 为经营税 TTM 的环比增速，窗口默认 8 个报告期（约 2 年）。
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_cashflow_factors = ["c_paid_for_taxes"]
        self.window = 8  # 报告期滚动窗口

    def _calculate(self, start_time, end_time):
        # 1) 取数：累计经营税（支付的各项税费）
        cashflow_data = self.fetch_cashflow_data(start_time, end_time)
        taxes_cumsum = cashflow_data.xs("c_paid_for_taxes", level=0)

        # 2) 单季度经营税（日频近似）
        taxes_quarter = cumulative_to_single_quarter(taxes_cumsum)

        # 3) 经营税 TTM（日频近似）
        taxes_ttm = rolling_sum_quarters(taxes_quarter, n_quarters=4)

        # 4) 经营税 TTM 环比增速（交易日偏移法）
        taxes_ttm_lag = lag_by_quarters(taxes_ttm, 1)
        taxes_ttm_qoq = (taxes_ttm - taxes_ttm_lag) / taxes_ttm_lag.replace(0, np.nan)

        # 5) 时序标准分数（滚动窗口）
        rolling_mean, rolling_std = rolling_mean_std_quarters(
            taxes_ttm_qoq,
            window_quarters=self.window,
            min_quarters=4,
        )
        ot_z = (taxes_ttm_qoq - rolling_mean) / rolling_std

        ot_z = ot_z.replace([np.inf, -np.inf], np.nan)
        return ot_z


# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "OT_Z",
    "creator": "zhangyudong",
    "create_date": "2026.01.08",
    "formula": "(经营税TTM环比增速 - 均值) / 标准差；均值/标准差为过去8个报告期滚动",
    "description": "经营税TTM环比增速的时序标准分数，衡量企业纳税现金流变动的相对强弱",
    "lookback_days": 750,
    "method": "OT_Z_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子",
}


'''
# 说明：
# 1. 数据来源：
#    - c_paid_for_taxes: 现金流量表 cashflow_table，支付的各项税费（累计值）
#
# 2. 计算逻辑：
#    - 单季度经营税 = 累计值差分（Q1 直接用累计）
#    - 经营税TTM = 单季度经营税滚动4期求和
#    - 环比增速 = (TTM_t - TTM_{t-1}) / TTM_{t-1}
#    - OT_Z：对环比增速做时序 z-score，窗口 8 报告期（min_periods=4）
#
# 3. 注意事项：
#    - TTM_{t-1} 接近0或为负时，增速可能异常；std 为 0 时会产生 inf，已转为 NaN
#    - 至少需要5个季度才能计算环比，8个季度后 z-score 更稳定
#    - 若需改窗口或口径（如同比/单季），调整窗口或增速定义即可
'''
