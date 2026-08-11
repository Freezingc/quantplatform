from .base import FactorCalculator, cumulative_to_single_quarter, lag_by_quarters, rolling_sum_quarters
import numpy as np
import pandas as pd


class OCF_YOY_Calculator(FactorCalculator):

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_cashflow_factors = ["n_cashflow_act"]

    def _calculate(self, start_time, end_time):
        # 1) 取数：累计经营现金流
        cashflow_data = self.fetch_cashflow_data(start_time, end_time)
        ocf_cumsum = cashflow_data.xs("n_cashflow_act", level=0)

        # 2) 单季度经营现金流（日频近似）
        ocf_quarter = cumulative_to_single_quarter(ocf_cumsum)

        # 3) 经营现金流 TTM（日频近似）
        ocf_ttm = rolling_sum_quarters(ocf_quarter, n_quarters=4)

        # 4) 去年同期 TTM（交易日偏移法）
        ocf_ttm_lag = lag_by_quarters(ocf_ttm, 4)

        # 5) TTM 同比（日频近似）
        ocf_yoy = (ocf_ttm - ocf_ttm_lag) / ocf_ttm_lag.replace(0, np.nan)

        ocf_yoy = ocf_yoy.replace([np.inf, -np.inf], np.nan)
        return ocf_yoy


# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "OCF_YOY",
    "creator": "zhangyudong",
    "create_date": "2026.01.08",
    "formula": "(当期经营现金流TTM - 去年同期经营现金流TTM) / 去年同期经营现金流TTM",
    "description": "经营现金流TTM同比增速，平滑季节性波动，衡量经营现金流的年度变化趋势",
    "lookback_days": 750,
    "method": "OCF_YOY_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子",
}


'''
# 说明：
# 1. 数据来源：
#    - n_cashflow_act: 现金流量表 cashflow_table，经营活动产生的现金流量净额（累计值）
#
# 2. 计算逻辑：
#    - 单季度经营现金流 = 累计值差分（Q1 直接用累计）
#    - OCF_TTM = 单季度经营现金流滚动4期求和
#    - OCF_YOY = (OCF_TTM_t - OCF_TTM_{t-4}) / OCF_TTM_{t-4}
#
# 3. 注意事项：
#    - 至少需要8个季度数据才能稳定计算同比
#    - 去年同期 TTM 接近0或为负时，增速可能异常；inf/-inf 已转为 NaN
#    - 若需改窗口或口径（如改为环比/单季同比），调整 rolling 或差分逻辑即可
'''

