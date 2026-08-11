from .base import FactorCalculator, cumulative_to_single_quarter, lag_by_quarters
import numpy as np
import pandas as pd


class OCF_Q_YOY_Calculator(FactorCalculator):
    """
    单季度经营现金流同比增速（日频近似版）：
    (本季度经营现金流 - 去年同季度经营现金流) / 去年同季度经营现金流

    说明：
    - 使用累计经营现金流 n_cashflow_act 差分得到单季度值
    - 在日频交易日序列上，使用约60交易日作为季度偏移来近似“去年同季度”
    - 这样每个交易日都有值，且在新公告到来时会逐步更新，而不是整个季度完全相同
    """

    QUARTER_DAYS_APPROX = 60      # 平均一个季度交易日数（可微调 58~63）
    LOOKBACK_QUARTERS = 4         # 回看几个季度（同比通常是4）

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_cashflow_factors = ["n_cashflow_act"]

    def _calculate(self, start_time, end_time):
        # 1. 获取累计经营活动现金流（日频，已 reindex 到交易日 + ffill）
        cashflow_data = self.fetch_cashflow_data(start_time, end_time)
        ocf_cum = cashflow_data.xs("n_cashflow_act", level=0)   # shape: (交易日, ts_code)

        # 2. 转换为单季度值（日频近似）
        single_quarter = cumulative_to_single_quarter(ocf_cum)

        # 3. 计算同比（交易日偏移法）
        lag_single_quarter = lag_by_quarters(single_quarter, 4)

        # 同比增速（vectorized，避免逐列操作）
        ocf_q_yoy = (single_quarter - lag_single_quarter) / lag_single_quarter.replace(0, np.nan)

        # 4. 最终清理（copy 确保输出干净）
        ocf_q_yoy = ocf_q_yoy.replace([np.inf, -np.inf], np.nan).copy()

        return ocf_q_yoy

# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "OCF_Q_YOY",
    "creator": "zhangyudong",
    "create_date": "2026.01.08",
    "formula": "(本季度经营现金流 - 去年同季度经营现金流) / 去年同季度经营现金流",
    "description": "单季度经营现金流同比增速，衡量经营现金流的年度提升或恶化程度，捕捉经营质量变化",
    "lookback_days": 400,
    "method": "OCF_Q_YOY_Calculator",
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
#    - 单季度经营现金流 = 当期累计 - 上期累计（Q1 直接用累计值）
#    - 去年同季度 = 单季度经营现金流_{t-4}
#    - OCF_Q_YOY = (单季度经营现金流_t - 单季度经营现金流_{t-4}) / 单季度经营现金流_{t-4}
#
# 3. 注意事项：
#    - 上年同期为负或接近0时，增速可能异常；结果中的 inf 已转为 NaN
#    - 至少需要5个季度数据才能稳定计算
#    - 如需改为 TTM 或环比，可调整计算部分
'''

