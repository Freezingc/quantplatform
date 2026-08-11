from .base import FactorCalculator, lag_by_quarters
import numpy as np
import pandas as pd

class ROED_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        # 使用 financial_metrics_table 中已有的 ROE 数据
        self.required_financial_factors = ["roe"]

    def _calculate(self, start_time, end_time):
        # 1. 取数
        financial_data = self.fetch_financial_data(start_time, end_time)
        roe = financial_data.xs("roe", level=0)

        # ===============================
        # 2. 去年同期 ROE（交易日偏移法）
        # ===============================
        roe_lag = lag_by_quarters(roe, 4)

        # ===============================
        # 3. ROED（日频近似）
        # ===============================
        roed = roe - roe_lag

        roed = roed.replace([np.inf, -np.inf], np.nan)
        return roed
'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "ROED",
    "creator": "zhangyudong",
    "create_date": "2025.12.17",
    "formula": "当期净资产收益率TTM - 上期净资产收益率TTM（4个季度前）",
    "description": "净资产收益率变动，衡量企业股东权益回报能力的变化趋势。正值表示盈利能力提升，负值表示盈利能力下降",
    "lookback_days": 400,
    "method": "ROED_Calculator",
    "start_time": "",  # 可以为空，默认从2020.01.01开始
    "end_time": "",  # 可以为空，默认为当日日期
    "category": "基本面因子",
}

# 说明：
# 1. 该因子直接使用 financial_metrics_table 中的 roe 字段
# 2. roe 字段来源于 Tushare 的 fina_indicator 接口
# 3. 计算同比变动：当期 - 4个季度前（即1年前同期）
# 4. 因子意义：
#    - 正值：净资产收益率提升，股东回报能力改善
#    - 负值：净资产收益率下降，股东回报能力恶化
#    - 变动幅度：反映股东权益回报变化的速度和程度
# 5. ROE 是衡量企业盈利能力的核心指标，其变动反映企业经营质量的变化
# 
# financial_metrics_table 中相关的 ROE 字段：
# - roe: 净资产收益率（标准计算）
# - roe_waa: 加权平均净资产收益率
# - roe_dt: 净资产收益率(扣除非经常损益)
# - roe_yearly: 年化净资产收益率
# - roe_avg: 平均净资产收益率
# - q_roe: 单季度净资产收益率
# - q_dt_roe: 单季度净资产收益率(扣除非经常损益)
# 
# 如需使用其他字段，修改 self.required_financial_factors = ["字段名"] 即可
'''

