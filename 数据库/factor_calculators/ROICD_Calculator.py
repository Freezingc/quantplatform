from .base import FactorCalculator, lag_by_quarters
import numpy as np
import pandas as pd

class ROICD_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_financial_factors = ["roic"]

    def _calculate(self, start_time, end_time):
        # 1. 取数
        financial_data = self.fetch_financial_data(start_time, end_time)
        roic = financial_data.xs("roic", level=0)

        # ===============================
        # 2. 上年同期（交易日偏移法）
        # ===============================
        roic_lag = lag_by_quarters(roic, 4)

        # ===============================
        # 3. ROICD（日频近似）
        # ===============================
        roicd = roic - roic_lag

        roicd = roicd.replace([np.inf, -np.inf], np.nan)
        return roicd


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "ROICD",
    "creator": "zhangyudong",
    "create_date": "2026.01.04",
    "formula": "当期投资资本回报率(TTM) - 上期投资资本回报率(TTM)（4个季度前）",
    "description": "投资资本回报率变动，衡量企业资本使用效率的变化趋势。正值表示资本回报率提升，负值表示资本回报率下降",
    "lookback_days": 400,
    "method": "ROICD_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子"
}

# 说明：
# 1. 该因子直接使用 financial_metrics_table 中的 roic 字段
# 2. roic 字段来源于 Tushare 的 fina_indicator 接口
# 3. 计算同比变动：当期 - 4个季度前（即1年前同期）
# 4. 因子意义：
#    - 正值：投资资本回报率提升，企业资本使用效率改善
#    - 负值：投资资本回报率下降，企业资本使用效率恶化
#    - 变动幅度：反映资本使用效率变化的速度和程度
# 
# ROIC 的定义：
# ROIC = (净利润 + 利息费用 × (1 - 税率)) / 投资资本
# 其中：投资资本 = 股东权益 + 有息负债
# 
# ROIC vs ROA vs ROE:
# - ROA：衡量全部资产的回报率
# - ROE：衡量股东权益的回报率
# - ROIC：衡量投资资本（股权+债权）的回报率
# 
# ROICD 的应用：
# - 识别资本使用效率改善的公司
# - 评估管理层的资本配置能力变化
# - 与 ROAD、ROED 结合使用，全面评估盈利能力变化
'''


