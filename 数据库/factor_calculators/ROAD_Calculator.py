from .base import FactorCalculator, lag_by_quarters
import numpy as np
import pandas as pd

from .base import FactorCalculator
import numpy as np
import pandas as pd

class ROAD_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        # 使用 financial_metrics_table 中已有的 ROA 数据
        self.required_financial_factors = ["roa"]

    def _calculate(self, start_time, end_time):
        # 1. 取数
        financial_data = self.fetch_financial_data(start_time, end_time)
        roa = financial_data.xs("roa", level=0)

        # ===============================
        # 2. 去年同期 ROA（交易日偏移法）
        # ===============================
        roa_lag = lag_by_quarters(roa, 4)

        # ===============================
        # 3. ROAD（日频近似）
        # ===============================
        road = roa - roa_lag

        road = road.replace([np.inf, -np.inf], np.nan)
        return road


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "ROAD",
    "creator": "zhangyudong",
    "create_date": "2025.12.17",
    "formula": "当期资产回报率 - 上期资产回报率（4个季度前）",
    "description": "资产回报率变动，衡量企业盈利能力的变化趋势。正值表示盈利能力提升，负值表示盈利能力下降",
    "lookback_days": 400,
    "method": "ROAD_Calculator",
    "start_time": "",  # 可以为空，默认从2020.01.01开始
    "end_time": "",  # 可以为空，默认为当日日期
    "category": "基本面因子",
}

# 说明：
# 1. 该因子直接使用 financial_metrics_table 中的 roa 字段
# 2. roa 字段来源于 Tushare 的 fina_indicator 接口
# 3. 计算同比变动：当期 - 4个季度前（即1年前同期）
# 4. 因子意义：
#    - 正值：资产回报率提升，企业盈利能力改善
#    - 负值：资产回报率下降，企业盈利能力恶化
#    - 变动幅度：反映盈利能力变化的速度和程度
# 
# 可选的其他 ROA 字段：
# - roa2_yearly: 年化总资产报酬率
# - roa_yearly: 年化总资产净利率
# - roa_dp: 总资产净利率(杜邦分析)
# 
# 如需使用其他字段，修改 self.required_financial_factors = ["字段名"] 即可
'''

