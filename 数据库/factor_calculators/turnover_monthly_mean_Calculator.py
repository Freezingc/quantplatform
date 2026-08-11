from .base import FactorCalculator
import numpy as np
import pandas as pd

class turnover_monthly_mean_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["turnover_rate"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        turnover = data.xs('turnover_rate',level=0)

        # 3. 实现因子计算逻辑
        factor = turnover.rolling(20).mean()
        
        
        # 处理可能的除零错误和缺失值
        factor = factor.replace([np.inf, -np.inf], np.nan)
        
        return factor
'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "月均换手",
    "creator": "suyanning",
    "create_date": "2025.08.28",
    "formula": "SMA(TURNOVER,20)",
    "description": "加速换手-月均换手",
    "lookback_days": 20,
    "method": "turnover_monthly_mean_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": ""  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''