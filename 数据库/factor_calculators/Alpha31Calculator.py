from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha31Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)

        # 3. 实现因子计算逻辑
        # 计算12日移动平均
        mean_close_12 = close.rolling(window=12).mean()
        
        # 计算因子：(CLOSE - MEAN(CLOSE,12)) / MEAN(CLOSE,12) * 100
        factor = (close - mean_close_12) / mean_close_12 * 100
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha31",
    "creator": "chenxutian",
    "create_date": "2025.08.25",
    "formula": "(CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100",
    "description": "alpha191因子31",
    "lookback_days": 11,
    "method": "Alpha31Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''