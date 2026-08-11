from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha14Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)

        # 3. 实现因子计算逻辑
        # 计算 DELAY(CLOSE, 5) - 将收盘价向后平移5个时间点
        close_delay_5 = close.shift(5)
        
        # 计算因子：CLOSE - DELAY(CLOSE, 5)
        factor = close - close_delay_5
        
        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha14",
    "creator": "chenxutian",
    "create_date": "2025.08.22",
    "formula": "CLOSE-DELAY(CLOSE,5)",
    "description": "alpha191因子14",
    "lookback_days": 5,
    "method": "Alpha14Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''