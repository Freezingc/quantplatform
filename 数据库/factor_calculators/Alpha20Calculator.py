from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha20Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        # 3. 实现因子计算逻辑
        # DELAY(CLOSE,6) 表示6天前的收盘价
        close_delay_6 = close.shift(6)  # 使用shift函数实现延迟操作
        
        # 计算因子：(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100
        factor = (close - close_delay_6) / close_delay_6 * 100
        
        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha20",
    "creator": "chenxutian",
    "create_date": "2025.08.22",
    "formula": "(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100",
    "description": "alpha191因子20​",
    "lookback_days": 6,
    "method": "Alpha20Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''