from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha29Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "vol"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)  # 需要获取vol数据

        # 3. 实现因子计算逻辑
        # 计算6期前的收盘价
        close_delay_6 = close.shift(6)
        
        # 计算价格变化率：(CLOSE - DELAY(CLOSE,6)) / DELAY(CLOSE,6)
        price_change_ratio = (close - close_delay_6) / close_delay_6
        
        # 计算最终因子：价格变化率乘以成交量
        factor = price_change_ratio * vol
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha29",
    "creator": "chenxutian",
    "create_date": "2025.08.25",
    "formula": "(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*vol",
    "description": "alpha191因子29​",
    "lookback_days": 6,
    "method": "Alpha29Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''