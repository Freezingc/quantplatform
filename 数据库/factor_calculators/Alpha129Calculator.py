from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha129Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        
        # 3. 计算每日的收盘价变化
        close_change = close - close.shift(1)  # CLOSE - DELAY(CLOSE,1)
        
        # 4. 计算每日的负向变化绝对值
        # 如果收盘价变化为负，取绝对值；否则为0
        negative_change_abs = close_change.where(close_change < 0, 0).abs()
        
        # 5. 计算过去12天的滚动和
        # SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)
        factor = negative_change_abs.rolling(window=12, min_periods=1).sum()
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha129",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)",
        "description": "计算的是过去12个交易日中，每日收盘价下跌部分的绝对值累计总和",
        "lookback_days": 12,
        "method": "Alpha129Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
