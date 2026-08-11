from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha78Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low", "close", "high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        
        # 3. 计算典型价格(TP)
        typical_price = (high + low + close) / 3
        
        # 4. 计算12期移动平均
        ma_12 = typical_price.rolling(window=12,min_periods=1).mean()
        
        # 5. 计算分子部分
        numerator = typical_price - ma_12
        
        # 6. 计算分母部分
        # 6.1 计算收盘价与12期均值的绝对偏差
        abs_deviation = abs(close - ma_12)
        # 6.2 计算12期平均绝对偏差
        mean_abs_deviation = abs_deviation.rolling(window=12,min_periods=1).mean()
        # 6.3 计算分母
        denominator = 0.015 * mean_abs_deviation
        
        # 7. 计算因子值
        factor = numerator / denominator
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha78",
        "creator": "chenxutian",
        "create_date": "2025.09.07",
        "formula": "((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))",
        "description": "alpha191因子78",
        "lookback_days": 22,
        "method": "Alpha78Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.09"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

