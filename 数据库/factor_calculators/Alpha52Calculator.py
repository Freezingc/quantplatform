from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha52Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low", "high", "close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)

        # 计算(HIGH+LOW+CLOSE)/3
        avg_price = (high + low + close) / 3
        
        # 计算延迟一天的平均价格
        delay_avg = avg_price.shift(1)
        
        # 计算分子部分：MAX(0, HIGH - DELAY_AVG)
        numerator_part = np.maximum(0, high - delay_avg)
        
        # 计算分母部分：MAX(0, DELAY_AVG - LOW)
        denominator_part = np.maximum(0, delay_avg - low)
        
        # 计算26天滚动求和（分子和分母）
        numerator = numerator_part.rolling(window=26).sum()
        denominator = denominator_part.rolling(window=26).sum()
        
        # 计算最终因子值（乘以100）
        factor = (numerator / denominator) * 100
        
        # 处理分母为0的情况（设置为NaN）
        factor[denominator == 0] = np.nan
        
        return factor
'''
# 添加因子信息
    {
        "factor_name": "Alpha52",
        "creator": "chenxutian",
        "create_date": "2025.08.27",
        "formula": "SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-L),26)* 100",
        "description": "alpha191因子52",
        "lookback_days": 26,
        "method": "Alpha52Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''
