from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha60Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "low", "high", "vol"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        vol = data.xs('vol', level=0)
        
        # 3. 计算因子
        # 计算分子部分 (CLOSE-LOW)-(HIGH-CLOSE)
        numerator = (close - low) - (high - close)
        
        # 计算分母部分 (HIGH-LOW)
        denominator = high - low
        
        # 避免除以0的情况
        denominator = denominator.replace(0, float('nan'))
        
        # 计算核心部分 ((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*vol
        core = (numerator / denominator) * vol
        
        # 计算20日滚动求和
        factor = core.rolling(20).sum()
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha60",
        "creator": "chenxutian",
        "create_date": "2025.09.01",
        "formula": "SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,20)",
        "description": "alpha191因子60",
        "lookback_days": 19,
        "method": "Alpha60Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

