from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha188Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","low","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        high = data.xs('high', level=0)
        
        # 3. 计算因子
        high_low_diff = high - low
        
        # 计算加权移动平均 (SMA with weight=2)
        # 注意：这里的实现基于你的公式描述
        sma_hl = high_low_diff.rolling(window=11, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*9)/11 if len(x) == 11 else x.mean()
        )  # 对于 span=11, alpha=2/(span+1)
        
        # 计算因子
        factor = ((high_low_diff - sma_hl) / sma_hl) * 100
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha188",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100",
        "description": "当前股价波动幅度（最高-最低价差）相对于其近期波动均值的偏离程度百分比",
        "lookback_days": 10,
        "method": "Alpha188Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "波动率因子"
    }
'''
