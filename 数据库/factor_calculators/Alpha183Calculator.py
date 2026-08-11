from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha183Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        close = data.xs('close', level=0)
        
        # 3. 实现因子计算逻辑
        # 计算24日均值
        mean_close = close.rolling(window=24).mean()
        
        # 计算CLOSE与均值的差
        diff = close - mean_close
        
        # 计算SUMAC（累积和）
        sumac = diff.expanding().sum()
        
        # 计算SUMAC在24日窗口内的最大值和最小值
        max_sumac = sumac.rolling(window=24).max()
        min_sumac = sumac.rolling(window=24).min()
        
        # 计算24日标准差
        std_close = close.rolling(window=24).std()
        
        # 计算最终因子：(MAX - MIN) / STD
        factor = (max_sumac - min_sumac) / std_close
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha183",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)",
        "description": "衡量的是价格相对于24日移动平均线的累计偏离波动范围，用其最大值与最小值之差除以价格波动率来标准化",
        "lookback_days": 23,
        "method": "Alpha183Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
