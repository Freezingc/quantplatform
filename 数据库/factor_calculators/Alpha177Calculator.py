from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha177Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        high = data.xs('high', level=0)
        
        # 3. 实现因子计算逻辑
        # 计算过去20日的最高价
        high_day_20 = high.rolling(window=20).max()
        
        # 计算每个日期过去20日的最高价是在哪一天出现的
        # 这里需要计算每个时间点上，过去20天中最高价出现的相对位置
        def get_high_day(arr):
            # 找到过去20天内最高值的索引（0表示最远，19表示最近）
            if len(arr) < 20 or arr.isna().any():
                return np.nan
            # 找到最高值最后一次出现的位置（最近的位置）
            max_val = arr.max()
            # 从后往前找，找到第一个最大值的位置
            for i in range(len(arr)-1, -1, -1):
                if arr.iloc[i] == max_val:
                    return len(arr) - 1 - i  # 转换为距离当前的天数
        
        # 计算HIGHDAY(HIGH,20)
        high_day = high.rolling(window=20).apply(get_high_day, raw=False)
        
        # 计算因子 ((20-HIGHDAY(HIGH,20))/20)*100
        factor = ((20 - high_day) / 20) * 100
        
        return factor

'''
# 添加因子信息
    {
        "factor_name": "Alpha177",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": ((20-HIGHDAY(HIGH,20))/20)*100",
        "description": "衡量过去20天内股票最高价出现的日期距离当前的远近程度，数值越小表示近期出现过新高",
        "lookback_days": 19,
        "method": "Alpha177Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
