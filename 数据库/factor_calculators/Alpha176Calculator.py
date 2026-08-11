from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha176Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high",'low']  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        vol = data.xs('vol', level=0)  # 需要vol数据

        # 3. 计算第一个序列：(CLOSE - TSMIN(LOW, 12)) / (TSMAX(HIGH, 12) - TSMIN(LOW, 12))
        # 计算12期最低价
        tmin_low_12 = low.rolling(window=12, min_periods=1).min()
        
        # 计算12期最高价
        tmax_high_12 = high.rolling(window=12, min_periods=1).max()
        
        # 计算分子：CLOSE - TSMIN(LOW, 12)
        numerator = close - tmin_low_12
        
        # 计算分母：TSMAX(HIGH, 12) - TSMIN(LOW, 12)
        denominator = tmax_high_12 - tmin_low_12
        
        # 避免除零，将分母为0的替换为NaN
        denominator = denominator.replace(0, np.nan)
        
        # 计算第一个序列
        series1 = numerator / denominator
        
        # 4. 对两个序列进行排名
        # 对series1进行横截面排名
        rank1 = series1.rank(axis=1, method='average', pct=True)  # 使用百分比排名，范围[0,1]
        
        # 对vol进行横截面排名
        rank2 = vol.rank(axis=1, method='average', pct=True)  # 使用百分比排名，范围[0,1]
        
        # 5. 计算6期相关系数
        # 使用rolling计算相关性
        factor = rank1.rolling(window=6, min_periods=1).corr(rank2)
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha176",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "CORR(RANK(((CLOSE - TSMIN(LOW, 12)) / (TSMAX(HIGH, 12) - TSMIN(LOW,12)))), RANK(VOLUME), 6)",
        "description": "成交量20日标准差",
        "lookback_days": 18,
        "method": "Alpha176Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
