from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha82Calculator(FactorCalculator):
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

        # 3. 计算因子逻辑
        # 计算TSMAX(HIGH,6)
        tsmax_high_6 = high.rolling(window=6).max()
        
        # 计算TSMIN(LOW,6)
        tsmin_low_6 = low.rolling(window=6).min()
        
        # 计算分子：(TSMAX(HIGH,6)-CLOSE)
        numerator = tsmax_high_6 - close
        
        # 计算分母：(TSMAX(HIGH,6)-TSMIN(LOW,6))
        denominator = tsmax_high_6 - tsmin_low_6
        
        # 避免除零错误，将分母中的0替换为NaN
        denominator = denominator.replace(0, float('nan'))
        
        # 计算原始值：(TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100
        raw_value = (numerator / denominator) * 100
        
        # 计算SMA(raw_value, 20, 1)
        # SMA(X, N, M) = M/N * X + (N-M)/N * SMA(X, N, M)[-1]
        # 这里使用EWMA（指数加权移动平均）来实现SMA
        factor = raw_value.rolling(window=20, min_periods=1).apply(
            lambda x: (x[-1]*1 + x[:-1].mean()*19)/20 if len(x) == 20 else x.mean()
        )
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha82",
        "creator": "chenxutian",
        "create_date": "2025.10.05",
        "formula": "SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)",
        "description": "alpha191因子82",
        "lookback_days": 24,
        "method": "Alpha82Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.09"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

