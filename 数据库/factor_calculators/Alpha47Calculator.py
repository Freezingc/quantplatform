from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha47Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "low", "high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)

        # 3. 实现因子计算逻辑
        # 计算TSMAX(HIGH,6)
        tsmax_high_6 = high.rolling(window=6,min_periods=1).max()
        
        # 计算TSMIN(LOW,6)
        tsmin_low_6 = low.rolling(window=6,min_periods=1).min()
        
        # 计算分子和分母
        numerator = tsmax_high_6 - close
        denominator = tsmax_high_6 - tsmin_low_6
        
        # 避免除零错误
        denominator = denominator.replace(0, float('nan'))
        
        # 计算原始值
        raw_value = (numerator / denominator) * 100
        
        # 使用ewm计算SMA(X,9,1)，等价于alpha=1/9的指数移动平均
        # SMA(X,N,M) = EMA(X, M/N) = EMA(X, 1/9)
        factor = raw_value.rolling(window=9, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*8)/9 if len(x) == 9 else x.mean()
        )
        return factor

'''
# 添加因子信息
{
    "factor_name": "Alpha47",
    "creator": "chenxutian",
    "create_date": "2025.08.27",
    "formula": "SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)",
    "description": "alpha191因子47",
    "lookback_days": 13,
    "method": "Alpha47Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''