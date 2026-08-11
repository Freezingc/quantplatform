from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha9Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low", "high", "vol"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        high = data.xs('high', level=0)
        vol = data.xs('vol', level=0) 

        # 3. 计算因子
        # 计算(HIGH+LOW)/2
        hl_avg = (high + low) / 2

        # 计算DELAY(HIGH,1)+DELAY(LOW,1))/2
        delay_hl_avg = (high.shift(1) + low.shift(1)) / 2
        
        # 计算(HIGH-LOW)
        h_l = high - low
        
        # 计算((HIGH+LOW)/2 - (DELAY(HIGH,1)+DELAY(LOW,1))/2)
        numerator = hl_avg - delay_hl_avg
        
        # 计算整个分子部分：((...))*(HIGH-LOW)
        total_numerator = numerator * h_l

        # 计算整个表达式：(分子)/vol
        raw_factor = total_numerator / vol

        # 计算SMA(raw_factor, 7, 2)
        # SMA(X,N,M) = [X*M + REF(SMA(X,N,M),1)*(N-M)]/N
        # 这里实现SMA7,2
        sma_factor = raw_factor.rolling(window=7, min_periods=1).apply(
            lambda x: (x[-1]*2 + x[:-1].mean()*5)/7 if len(x) == 7 else x.mean()
        )
        
        factor = sma_factor
        return factor



'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha9",
    "creator": "chenxutian",
    "create_date": "2025.08.21",
    "formula": "SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/vol,7,2)",
    "description": "alpha191因子9",
    "lookback_days": 6,
    "method": "Alpha9Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''