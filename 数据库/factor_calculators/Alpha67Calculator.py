from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha67Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)

        # 3. 实现因子计算逻辑
        # 计算价格变化
        delta = close - close.shift(1)
        
        # 计算上涨部分和下跌部分的绝对值
        up = delta.where(delta > 0, 0)
        down = delta.abs()
        
        # 计算24日的SMA
        sma_up = up.rolling(window=24, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*23)/24 if len(x) == 24 else x.mean()
        )
        sma_down = down.rolling(window=24, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*23)/24 if len(x) == 24 else x.mean()
        )
        # 计算RSI因子
        factor = sma_up / sma_down * 100
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha67",
        "creator": "chenxutian",
        "create_date": "2025.09.07",
        "formula": "SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100",
        "description": "alpha191因子67",
        "lookback_days": 6,
        "method": "Alpha67Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

