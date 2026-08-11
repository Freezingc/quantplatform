from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha79Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)

        # 3. 实现因子计算逻辑
        # 计算每日收盘价变化
        close_diff = close - close.shift(1)
        
        # 计算正变化部分
        positive_diff = close_diff.clip(lower=0)
        
        # 计算绝对值变化
        abs_diff = close_diff.abs()
        
        # 计算12日平滑移动平均（SMA）
        # 这里使用rolling和mean来近似SMA(...,12,1)
        sma_positive = positive_diff.rolling(window=12, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*11)/12 if len(x) == 12 else x.mean()
        )
        sma_abs = abs_diff.rolling(window=12, min_periods=1).mean()
        
        # 计算因子值
        factor = sma_positive / sma_abs * 100
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha79",
        "creator": "chenxutian",
        "create_date": "2025.09.07",
        "formula": "SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100",
        "description": "alpha191因子79",
        "lookback_days": 12,
        "method": "Alpha79Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.09"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

