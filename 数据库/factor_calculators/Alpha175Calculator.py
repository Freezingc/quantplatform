from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha175Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        
        # 3. 计算因子逻辑
        # 计算DELAY(CLOSE, 1) - 前一日的收盘价
        delay_close = close.shift(1)
        
        # 计算三个组成部分：
        # 1) HIGH - LOW
        hl_diff = high - low
        
        # 2) ABS(DELAY(CLOSE, 1) - HIGH)
        delay_high_diff = (delay_close - high).abs()
        
        # 3) ABS(DELAY(CLOSE, 1) - LOW)
        delay_low_diff = (delay_close - low).abs()
        
        # 计算MAX(MAX(第一部分, 第二部分), 第三部分)
        # 先计算max1 = MAX(第一部分, 第二部分)
        max1 = pd.DataFrame(np.maximum(hl_diff.values, delay_high_diff.values), 
                        index=hl_diff.index, columns=hl_diff.columns)
        
        # 再计算max2 = MAX(max1, 第三部分)
        max2 = pd.DataFrame(np.maximum(max1.values, delay_low_diff.values),
                        index=max1.index, columns=max1.columns)
        
        # 对max2取6日均值
        factor = max2.rolling(window=6, min_periods=6).mean()
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha175",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)",
        "description": "每日真实波幅（True Range）的6日移动平均值",
        "lookback_days": 6,
        "method": "Alpha175Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "波动率因子"
    }
'''
