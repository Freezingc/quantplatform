from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha152Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        close = data.xs('close', level=0)
        
        # 3. 分解计算步骤
        # SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12) - 
        # MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),9,1)
        
        # 步骤1: CLOSE/DELAY(CLOSE,9)
        close_ratio = close / close.shift(9)
        
        # 步骤2: DELAY(CLOSE/DELAY(CLOSE,9),1)
        close_ratio_delayed_1 = close_ratio.shift(1)
        
        # 步骤3: SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1)
        # SMA计算（简单移动平均），这里假设SMA(x,n,m)表示周期n的简单移动平均
        sma_9_1 = close_ratio_delayed_1.rolling(window=9, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*8)/9 if len(x) == 9 else x.mean()
        )

        
        # 步骤4: DELAY(SMA(...),1)
        sma_delayed_1 = sma_9_1.shift(1)
        
        # 步骤5: MEAN(DELAY(SMA(...),1),12)
        mean_12 = sma_delayed_1.rolling(window=12, min_periods=1).mean()
        
        # 步骤6: MEAN(DELAY(SMA(...),1),26)
        mean_26 = sma_delayed_1.rolling(window=26, min_periods=1).mean()
        
        # 步骤7: MEAN_12 - MEAN_26
        diff = mean_12 - mean_26
        
        # 步骤8: SMA(...,9,1) 对差值进行9日移动平均
        factor = diff.rolling(window=9, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*8)/9 if len(x) == 9 else x.mean()
        )

        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha152",
        "creator": "chenxutian",
        "create_date": "2026.02.04",
        "formula": "SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY (CLOSE,9),1),9,1),1),26),9,1)",
        "description": "基于价格比率9日移动平均的12日与26日移动平均的差值的9日移动平均",
        "lookback_days": 57,
        "method": "Alpha152Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
