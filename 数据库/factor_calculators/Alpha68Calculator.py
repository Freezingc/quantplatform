from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha68Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["high", "low", "vol"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        high = data.xs('high', level=0)
        vol = data.xs('vol', level=0)  # 假设成交量字段是'vol'

        # 3. 计算因子
        # 计算(HIGH+LOW)/2
        hl_avg = (high + low) / 2
        
        # 计算DELAY(HIGH,1)+DELAY(LOW,1))/2
        delay_hl_avg = (high.shift(1) + low.shift(1)) / 2
        
        # 计算(HIGH-LOW)
        hl_diff = high - low
        
        # 计算((HIGH+LOW)/2 - (DELAY(HIGH,1)+DELAY(LOW,1))/2)
        numerator = hl_avg - delay_hl_avg
        
        # 计算整个分子部分：(numerator * (HIGH-LOW))
        product = numerator * hl_diff
        
        # 计算整个表达式：(product / vol)
        expression = product / vol.replace(0, np.nan)  # 避免除以0
        
        # 计算SMA(expression, 15, 2)
        # SMA公式: SMA = alpha * current + (1-alpha) * previous_SMA
        # alpha = 2/(period+1) = 2/(15+1) = 0.125
        factor = expression.rolling(window=15, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*14)/15 if len(x) == 15 else x.mean()
        )
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha68",
        "creator": "chenxutian",
        "create_date": "2025.09.07",
        "formula": "(SUM(DTM,20)>SUM(DBM,20) ? (SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20) : (SUM(DTM,20)=SUM(DBM,20) ? 0:(SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))",
        "description": "alpha191因子68",
        "lookback_days": 15,
        "method": "Alpha68Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

