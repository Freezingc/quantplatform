from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha84Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol", "close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)   # 需要vol数据

        # 实现因子计算逻辑
        # 计算CLOSE与前一天CLOSE的比较
        close_diff = close - close.shift(1)
        
        # 根据条件计算每日的vol符号
        vol_sign = pd.DataFrame(0, index=close.index, columns=close.columns)
        vol_sign[close_diff > 0] = vol[close_diff > 0]      # CLOSE>DELAY(CLOSE,1)
        vol_sign[close_diff < 0] = -vol[close_diff < 0]     # CLOSE<DELAY(CLOSE,1)
        
        # 计算20日滚动求和
        factor = vol_sign.rolling(window=20).sum()
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha84",
        "creator": "chenxutian",
        "create_date": "2025.10.05",
        "formula": "SUM((CLOSE>DELAY(CLOSE,1)?vol:(CLOSE<DELAY(CLOSE,1)?-vol:0)),20)",
        "description": "alpha191因子84",
        "lookback_days": 20,
        "method": "Alpha84Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.09"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

