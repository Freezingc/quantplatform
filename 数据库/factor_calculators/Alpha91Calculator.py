from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha91Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low","close","vol"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)
        
        # 3. 计算因子
        # CLOSE - MAX(CLOSE, 5)
        max_close_5 = close.rolling(window=5).max()
        close_minus_max = close - max_close_5
        
        # RANK((CLOSE - MAX(CLOSE, 5)))
        rank_close_minus_max = close_minus_max.rank(axis=1, pct=True)
        
        # MEAN(vol,40)
        mean_vol_40 = vol.rolling(window=40).mean()
        
        # CORR((MEAN(vol,40)), LOW, 5) - 使用rolling correlation
        corr_result = mean_vol_40.rolling(window=5).corr(low)
        
        # RANK(CORR((MEAN(vol,40)), LOW, 5))
        rank_corr = corr_result.rank(axis=1, pct=True)
        
        # 最终因子计算
        factor = rank_close_minus_max * rank_corr * (-1)
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha91",
        "creator": "chenxutian",
        "create_date": "2025.10.15",
        "formula": "((RANK((CLOSE - MAX(CLOSE, 5)))*RANK(CORR((MEAN(vol,40)), LOW, 5))) * -1)",
        "description": "alpha191因子91",
        "lookback_days": 43,
        "method": "Alpha91Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''
