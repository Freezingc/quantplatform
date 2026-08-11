from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha89Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        
        # 4. 计算各个SMA分量
        sma_close_13 = close.rolling(window=13, min_periods=1).apply(
            lambda x: (x[-1]*2 + x[:-1].mean()*11)/13 if len(x) == 13 else x.mean()
        )
        
        sma_close_27 = close.rolling(window=27, min_periods=1).apply(
            lambda x: (x[-1]*2 + x[:-1].mean()*25)/27 if len(x) == 27 else x.mean()
        )
        
        # 计算差值
        diff = sma_close_13 - sma_close_27
        
        # 计算差值的SMA
        sma_diff_10 = diff.rolling(window=10, min_periods=1).apply(
            lambda x: (x[-1]*2 + x[:-1].mean()*8)/10 if len(x) == 10 else x.mean()
        )
        
        # 5. 计算最终因子
        factor = 2 * (sma_close_13 - sma_close_27 - sma_diff_10)
        
        # 确保factor保持与输入相同的pivot格式
        factor = factor.reindex(close.index).reindex(close.columns, axis=1)
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha89",
        "creator": "chenxutian",
        "create_date": "2025.10.05",
        "formula": "2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))",
        "description": "alpha191因子89",
        "lookback_days": 35,
        "method": "Alpha89Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''
