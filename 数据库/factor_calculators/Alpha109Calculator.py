from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha109Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
            # 1. 获取数据
            data = self.fetch_data(start_time, end_time)
            
            # 2. 提取所需因子
            low = data.xs('low', level=0)
            high = data.xs('high', level=0)
            
            # 3. 实现因子计算逻辑
            # 计算 HIGH - LOW
            hl_diff = high - low
            
            
            # 计算 SMA(HIGH-LOW, 10, 2)
            sma_hl = hl_diff.rolling(window=10, min_periods=2).apply(
                lambda x: (x[-1] + x[:-1].mean()*8)/10 if len(x) == 10 else x.mean()
            )
            # 计算 SMA(SMA(HIGH-LOW,10,2), 10, 2)
            sma_sma_hl = sma_hl.rolling(window=10, min_periods=2).apply(
                lambda x: (x[-1] + x[:-1].mean()*8)/10 if len(x) == 10 else x.mean()
            )
            # 计算最终因子
            factor = sma_hl / sma_sma_hl
            
            return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha109",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)",
        "description": "衡量近期股价波动率相对于其平滑趋势的相对强度指标，用于识别波动率的加速或减速状态",
        "lookback_days": 18,
        "method": "Alpha109Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "波动率因子"
    }
'''
