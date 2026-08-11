from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha72Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "low", "high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        
        # 3. 计算TSMAX(HIGH,6)
        tsmax_high_6 = high.rolling(6).max()
        
        # 4. 计算TSMIN(LOW,6)
        tsmin_low_6 = low.rolling(6).min()
        
        # 5. 计算分子部分 (TSMAX(HIGH,6)-CLOSE)
        numerator = tsmax_high_6 - close
        
        # 6. 计算分母部分 (TSMAX(HIGH,6)-TSMIN(LOW,6))
        denominator = tsmax_high_6 - tsmin_low_6
        
        # 7. 计算比率并乘以100
        ratio = (numerator / denominator) * 100
        
        # 8. 计算SMA(ratio, 15, 1)
        # SMA(X,N,M) = (X*M + Y*(N-M))/N，其中Y是前一天的SMA值
        sma_ratio = ratio.rolling(window=15, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*14)/15 if len(x) == 15 else x.mean()
        )
        factor = sma_ratio
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha72",
        "creator": "chenxutian",
        "create_date": "2025.09.07",
        "formula": "SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)",
        "description": "alpha191因子72",
        "lookback_days": 19,
        "method": "Alpha72Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.09"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

