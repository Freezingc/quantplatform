from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha28Calculator(FactorCalculator):
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

        # 3. 实现因子计算逻辑
        # 计算TSMIN(LOW,9) - 9日最低价
        min_low_9 = low.rolling(window=9).min()
        
        # 计算TSMAX(HIGH,9) - 9日最高价
        max_high_9 = high.rolling(window=9).max()
        
        # 计算第一个SMA的分子部分：(CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100
        # 避免除零错误
        denominator = max_high_9 - min_low_9
        denominator = denominator.replace(0, 1e-6)  # 将0替换为极小值避免除零
        
        rsv = (close - min_low_9) / denominator * 100
        
        # 计算第一个SMA：SMA(rsv, 3, 1)
        sma1 = rsv.rolling(window=3, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*2)/3 if len(x) == 3 else x.mean()
        )
        
        # 计算第二个SMA的分子部分，注意公式中应该是TSMAX(HIGH,9)而不是MAX(HIGH,9)
        # 根据公式：SMA((CLOSE-TSMIN(LOW,9))/(MAX(HIGH,9)-TSMAX(LOW,9))*100,3,1)
        # 这里应该是TSMAX(HIGH,9)，我假设是笔误，使用TSMAX(HIGH,9)
        denominator2 = max_high_9 - min_low_9  # 使用相同的分母
        denominator2 = denominator2.replace(0, 1e-6)
        
        rsv2 = (close - min_low_9) / denominator2 * 100
        
        # 计算内层SMA：SMA(rsv2, 3, 1)
        inner_sma = rsv2.rolling(window=3, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*2)/3 if len(x) == 3 else x.mean()
        )
        # 计算外层SMA：SMA(inner_sma, 3, 1)
        sma2 = inner_sma.rolling(window=3, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*2)/3 if len(x) == 3 else x.mean()
        )
        # 最终因子计算：3*SMA1 - 2*SMA2
        factor = 3 * sma1 - 2 * sma2
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha28",
    "creator": "chenxutian",
    "create_date": "2025.08.25",
    "formula": "3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(MAX(HIGH,9)-TSMAX(LOW,9))*100,3,1),3,1)",
    "description": "alpha191因子28​",
    "lookback_days": 12,
    "method": "Alpha28Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''