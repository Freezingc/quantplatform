from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha96Calculator(FactorCalculator):
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

        # 实现因子计算逻辑
        # 计算TSMIN(LOW,9) - 过去9天的最低值
        tsmin_low_9 = low.rolling(window=9, min_periods=1).min()
        
        # 计算TSMAX(HIGH,9) - 过去9天的最高值
        tsmax_high_9 = high.rolling(window=9, min_periods=1).max()
        
        # 计算(CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100
        numerator = close - tsmin_low_9
        denominator = tsmax_high_9 - tsmin_low_9
        # 避免除零错误
        denominator = denominator.replace(0, np.nan)
        intermediate = (numerator / denominator) * 100
        
        # 第一次SMA计算，窗口为3，权重为1
        sma1 = intermediate.rolling(window=3, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*2)/3 if len(x) == 3 else x.mean()
        )
        
        # 第二次SMA计算，窗口为3，权重为1
        factor = sma1.rolling(window=3, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*2)/3 if len(x) == 3 else x.mean()
        )
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha96",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)",
        "description": "衡量当前收盘价在最近9天价格区间中的相对位置",
        "lookback_days": 11,
        "method": "Alpha96Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
