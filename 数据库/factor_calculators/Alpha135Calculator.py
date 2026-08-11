from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha135Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        close = data.xs('close', level=0)
        
        # 3. 计算因子: SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)
        # 3.1 计算 DELAY(CLOSE, 20) - 20天前的收盘价
        close_lag20 = close.shift(20)
        
        # 3.2 计算 CLOSE / DELAY(CLOSE, 20) - 当前收盘价与20天前收盘价的比值
        close_ratio = close / close_lag20
        
        # 3.3 计算 DELAY(CLOSE/DELAY(CLOSE,20),1) - 对比值做1天的滞后
        close_ratio_lag1 = close_ratio.shift(1)
        
        # 3.4 计算 SMA(..., 20, 1) - 简单移动平均，权重为1
        # 这里使用rolling窗口计算，min_periods=1确保即使数据不足20天也能有输出
        factor = close_ratio_lag1.rolling(window=20, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*19)/20 if len(x) == 20 else x.mean()
        )
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha135",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)",
        "description": "当前收盘价相对于20天前收盘价的变化率的滞后一期（前一天）的20日简单移动平均",
        "lookback_days": 40,
        "method": "Alpha135Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
