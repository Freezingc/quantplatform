from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha160Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        
        # 3. 实现因子计算逻辑
        # 计算每日的收盘价标准差（20日）
        close_std_20 = close.rolling(window=20, min_periods=1).std()
        
        # 计算前一日收盘价
        close_lag1 = close.shift(1)
        
        # 条件判断：当收盘价 <= 前一日收盘价时，取标准差，否则取0
        condition = (close <= close_lag1)
        condition_result = close_std_20.where(condition, 0)
        
        # 计算SMA(20,1)
        
        # 对每个股票分别计算SMA(20,1)
        factor = condition_result.rolling(window=20, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*19)/20 if len(x) == 20 else x.mean()
        )
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha160",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)",
        "description": "在股价下跌时，收盘价20日波动率（标准差）的20日特殊加权移动平均",
        "lookback_days": 20,
        "method": "Alpha160Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "波动率因子"
    }
'''
