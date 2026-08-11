from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha76Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "vol"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)  # 需要vol数据来计算因子

        # 3. 计算每日收益率绝对值除以成交量
        daily_return_abs_div_vol = abs(close / close.shift(1) - 1) / vol
        
        # 4. 计算20日滚动标准差
        std_20 = daily_return_abs_div_vol.rolling(window=20,min_periods=1).std()
        
        # 5. 计算20日滚动均值
        mean_20 = daily_return_abs_div_vol.rolling(window=20,min_periods=1).mean()
        
        # 6. 计算最终因子值
        factor = std_20 / mean_20
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha76",
        "creator": "chenxutian",
        "create_date": "2025.09.07",
        "formula": "STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)",
        "description": "alpha191因子76",
        "lookback_days": 20,
        "method": "Alpha76Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.09"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

