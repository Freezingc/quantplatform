from .base import FactorCalculator
import numpy as np
import pandas as pd

class SkipMean10Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["open", "close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        open_price = data.xs('open', level=0)
        close = data.xs('close', level=0)
        
        # 3. 计算前一日收盘价
        pre_close = close.shift(1)
        
        # 4. 计算隔夜跳空 = open / pre_close
        skip = open_price / pre_close
        
        # 5. 计算10日滚动均值
        skip_mean_10 = skip.rolling(10).mean()
        
        # 6. 处理可能的无穷大值和缺失值
        skip_mean_10 = skip_mean_10.replace([np.inf, -np.inf], np.nan)
        
        return skip_mean_10

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "隔夜跳空10日均值",
    "creator": "suyanning",
    "create_date": "2025.01.01",
    "formula": "SMA(open / pre_close, 10)",
    "description": "隔夜跳空10日滚动均值",
    "lookback_days": 10,
    "method": "SkipMean10Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": ""  # 可以为空，默认为当日日期，如果为空，请填写空字符串"",
    "category": "动量反转因子"
}
'''

