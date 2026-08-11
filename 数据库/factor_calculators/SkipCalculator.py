from .base import FactorCalculator
import numpy as np
import pandas as pd

class SkipCalculator(FactorCalculator):
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
        factor = open_price / pre_close
        
        # 5. 处理可能的无穷大值和缺失值
        factor = factor.replace([np.inf, -np.inf], np.nan)
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "隔夜跳空",
    "creator": "suyanning",
    "create_date": "2025.01.01",
    "formula": "open / pre_close",
    "description": "隔夜跳空因子，计算当日开盘价相对于前一日收盘价的比值",
    "lookback_days": 1,
    "method": "SkipCalculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": ""  # 可以为空，默认为当日日期，如果为空，请填写空字符串"",
    "category": "动量反转因子"
}
'''

