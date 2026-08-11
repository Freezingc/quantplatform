from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha15Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close","open"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        open = data.xs('open', level=0)  # 需要获取开盘价数据

        # 3. 实现因子计算逻辑
        # DELAY(CLOSE,1) - 将收盘价滞后一期
        close_lag1 = close.shift(1)
        
        # OPEN / DELAY(CLOSE,1) - 1
        factor = open / close_lag1 - 1
        
        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha15",
    "creator": "chenxutian",
    "create_date": "2025.08.22",
    "formula": "OPEN/DELAY(CLOSE,1)-1",
    "description": "alpha191因子15。今日开盘价相对于昨日收盘价的涨跌幅​​",
    "lookback_days": 1,
    "method": "Alpha15Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''