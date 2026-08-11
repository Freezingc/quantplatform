from .base import FactorCalculator
import numpy as np
import pandas as pd

class vol_down_speed_turnover_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["pct_chg", "turnover_rate","vol"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        pct_chg = data.xs('pct_chg', level=0)
        turnover = data.xs('turnover_rate', level=0)
        vol = data.xs('vol', level=0)

        vol_down = vol < vol.rolling(3).mean()
        speed_turnover = (turnover.pct_change()).rolling(20).mean()
        factor = (vol_down * speed_turnover).rolling(20).mean()

        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "成交量下降加速换手",
    "creator": "suyanning",
    "create_date": "2025.09.01",
    "formula": "vol_down = vol < SMA(vol,3); speed_turnover = SMA(turnover.pct_change(),20); SMA(vol_down*speed_turnover,20)",
    "description": "成交量下降加速换手",
    "lookback_days": 20,
    "method": "vol_down_speed_turnover_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": ""  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
