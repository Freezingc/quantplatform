from .base import FactorCalculator
import numpy as np
import pandas as pd

class price_up_speed_turnover_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["pct_chg", "turnover_rate","close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        pct_chg = data.xs('pct_chg', level=0)
        turnover = data.xs('turnover_rate', level=0)
        close = data.xs('close', level=0)

        price_up = close > close.rolling(3).mean()
        speed_turnover = (turnover.pct_change()).rolling(20).mean()
        factor = (price_up * speed_turnover).rolling(20).mean()

        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "价格上涨加速换手",
    "creator": "suyanning",
    "create_date": "2025.09.01",
    "formula": "price_up = close > SMA(close,3); speed_turnover = SMA(turnover.pct_change(),20); SMA(price_up*speed_turnover,20)",
    "description": "价格上涨加速换手",
    "lookback_days": 20,
    "method": "price_up_speed_turnover_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": ""  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
