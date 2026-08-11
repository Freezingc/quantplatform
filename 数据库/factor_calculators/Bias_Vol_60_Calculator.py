from .base import FactorCalculator
import numpy as np
import pandas as pd


class Bias_Vol_60_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        # 建议 lookback_days 取 60
        super().__init__(s, lookback_days)
        self.required_factors = ["vol"]  # 使用成交量

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)

        # 2. 提取所需因子
        vol = data.xs("vol", level=0)

        # 3. 成交量均线乖离率：vol / MA(vol, 60)
        vol_ma_60 = vol.rolling(60).mean()
        factor = vol / vol_ma_60

        # 4. 处理无穷大与缺失值
        factor = factor.replace([np.inf, -np.inf], np.nan)

        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "60日成交量均线乖离率",
    "creator": "suyanning",
    "create_date": "2025.01.01",
    "formula": "VOL / SMA(VOL, 60)",
    "description": "成交量相对于60日均量的乖离率",
    "lookback_days": 60,
    "method": "Bias_Vol_60_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    "category": "流动性因子"
}
'''


