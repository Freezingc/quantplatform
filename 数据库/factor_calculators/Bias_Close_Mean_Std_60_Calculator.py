from .base import FactorCalculator
import numpy as np
import pandas as pd


class Bias_Close_Mean_Std_60_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        # 建议 lookback_days 取 60
        super().__init__(s, lookback_days)
        self.required_factors = ["close"]  # 使用收盘价

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)

        # 2. 提取所需因子
        close = data.xs("close", level=0)

        # 3. 计算 60 日均值和标准差
        ma_60 = close.rolling(60).mean()
        std_60 = close.rolling(60).std()

        # 4. 价格相对“均值 + 3倍标准差”的乖离：close / (MA + 3*STD)
        denom = ma_60 + 3 * std_60
        factor = close / denom

        # 5. 处理无穷大与缺失值
        factor = factor.replace([np.inf, -np.inf], np.nan)

        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "60日价格均值加3倍标准差乖离率",
    "creator": "suyanning",
    "create_date": "2025.01.01",
    "formula": "CLOSE / (SMA(CLOSE, 60) + 3*STD(CLOSE, 60))",
    "description": "价格相对60日均值加3倍标准差的乖离程度",
    "lookback_days": 60,
    "method": "Bias_Close_Mean_Std_60_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    "category": "动量反转因子"
}
'''


