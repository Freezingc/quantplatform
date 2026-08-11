from .base import FactorCalculator
import numpy as np
import pandas as pd


class Bias_Close_Mean_Std_Low_Vol_Std_60_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        # 这里有 60 日与 10 日窗口，建议 lookback_days 取 60
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "vol"]  # 使用收盘价和成交量

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)

        # 2. 提取所需因子
        close = data.xs("close", level=0)
        vol = data.xs("vol", level=0)

        # 3. 计算 60 日价格均值和标准差
        ma_60 = close.rolling(60).mean()
        std_60 = close.rolling(60).std()
        denom = ma_60 + 3 * std_60
        base_bias = close / denom

        # 4. 计算 10 日成交量标准差
        vol_std_10 = vol.rolling(10).std()

        # 5. 组合因子：价格乖离 * 短期成交量波动
        factor = base_bias * vol_std_10

        # 6. 处理无穷大与缺失值
        factor = factor.replace([np.inf, -np.inf], np.nan)

        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "价格均值标准差乖离乘短期成交量波动",
    "creator": "suyanning",
    "create_date": "2025.01.01",
    "formula": "CLOSE / (SMA(CLOSE, 60) + 3*STD(CLOSE, 60)) * STD(VOL, 10)",
    "description": "价格相对60日均值+3倍标准差的乖离，叠加10日成交量波动",
    "lookback_days": 60,
    "method": "Bias_Close_Mean_Std_Low_Vol_Std_60_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    "category": "动量反转因子"
}
'''


