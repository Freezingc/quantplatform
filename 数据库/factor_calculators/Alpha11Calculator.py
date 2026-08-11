from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha11Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "low", "high", "vol"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        vol = data.xs('vol', level=0)  # 需要获取vol数据

        # 3. 实现因子计算逻辑
        # 计算分子部分：(CLOSE-LOW)-(HIGH-CLOSE) = 2*CLOSE - LOW - HIGH
        numerator = 2 * close - low - high
        
        # 计算分母部分：(HIGH-LOW)
        denominator = high - low

        # 避免除零错误，将分母为0的地方替换为NaN
        denominator = denominator.replace(0, float('nan'))

        # 计算单日因子值：((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*vol
        daily_factor = (numerator / denominator) * vol

        # 计算6日滚动求和
        factor = daily_factor.rolling(window=6).sum()

        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha11",
    "creator": "chenxutian",
    "create_date": "2025.08.22",
    "formula": "SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*vol,6)",
    "description": "alpha191因子11",
    "lookback_days": 5,
    "method": "Alpha11Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''