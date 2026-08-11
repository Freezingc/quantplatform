from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha27Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)

        # 3. 实现因子计算逻辑
        # 计算3日收益率和6日收益率
        ret3 = (close - close.shift(3)) / close.shift(3) * 100
        ret6 = (close - close.shift(6)) / close.shift(6) * 100
        
        # 计算组合收益率
        combined_ret = ret3 + ret6
        
        # 计算WMA(12) - 使用滚动窗口和权重计算
        window = 12
        weights = np.arange(1, window + 1)
        
        # 使用滚动apply计算WMA
        def wma_func(x):
            if len(x) < window:
                return np.nan
            return np.sum(x * weights) / weights.sum()
        
        factor = combined_ret.rolling(window=window).apply(wma_func, raw=True)
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha27",
    "creator": "chenxutian",
    "create_date": "2025.08.25",
    "formula": "WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)",
    "description": "alpha191因子27​",
    "lookback_days": 17,
    "method": "Alpha27Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''