from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha146Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        
        # 实现因子计算逻辑
        # 计算日收益率: (CLOSE - DELAY(CLOSE,1)) / DELAY(CLOSE,1)
        returns = (close - close.shift(1)) / close.shift(1)

        # 计算加权SMA
        sma_61_2 = returns.rolling(window=61, min_periods=2).apply(
            lambda x: (x[-1] + x[:-1].mean()*59)/61 if len(x) == 61 else x.mean()
        )
        # 计算中间变量: returns - sma_61_2
        diff = returns - sma_61_2
        
        # 计算MEAN(diff, 20) - 简单的20日移动平均
        mean_diff_20 = diff.rolling(window=20, min_periods=1).mean()
        
        # 计算分子的第一部分: MEAN(diff, 20) * diff
        numerator_part1 = mean_diff_20 * diff
        
        # 计算分母部分: SMA((diff - (diff - sma_61_2))^2, 60)
        # 注意: diff - (diff - sma_61_2) = sma_61_2
        # 但根据公式: ((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA(...)))
        # 这实际上等于 SMA(...)
        variance_term = sma_61_2
        
        # 计算SMA(variance_term^2, 60) - 使用简单的60日移动平均
        sma_variance_60 = (variance_term ** 2).rolling(window=60, min_periods=1).mean()
        
        # 完整的因子计算: numerator_part1 / sma_variance_60
        factor = numerator_part1 / sma_variance_60
        
        # 处理可能的无穷大值和NaN值
        factor = factor.replace([np.inf, -np.inf], np.nan)
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha146",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2),20)*(( CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2))/SMA(((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE, 1))/DELAY(CLOSE,1),61,2)))^2,60)",
        "description": "通过计算收益率相对于其加权移动平均的偏离程度，捕捉价格在短期动量与长期趋势之间的反转信号，当短期收益率显著偏离长期趋势时产生交易信号",
        "lookback_days": 121,
        "method": "Alpha146Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
