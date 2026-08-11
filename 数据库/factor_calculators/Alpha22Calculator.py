from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha22Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)

        # 3. 实现因子计算逻辑
        # 计算MEAN(CLOSE,6)
        mean_close_6 = close.rolling(window=6).mean()
        
        # 计算(CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)
        ratio = (close - mean_close_6) / mean_close_6

        # 计算DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)
        ratio_delay_3 = ratio.shift(3)

        # 计算差值：ratio - ratio_delay_3
        diff = ratio - ratio_delay_3

        # 计算SMEAN(diff,12,1) - 12期简单移动平均
        factor = diff.rolling(window=12, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*11)/12 if len(x) == 12 else x.mean()
        )
        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha22",
    "creator": "chenxutian",
    "create_date": "2025.08.22",
    "formula": "SMEAN(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)",
    "description": "alpha191因子22​",
    "lookback_days": 19,
    "method": "Alpha22Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''