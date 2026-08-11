from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha38Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        high = data.xs('high', level=0)

        # 3. 实现因子计算逻辑
        # 计算20日最高价的移动平均
        high_20ma = high.rolling(window=20).mean()
        
        # 计算HIGH的2日差分
        high_delta2 = high.diff(periods=2)
        
        # 构建条件：当HIGH大于其20日移动平均时
        condition = high > high_20ma
        
        # 根据条件计算结果：条件为真时取-1 * DELTA(HIGH, 2)，否则为0
        factor = pd.DataFrame(index=high.index, columns=high.columns)
        factor = condition * (-1 * high_delta2)
        factor = factor.fillna(0)  # 处理NaN值
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha38",
    "creator": "chenxutian",
    "create_date": "2025.08.26",
    "formula": "(((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0)",
    "description": "alpha191因子38",
    "lookback_days": 19,
    "method": "Alpha38Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''