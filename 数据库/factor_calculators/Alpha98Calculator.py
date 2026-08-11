from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha98Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)

        # 3. 实现因子计算逻辑
        # 计算 SUM(CLOSE, 100) / 100
        sum_close_100 = close.rolling(window=100, min_periods=1).sum() / 100
        
        # 计算 DELTA((SUM(CLOSE, 100) / 100), 100)
        delta_sum = sum_close_100 - sum_close_100.shift(100)
        
        # 计算 DELAY(CLOSE, 100)
        delay_close_100 = close.shift(100)
        
        # 计算条件表达式：DELTA(...)/DELAY(...) <= 0.05
        condition = (delta_sum / delay_close_100) <= 0.05
        
        # 计算条件为真时的值：-1 * (CLOSE - TSMIN(CLOSE, 100))
        true_value = -1 * (close - close.rolling(window=100, min_periods=1).min())
        
        # 计算条件为假时的值：-1 * DELTA(CLOSE, 3)
        false_value = -1 * (close - close.shift(3))
        
        # 应用条件判断
        factor = np.where(condition, true_value, false_value)
        
        # 确保factor也是pivot形式（与输入数据格式一致）
        factor = pd.DataFrame(factor, index=close.index, columns=close.columns)
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha98",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "((((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) < 0.05) || ((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) == 0.05)) ? (-1 * (CLOSE - TSMIN(CLOSE, 100))) : (-1 * DELTA(CLOSE, 3)))",
        "description": "判断过去100日的价格动量是否较弱",
        "lookback_days": 99,
        "method": "Alpha98Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
