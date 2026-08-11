from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha50Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low", "high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        high = data.xs('high', level=0)

        # 计算延迟一天的数据
        delay_high = high.shift(1)
        delay_low = low.shift(1)
        
        # 计算条件判断部分
        cond1 = (high + low) <= (delay_high + delay_low)  # 条件1
        cond2 = (high + low) >= (delay_high + delay_low)  # 条件2
        
        # 计算绝对值部分并保持DataFrame格式
        abs_high = (high - delay_high).abs()
        abs_low = (low - delay_low).abs()
        max_abs = abs_high.where(abs_high > abs_low, abs_low)  # 保持DataFrame格式
        
        # 计算分子部分
        numerator_part1 = max_abs.where(~cond1, 0)  # 条件1不满足时取max_abs，否则0
        numerator_part2 = max_abs.where(~cond2, 0)  # 条件2不满足时取max_abs，否则0
        
        # 计算滚动求和
        sum_num_part1 = numerator_part1.rolling(window=12, min_periods=1).sum()
        sum_num_part2 = numerator_part2.rolling(window=12, min_periods=1).sum()
        
        # 计算分母部分
        denominator_part1 = sum_num_part1 + sum_num_part2
        denominator_part2 = sum_num_part2 + sum_num_part1
        
        # 避免除以0的情况
        denominator_part1 = denominator_part1.replace(0, np.nan)
        denominator_part2 = denominator_part2.replace(0, np.nan)
        
        # 计算最终因子
        factor = (sum_num_part1 / denominator_part1) - (sum_num_part2 / denominator_part2)
        
        return factor
'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha50",
    "creator": "chenxutian",
    "create_date": "2025.08.27",
    "formula": "SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))-SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))",
    "description": "alpha191因子50",
    "lookback_days": 12,
    "method": "Alpha50Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
