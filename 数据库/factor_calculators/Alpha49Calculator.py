from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha49Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low", "high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        high = data.xs('high', level=0)

        # 3. 实现因子计算逻辑
        # 计算前一日的高低价
        delay_high = high.shift(1)
        delay_low = low.shift(1)
        
        # 计算价格变化量
        high_diff = abs(high - delay_high)
        low_diff = abs(low - delay_low)
        price_change = pd.DataFrame(np.maximum(high_diff.values, low_diff.values), 
                                index=high_diff.index, columns=high_diff.columns)
        
        # 条件1: (HIGH+LOW) >= (DELAY(HIGH,1)+DELAY(LOW,1))
        condition1 = (high + low) >= (delay_high + delay_low)
        
        # 条件2: (HIGH+LOW) <= (DELAY(HIGH,1)+DELAY(LOW,1))
        condition2 = (high + low) <= (delay_high + delay_low)
        
        # 计算分子部分：SUM(condition1为False时的price_change, 12)
        numerator_part = pd.DataFrame(np.where(condition1, 0, price_change), 
                                    index=price_change.index, columns=price_change.columns)
        numerator = numerator_part.rolling(window=12).sum()
        
        # 计算分母的第二部分：SUM(condition2为False时的price_change, 12)
        denominator_part2 = pd.DataFrame(np.where(condition2, 0, price_change), 
                                    index=price_change.index, columns=price_change.columns)
        denominator_part2_sum = denominator_part2.rolling(window=12).sum()
        
        # 计算完整分母
        denominator = numerator + denominator_part2_sum
        
        # 计算最终因子值
        factor = numerator / denominator
        
        # 处理可能的除零错误
        factor = factor.replace([np.inf, -np.inf], np.nan)
        
        return factor
'''
# 添加因子信息
{
    "factor_name": "Alpha49",
    "creator": "chenxutian",
    "create_date": "2025.08.27",
    "formula": "SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))",
    "description": "alpha191因子49",
    "lookback_days": 12,
    "method": "Alpha49Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
