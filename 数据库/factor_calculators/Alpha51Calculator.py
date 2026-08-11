from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha51Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low", "high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        high = data.xs('high', level=0)

        # 计算延迟值
        delay_high = high.shift(1)
        delay_low = low.shift(1)
        
        # 计算条件1：(HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))
        condition1 = (high + low) <= (delay_high + delay_low)
        
        # 计算条件2：(HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))
        condition2 = (high + low) >= (delay_high + delay_low)
        
        # 计算MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))
        max_abs = pd.DataFrame(np.maximum(np.abs(high - delay_high), np.abs(low - delay_low)), 
                            index=high.index, columns=high.columns)
        
        # 计算分子部分：SUM(condition1为真时取0，否则取max_abs, 12)
        numerator = max_abs.where(~condition1, 0).rolling(window=12).sum()
        
        # 计算分母的第二部分：SUM(condition2为真时取0，否则取max_abs, 12)
        denominator_part2 = max_abs.where(~condition2, 0).rolling(window=12).sum()
        
        # 计算完整分母
        denominator = numerator + denominator_part2
        
        # 计算最终因子值
        factor = numerator / denominator
        
        return factor
'''
# 添加因子信息
{
    "factor_name": "Alpha51",
    "creator": "chenxutian",
    "create_date": "2025.08.27",
    "formula": "SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))",
    "description": "alpha191因子51",
    "lookback_days": 12,
    "method": "Alpha51Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
