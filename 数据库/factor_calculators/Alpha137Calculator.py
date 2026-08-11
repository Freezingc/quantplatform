from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha137Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low","close","high","open"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        open = data.xs('open', level=0)  # 提取open数据
        
        # 3. 计算滞后值
        delay_close = close.shift(1)  # DELAY(CLOSE,1)
        delay_open = open.shift(1)  # DELAY(OPEN,1)
        delay_low = low.shift(1)  # DELAY(LOW,1)
        
        # 4. 计算中间序列
        A = (high - delay_close).abs()  # ABS(HIGH-DELAY(CLOSE,1))
        B = (low - delay_close).abs()   # ABS(LOW-DELAY(CLOSE,1))
        C = (high - delay_low).abs()    # ABS(HIGH-DELAY(LOW,1))
        D = (delay_close - delay_open).abs()  # ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))
        
        # 5. 条件判断
        cond1 = (A > B) & (A > C)  # 第一个条件
        cond2 = (B > C) & (B > A)  # 第二个条件
        
        # 6. 计算条件值
        val1 = A + B/2 + D/4
        val2 = B + A/2 + D/4
        val3 = C + D/4
        conditional_value = val1.where(cond1, val2.where(cond2, val3))
        
        # 7. 计算最大值 MAX(ABS(HIGH-DELAY(CLOSE,1)), ABS(LOW-DELAY(CLOSE,1)))
        max_AB = A.where(A > B, B)
        
        # 8. 计算分母
        denominator = conditional_value * max_AB
        
        # 9. 计算分子
        numerator = 16 * (close - delay_close + (close - open)/2 + delay_close - delay_open)
        
        # 10. 计算因子
        factor = numerator / denominator
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha137",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE, 1))>ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOS E,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OP EN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1)))",
        "description": "通过比较当前价格变动与前期价格范围的关系来捕捉资金流动的相对强度",
        "lookback_days": 1,
        "method": "Alpha137Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "资金流因子"
    }
'''
