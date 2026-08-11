from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha59Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "low", "high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        
        # 3. 计算DELAY(CLOSE,1)
        delay_close = close.shift(1)
        
        # 4. 计算条件部分
        # 条件1: CLOSE == DELAY(CLOSE,1)
        condition_equal = (close == delay_close)
        
        # 条件2: CLOSE > DELAY(CLOSE,1)
        condition_greater = (close > delay_close)
        
        # 条件3: CLOSE < DELAY(CLOSE,1)
        condition_less = (close < delay_close)
        
        # 5. 计算MIN(LOW, DELAY(CLOSE,1))和MAX(HIGH, DELAY(CLOSE,1))
        min_low_delay = pd.DataFrame(np.minimum(low.values, delay_close.values), 
                                    index=low.index, columns=low.columns)
        max_high_delay = pd.DataFrame(np.maximum(high.values, delay_close.values), 
                                    index=high.index, columns=high.columns)
        
        # 6. 计算核心部分
        core_part = pd.DataFrame(np.zeros_like(close), index=close.index, columns=close.columns)
        
        # 当CLOSE > DELAY(CLOSE,1)时
        core_part[condition_greater] = close - min_low_delay
        # 当CLOSE < DELAY(CLOSE,1)时
        core_part[condition_less] = close - max_high_delay
        # 当CLOSE == DELAY(CLOSE,1)时
        core_part[condition_equal] = 0
        
        # 7. 计算20天SUM
        factor = core_part.rolling(20).sum()
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha59",
        "creator": "chenxutian",
        "create_date": "2025.09.01",
        "formula": "SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),20)",
        "description": "alpha191因子59",
        "lookback_days": 20,
        "method": "Alpha59Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

