from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha3Calculator(FactorCalculator):
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
        # 3. 实现因子计算逻辑
        # 计算前一天的收盘价
        delay_close = close.shift(1, axis=0)
        
        # 条件1: 当天收盘价等于前一天收盘价
        cond_equal = (close == delay_close)
        
        # 条件2: 当天收盘价大于前一天收盘价
        cond_greater = (close > delay_close)

        # 计算中间值: CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1))
        min_low_delay_close = np.minimum(low, delay_close)
        max_high_delay_close = np.maximum(high, delay_close)
        
        temp = np.where(cond_greater, min_low_delay_close, max_high_delay_close)
        # 计算每日的因子值: CLOSE=DELAY(CLOSE,1)?0:CLOSE-temp
        daily_factor = np.where(cond_equal, 0, close - temp)
        daily_factor = pd.DataFrame(daily_factor, 
                               index=close.index, 
                               columns=close.columns)
        # 计算6日累计和
        factor = daily_factor.rolling(window=6, axis=0).sum()
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha3",
    "creator": "chenxutian",
    "create_date": "2025.08.20",
    "formula": "SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)",
    "description": "alpha191因子3",
    "lookback_days": 6,
    "method": "Alpha3Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2025.08.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''