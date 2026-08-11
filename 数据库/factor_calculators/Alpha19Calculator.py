from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha19Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)

        # 3. 实现因子计算逻辑
        # 计算DELAY(CLOSE,5) - 5天前的收盘价
        delay_close_5 = close.shift(5)
        
        # 初始化因子值为0
        factor = pd.DataFrame(0, index=close.index, columns=close.columns)
        
        # 应用条件逻辑
        # CLOSE < DELAY(CLOSE,5)
        condition1 = close < delay_close_5
        # CLOSE = DELAY(CLOSE,5)
        condition2 = close == delay_close_5
        
        # 当CLOSE < DELAY(CLOSE,5)时，使用(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5)
        factor[condition1] = (close - delay_close_5) / delay_close_5
        
        # 当CLOSE = DELAY(CLOSE,5)时，保持为0
        # 这里不需要额外操作，因为初始值已经是0
        
        # 当CLOSE > DELAY(CLOSE,5)时，使用(CLOSE-DELAY(CLOSE,5))/CLOSE
        condition3 = close > delay_close_5
        factor[condition3] = (close - delay_close_5) / close
        
        # 处理NaN值（由于shift操作产生的前5天数据）
        factor = factor.fillna(0)
        
        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha19",
    "creator": "chenxutian",
    "create_date": "2025.08.22",
    "formula": "(CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))",
    "description": "alpha191因子19​",
    "lookback_days": 5,
    "method": "Alpha19Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''