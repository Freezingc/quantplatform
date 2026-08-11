from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha43Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "vol"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)

        # 3. 实现因子计算逻辑
        delay_close = close.shift(1)
        
        # 使用np.where进行条件判断
        signed_vol = np.where(close > delay_close, vol,
                                np.where(close < delay_close, -vol, 0))
        
        # 转换为DataFrame并保持相同的索引和列名
        signed_vol_df = pd.DataFrame(signed_vol, 
                                      index=vol.index, 
                                      columns=vol.columns)
        
        # 计算6日滚动求和
        factor = signed_vol_df.rolling(window=6).sum()
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha43",
    "creator": "chenxutian",
    "create_date": "2025.08.26",
    "formula": "SUM((CLOSE>DELAY(CLOSE,1)?vol:(CLOSE<DELAY(CLOSE,1)?-vol:0)),6)",
    "description": "alpha191因子43",
    "lookback_days": 5,
    "method": "Alpha43Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''