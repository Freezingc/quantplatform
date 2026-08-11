from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha5Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["high", "vol"]  # 指定该因子需要的原始因子


    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        high = data.xs('high', level=0)
        vol = data.xs('vol', level=0)

        # 3. 实现因子计算逻辑
        # 计算TSRANK(vol, 5)
        def tsrank(series, window):
            return series.rolling(window).apply(lambda x: x.rank().iloc[-1]/window, raw=False)
        
        vol_rank = vol.apply(lambda x: tsrank(x, 5), axis=0)
        high_rank = high.apply(lambda x: tsrank(x, 5), axis=0)

        # 计算CORR(vol_rank, high_rank, 5)
        def rolling_corr(df1, df2, window):
            corr_values = pd.DataFrame(index=df1.index, columns=df1.columns)
            for col in range(window-1, len(df1.columns)):
                start = col - window + 1
                end = col + 1
                slice1 = df1.iloc[:, start:end]
                slice2 = df2.iloc[:, start:end]
                corr = slice1.T.corrwith(slice2.T).values
                corr_values.iloc[:, col] = corr
            return corr_values.T
        
        corr_values = rolling_corr(vol_rank.T, high_rank.T, 5)

        # 计算TSMAX(corr_values, 3)
        tsmax_values = corr_values.rolling(3, axis=0).max()
        
        # 乘以-1得到最终因子
        factor = -1 * tsmax_values
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha5",
    "creator": "chenxutian",
    "create_date": "2025.08.21",
    "formula": "(-1 * TSMAX(CORR(TSRANK(vol, 5), TSRANK(HIGH, 5), 5), 3))",
    "description": "alpha191因子5",
    "lookback_days": 10,
    "method": "Alpha5Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
