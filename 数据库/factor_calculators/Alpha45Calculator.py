from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha45Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["amount", "vol", "open", "close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        open = data.xs('open', level=0)
        vol = data.xs('vol', level=0)
        amount = data.xs('amount', level=0)

        vwap = amount * 10 / vol

        # 实现因子计算逻辑
        # 计算第一部分：(((CLOSE * 0.6) + (OPEN * 0.4)))
        part1 = (close * 0.6) + (open * 0.4)
        
        # 计算DELTA：与1天前的差值
        delta_part = part1.diff(periods=1)
        # 计算第一个RANK：对DELTA进行横截面排名
        def rank_dataframe(df):
            return df.rank(axis=1, pct=True)
        
        rank1 = rank_dataframe(delta_part)


        # 计算第二部分：CORR(VWAP, MEAN(vol,150), 15)
        # 先计算MEAN(vol,150)
        mean_vol_150 = vol.rolling(window=150,min_periods=1).mean()

        # 计算15天的相关系数
        def rolling_corr(x, y, window):
            corr_df = pd.DataFrame(index=x.index, columns=x.columns)
            for col in x.columns:
                if col in y.columns:
                    corr_series = x[col].rolling(window=window,min_periods=1).corr(y[col])
                    corr_df[col] = corr_series
            return corr_df
        
        corr_part = rolling_corr(vwap, mean_vol_150, 15)
        
        # 计算第二个RANK：对相关系数进行横截面排名
        rank2 = rank_dataframe(corr_part)
        
        # 计算最终因子：两个RANK的乘积
        factor = rank1 * rank2
        '''
        for index, value in open['000004.SZ'].items():
            print(f"{index}: {value}")
        '''
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha45",
    "creator": "chenxutian",
    "create_date": "2025.08.27",
    "formula": "(RANK(DELTA((((CLOSE * 0.6) + (OPEN *0.4))), 1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15)))",
    "description": "alpha191因子45",
    "lookback_days": 163,
    "method": "Alpha45Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''