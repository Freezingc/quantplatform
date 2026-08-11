from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha35Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["open", "vol"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        open = data.xs('open', level=0)
        vol = data.xs('vol', level=0)

        # 3. 实现因子计算逻辑
        # DELTA(OPEN, 1)
        delta_open = open.diff(1)
        
        # DECAYLINEAR(DELTA(OPEN, 1), 15)
        def decay_linear(series, window):
            weights = np.arange(1, window + 1)
            weights = weights / weights.sum()
            
            def apply_decay(window_values):
                if len(window_values) < window:
                    return np.nan
                return np.sum(window_values * weights)
            
            return series.rolling(window).apply(apply_decay, raw=True)
        
        decay_delta_open = delta_open.apply(lambda x: decay_linear(x, 15))
        
        # RANK(DECAYLINEAR(DELTA(OPEN, 1), 15))
        rank1 = decay_delta_open.rank(axis=1, pct=True)
        
        # (OPEN * 0.65) + (OPEN * 0.35)
        open_weighted = open * 0.65 + open * 0.35
        
        # CORR(vol, ((OPEN * 0.65) + (OPEN * 0.35)), 17)
        def rolling_corr(df1, df2, window):
            result = pd.DataFrame(index=df1.index, columns=df1.columns)
            
            for i in range(window-1, len(df1)):
                window_df1 = df1.iloc[i-window+1:i+1]
                window_df2 = df2.iloc[i-window+1:i+1]
                
                for col in df1.columns:
                    if not window_df1[col].isna().any() and not window_df2[col].isna().any():
                        result.iloc[i, df1.columns.get_loc(col)] = window_df1[col].corr(window_df2[col])
            
            return result
        
        corr_result = rolling_corr(vol, open_weighted, 17)
        
        # DECAYLINEAR(CORR(...), 7)
        decay_corr = corr_result.apply(lambda x: decay_linear(x, 7))
        
        # RANK(DECAYLINEAR(CORR(...), 7))
        rank2 = decay_corr.rank(axis=1, pct=True)
        
        # MIN(RANK1, RANK2)
        min_rank = pd.DataFrame(np.minimum(rank1.values, rank2.values), 
                            index=rank1.index, columns=rank1.columns)
        
        # 最终因子 * -1
        factor = min_rank * -1
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha35",
    "creator": "chenxutian",
    "create_date": "2025.08.26",
    "formula": "(MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)), RANK(DECAYLINEAR(CORR((vol), ((OPEN * 0.65) + (OPEN *0.35)), 17),7))) * -1)",
    "description": "alpha191因子35",
    "lookback_days": 22,
    "method": "Alpha35Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''