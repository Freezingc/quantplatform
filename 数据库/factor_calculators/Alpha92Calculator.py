from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha92Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close","vol","amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)
        amount = data.xs('amount', level=0)
        
        # 3. 计算VWAP (成交金额/成交量)
        vwap = amount * 10 / vol

        # 实现因子计算逻辑
        # 步骤1: 计算 DELTA(((CLOSE * 0.35) + (VWAP * 0.65)), 2)
        weighted_price = (close * 0.35) + (vwap * 0.65)
        delta_weighted = weighted_price.diff(2)
        
        # 步骤2: 计算 DECAYLINEAR(DELTA(...), 3)
        def decay_linear(series, window):
            weights = np.arange(1, window + 1)
            weights = weights / weights.sum()
            return series.rolling(window).apply(lambda x: (x * weights).sum(), raw=True)
        
        decay_delta = decay_linear(delta_weighted, 3)
        
        # 步骤3: 计算 RANK(DECAYLINEAR(...))
        def rank(series):
            return series.rank(axis=1, pct=True)
        
        rank_decay = rank(decay_delta)
        
        # 步骤4: 计算 MEAN(vol,180)
        mean_vol_180 = vol.rolling(180).mean()
        
        # 步骤5: 计算 CORR(MEAN(vol,180), CLOSE, 13)
        def rolling_corr(x, y, window):
            return x.rolling(window).corr(y)
        
        corr_vol_close = rolling_corr(mean_vol_180, close, 13)
        
        # 步骤6: 计算 ABS(CORR(...))
        abs_corr = corr_vol_close.abs()
        
        # 步骤7: 计算 DECAYLINEAR(ABS(CORR(...)), 5)
        decay_abs_corr = decay_linear(abs_corr, 5)
        
        # 步骤8: 计算 TSRANK(DECAYLINEAR(...), 15)
        def tsrank(series, window):
            return series.rolling(window).apply(lambda x: (x.rank().iloc[-1] - 1) / (len(x) - 1) if len(x) == window else np.nan, raw=False)
        
        tsrank_decay = decay_abs_corr.rolling(15).apply(
            lambda x: (pd.Series(x).rank().iloc[-1] - 1) / (len(x) - 1) if len(x) == 15 else np.nan,
            raw=False
        )
        
        # 步骤9: 计算 MAX(RANK(...), TSRANK(...))
        max_value = np.maximum(rank_decay, tsrank_decay)
        
        # 步骤10: 乘以 -1
        factor = max_value * -1
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha92",
        "creator": "chenxutian",
        "create_date": "2025.10.15",
        "formula": "(MAX(RANK(DECAYLINEAR(DELTA(((CLOSE * 0.35) + (VWAP *0.65)), 2), 3)), TSRANK(DECAYLINEAR(ABS(CORR((MEAN(vol,180)), CLOSE, 13)), 5), 15)) * -1)",
        "description": "alpha191因子92",
        "lookback_days": 43,
        "method": "Alpha92Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''
