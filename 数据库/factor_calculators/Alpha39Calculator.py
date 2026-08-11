from .base import FactorCalculator
import numpy as np
import pandas as pd
from scipy.stats import rankdata

class Alpha39Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "open", "vol", "amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        open = data.xs('open', level=0)
        vol = data.xs('vol', level=0)
        amount = data.xs('amount', level=0)

        # 计算VWAP (成交量加权平均价)
        vwap = amount * 10 / vol

        # 辅助函数：DECAYLINEAR（加权移动平均，权重线性递减）
        def decay_linear(series, window):
            weights = np.arange(window, 0, -1)  # 线性递减权重
            weights = weights / weights.sum()    # 归一化
            return series.rolling(window=window).apply(lambda x: np.dot(x, weights), raw=True)
        
        
        # 计算各个部分
        # 第一部分：RANK(DECAYLINEAR(DELTA(CLOSE, 2), 8))
        delta_close = close.diff(2)  # DELTA(CLOSE, 2)
        decay_delta = decay_linear(delta_close, 8)  # DECAYLINEAR(DELTA(CLOSE, 2), 8)
        rank1 = decay_delta.rank(axis=1, pct=True) # RANK部分
        
        # 第二部分：RANK(DECAYLINEAR(CORR((VWAP*0.3 + OPEN*0.7), SUM(MEAN(vol,180), 37), 14), 12))
        # 计算 (VWAP * 0.3 + OPEN * 0.7)
        vwap_open_comb = vwap * 0.3 + open * 0.7

        # 计算 SUM(MEAN(vol, 180), 37)
        mean_vol = vol.rolling(window=180).mean()  # MEAN(vol, 180)

        sum_mean_vol = mean_vol.rolling(window=37).sum()  # SUM(MEAN(vol,180), 37)

        # 计算 CORR((VWAP*0.3 + OPEN*0.7), SUM(MEAN(vol,180), 37), 14)
        def rolling_corr(series1, series2, window):
            result = pd.DataFrame(index=series1.index, columns=series1.columns)
            for col in series1.columns:
                result[col] = series1[col].rolling(window).corr(series2[col])
            return result
        
        corr_result = rolling_corr(vwap_open_comb, sum_mean_vol, 14)

        # 计算 DECAYLINEAR(CORR(...), 12)
        decay_corr = decay_linear(corr_result, 12)

        # 计算 RANK(DECAYLINEAR(...))
        rank2 = decay_corr.rank(axis=1, pct=True)

        # 最终因子计算
        factor = (rank1 - rank2) * -1
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha39",
    "creator": "chenxutian",
    "create_date": "2025.08.26",
    "formula": "((RANK(DECAYLINEAR(DELTA((CLOSE), 2),8)) - RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)), SUM(MEAN(vol,180), 37), 14), 12))) * -1)",
    "description": "alpha191因子39",
    "lookback_days": 239,
    "method": "Alpha39Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''