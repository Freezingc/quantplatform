from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha156Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","low","open","amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        low = data.xs('low', level=0)
        open = data.xs('open', level=0)
        vol = data.xs('vol', level=0)
        amount = data.xs('amount', level=0)

        vwap = amount * 10 / vol  # 计算VWAP
        
    # 3. 计算第一个部分：RANK(DECAYLINEAR(DELTA(VWAP, 5), 3))
        # DELTA(VWAP, 5)
        delta_vwap_5 = vwap.diff(5)
        
        # DECAYLINEAR(DELTA(VWAP, 5), 3)
        weights_3 = np.array([3, 2, 1]) / 6  # 加权权重：3/6, 2/6, 1/6
        decay_vwap = delta_vwap_5.rolling(window=3).apply(
            lambda x: np.sum(x * weights_3[:len(x)]) if len(x) == 3 else np.nan
        )
        
        # RANK(DECAYLINEAR(DELTA(VWAP, 5), 3))
        rank_vwap = decay_vwap.rank(axis=1, pct=True)
        
        # 4. 计算第二个部分：RANK(DECAYLINEAR(((DELTA(((OPEN * 0.15) + (LOW *0.85)), 2) / ((OPEN * 0.15) + (LOW * 0.85))) * -1), 3))
        # 计算 (OPEN * 0.15) + (LOW * 0.85)
        ol_combination = open * 0.15 + low * 0.85
        
        # DELTA(((OPEN * 0.15) + (LOW *0.85)), 2)
        delta_ol_2 = ol_combination.diff(2)
        
        # ((DELTA(((OPEN * 0.15) + (LOW *0.85)), 2) / ((OPEN * 0.15) + (LOW * 0.85))) * -1)
        ratio_ol = (delta_ol_2 / ol_combination) * -1
        
        # DECAYLINEAR(ratio_ol, 3)
        decay_ol = ratio_ol.rolling(window=3).apply(
            lambda x: np.sum(x * weights_3[:len(x)]) if len(x) == 3 else np.nan
        )
        
        # RANK(DECAYLINEAR(...))
        rank_ol = decay_ol.rank(axis=1, pct=True)
        
        # 5. 计算最终因子
        # MAX(rank_vwap, rank_ol) * -1
        max_rank = pd.DataFrame(np.maximum(rank_vwap.values, rank_ol.values), 
                            index=rank_vwap.index, 
                            columns=rank_vwap.columns)
        
        factor = max_rank * -1
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha156",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(MAX(RANK(DECAYLINEAR(DELTA(VWAP, 5), 3)), RANK(DECAYLINEAR(((DELTA(((OPEN * 0.15) + (LOW *0.85)), 2) / ((OPEN * 0.15) + (LOW * 0.85))) * -1), 3))) * -1)",
        "description": "通过比较VWAP的5日差分动量与开盘低价组合的2日反转强度，选取两者中较强的信号作为做空依据",
        "lookback_days": 8,
        "method": "Alpha156Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
