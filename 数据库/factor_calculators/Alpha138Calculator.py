from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha138Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","low","amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        vol = data.xs('vol', level=0)  # 假设成交量列为'vol'
        amount = data.xs('amount', level=0)  # 假设成交额列为'amount'
        vwap = amount / vol * 10
        
        # 3. 辅助函数定义
        def ts_rank(series, window):
            """时间序列排名"""
            return series.rolling(window=window).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )
        
        def decay_linear(series, window):
            """线性衰减加权"""
            weights = np.arange(1, window + 1)
            weights = weights / weights.sum()
            
            def weighted_mean(x):
                if len(x) < window:
                    return np.nan
                return np.sum(x[-window:] * weights)
            
            return series.rolling(window=window).apply(weighted_mean, raw=False)
        
        def delta(series, period):
            """差值"""
            return series - series.shift(period)
        
        def corr_ts(series1, series2, window):
            """滚动相关系数"""
            return series1.rolling(window=window).corr(series2)
        
        def rank(series):
            """横截面排名"""
            return series.rank(axis=1, pct=True)
        
        # 4. 计算因子
        # 第一部分: RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP * 0.3))), 3), 20))
        part1 = (low * 0.7 + vwap * 0.3)
        part1_delta = delta(part1, 3)
        part1_decay = decay_linear(part1_delta, 20)
        part1_rank = rank(part1_decay)
        
        # 第二部分: TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME,60), 17), 5), 19), 16), 7)
        # 2.1 内层TSRANK
        tsrank_low = ts_rank(low, 8)
        mean_vol = vol.rolling(window=60).mean()
        tsrank_vol = ts_rank(mean_vol, 17)
        
        # 2.2 CORR
        corr_part = corr_ts(tsrank_low, tsrank_vol, 5)
        
        # 2.3 中间TSRANK
        tsrank_corr = ts_rank(corr_part, 19)
        
        # 2.4 DECAYLINEAR
        decay_part = decay_linear(tsrank_corr, 16)
        
        # 2.5 外层TSRANK
        part2 = ts_rank(decay_part, 7)
        
        # 5. 组合因子
        factor = (part1_rank - part2) * -1
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha138",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "((RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP *0.3))), 3), 20)) - TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME,60), 17), 5), 19), 16), 7)) * -1)",
        "description": "一个结合价格动量衰减与成交量相关性时序排名的复合反转因子，通过比较近期加权价格变化排名与量价联动模式的长期时序排名的相对强弱来捕捉反转信号",
        "lookback_days": 118,
        "method": "Alpha138Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
