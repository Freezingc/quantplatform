from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha131Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)
        amount = data.xs('amount', level=0)
        vwap = amount / vol * 10
        
        # 3. 计算因子
        def calculate_factor(close, vol, vwap):
            """计算因子 RANK(DELTA(VWAP, 1)) ^ TSRANK(CORR(CLOSE, MEAN(VOLUME, 50), 18), 18)"""
            
            # 计算DELTA(VWAP, 1)
            delta_vwap = vwap.diff(1)
            
            # 计算RANK(DELTA(VWAP, 1)) - 横截面排名
            rank_delta = delta_vwap.rank(axis=1, pct=True)
            
            # 计算MEAN(VOLUME, 50)
            mean_vol_50 = vol.rolling(window=50, min_periods=30).mean()
            
            # 计算CORR(CLOSE, MEAN(VOLUME, 50), 18)
            def rolling_corr(x, y, window):
                """计算滚动相关系数"""
                return x.rolling(window=window).corr(y)
            
            corr_values = pd.DataFrame(index=close.index, columns=close.columns)
            for col in close.columns:
                corr_values[col] = rolling_corr(close[col], mean_vol_50[col], 18)
            
            # 计算TSRANK(CORR(...), 18) - 时间序列排名
            ts_rank_corr = corr_values.rolling(window=18, min_periods=10).apply(
                lambda x: (x.rank().iloc[-1] - 1) / (len(x) - 1) if len(x) > 1 else 0.5
            )
            
            # 计算最终因子
            factor = rank_delta ** ts_rank_corr
            
            return factor
        
        # 4. 计算因子
        factor = calculate_factor(close, vol, vwap)
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha131",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(RANK(DELAT(VWAP, 1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50), 18), 18))",
        "description": "基于VWAP变化率横截面排名的幂运算，其中幂是收盘价与成交量均值18天滚动相关性的时间序列排名",
        "lookback_days": 83,
        "method": "Alpha131Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
