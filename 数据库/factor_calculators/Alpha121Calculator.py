from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha121Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","amount","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        vol = data.xs('vol', level=0)
        amount = data.xs('amount', level=0)
        
        # 3. 计算VWAP (成交量加权平均价格)
        vwap = amount / vol * 10
        
        # 4. 计算MIN(VWAP, 12)
        min_vwap_12 = vwap.rolling(window=12, min_periods=1).min()
        
        # 5. 计算VWAP - MIN(VWAP, 12)
        vwap_diff = vwap - min_vwap_12
        
        # 6. 计算RANK
        def rank_data(df):
            # 对每一行（交易日）进行横截面排名
            return df.rank(axis=1, pct=True)
        
        rank_part = rank_data(vwap_diff)
        
        # 7. 辅助函数：计算时间序列排名
        def ts_rank(series, window):
            """计算时间序列排名"""
            return series.rolling(window=window).apply(
                lambda x: (x.argsort().argsort()[-1] + 1) / len(x) if len(x) == window else np.nan
            )
        
        # 8. 计算TSRANK(VWAP, 20)
        tsrank_vwap_20 = vwap.apply(lambda col: ts_rank(col, 20))
        
        # 9. 计算MEAN(VOLUME, 60)
        mean_vol_60 = vol.rolling(window=60, min_periods=1).mean()
        
        # 10. 计算TSRANK(MEAN(VOLUME, 60), 2)
        tsrank_mean_vol_2 = mean_vol_60.apply(lambda col: ts_rank(col, 2))
        
        # 11. 计算CORR(TSRANK(VWAP, 20), TSRANK(MEAN(VOLUME, 60), 2), 18)
        def rolling_corr(series1, series2, window):
            """计算滚动相关性"""
            return series1.rolling(window=window).corr(series2)
        
        corr_result = pd.DataFrame(index=vwap.index, columns=vwap.columns)
        for col in vwap.columns:
            if col in tsrank_mean_vol_2.columns:
                corr_result[col] = rolling_corr(tsrank_vwap_20[col], tsrank_mean_vol_2[col], 18)
        
        # 12. 计算TSRANK(CORR(...), 3)
        tsrank_corr_3 = corr_result.apply(lambda col: ts_rank(col, 3))
        
        # 13. 计算幂运算
        power_factor = rank_part ** tsrank_corr_3
        
        # 14. 乘以-1
        factor = power_factor * -1
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha121",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "((RANK((VWAP - MIN(VWAP, 12)))^TSRANK(CORR(TSRANK(VWAP, 20), TSRANK(MEAN(VOLUME,60), 2), 18), 3)) * -1)",
        "description": "基于VWAP相对最低位置、结合成交量与价格时间序列相关性的反转类因子，通过量化价格在近期低位区域的反弹强度与成交量模式的相关性来预测股票未来表现",
        "lookback_days": 79,
        "method": "Alpha121Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
