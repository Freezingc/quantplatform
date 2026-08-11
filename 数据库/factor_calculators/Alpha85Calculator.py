from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha85Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol", "close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)

        # 3. 实现因子计算逻辑
        # 计算第一部分：TSRANK((vol / MEAN(vol,20)), 20)
        mean_vol_20 = vol.rolling(window=20).mean()
        vol_ratio = vol / mean_vol_20
        
        def ts_rank(series, window):
            """
            计算时间序列排名
            series: 时间序列数据
            window: 滚动窗口大小
            """
            def calculate_rank(window_data):
                if len(window_data) < window:
                    return np.nan
                # 计算当前值在窗口内的排名（百分比排名）
                return (np.argsort(np.argsort(window_data))[-1] + 1) / len(window_data)
            
            return series.rolling(window=window).apply(calculate_rank, raw=True)
        
        # 计算两个TSRANK
        tsrank_vol = vol_ratio.apply(lambda col: ts_rank(col, 20), axis=0)
        
        # 计算第二部分：TSRANK((-1 * DELTA(CLOSE, 7)), 8)
        delta_close_7 = close.diff(7)
        neg_delta_close_7 = -1 * delta_close_7
        tsrank_close = neg_delta_close_7.apply(lambda col: ts_rank(col, 8), axis=0)
        
        # 计算最终因子
        factor = tsrank_vol * tsrank_close
        
        # 确保factor的格式与输入一致（相同的索引和列）
        factor = factor.reindex(index=close.index, columns=close.columns)
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha85",
        "creator": "chenxutian",
        "create_date": "2025.10.05",
        "formula": "(TSRANK((vol / MEAN(vol,20)), 20) * TSRANK((-1 * DELTA(CLOSE, 7)), 8))",
        "description": "alpha191因子85",
        "lookback_days": 38,
        "method": "Alpha85Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.09"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

