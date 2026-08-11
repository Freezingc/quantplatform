from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha154Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","amount","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        amount = data.xs('amount', level=0)
        vol = data.xs('vol', level=0)  # 假设vol对应的key是'vol'
        
        # 3. 计算VWAP (成交量加权平均价)
        # VWAP = (high + low + close) / 3
        vwap = amount * 10 / vol
        
        # 4. 计算MIN(VWAP, 16) - 过去16天的最小值
        min_vwap_16 = vwap.rolling(window=16, min_periods=1).min()
        
        # 5. 计算VWAP - MIN(VWAP, 16)
        vwap_min_diff = vwap - min_vwap_16
        
        # 6. 计算MEAN(vol,180) - 过去180天的平均成交量
        mean_vol_180 = vol.rolling(window=180, min_periods=1).mean()
        
        # 7. 计算CORR(VWAP, MEAN(vol,180), 18) - 过去18天的相关系数
        # 这里需要手动计算滚动相关系数
        def rolling_corr(vwap_series, vol_series, window=18):
            """计算两个序列的滚动相关系数"""
            result = pd.DataFrame(index=vwap_series.index, columns=vwap_series.columns)
            
            for i in range(len(vwap_series)):
                if i >= window - 1:
                    start_idx = i - window + 1
                    end_idx = i + 1
                    
                    for col in vwap_series.columns:
                        vwap_window = vwap_series.iloc[start_idx:end_idx][col]
                        vol_window = vol_series.iloc[start_idx:end_idx][col]
                        
                        # 计算相关系数
                        if vwap_window.notna().all() and vol_window.notna().all():
                            corr = vwap_window.corr(vol_window)
                            result.iloc[i, result.columns.get_loc(col)] = corr
            return result
        
        corr_vwap_vol = rolling_corr(vwap, mean_vol_180, window=18)
        
        # 8. 计算最终因子：((VWAP - MIN(VWAP, 16)) < CORR(VWAP, MEAN(vol,180), 18))
        # 这里返回布尔值，如果需要数值可以转换为1/0
        factor = (vwap_min_diff < corr_vwap_vol).astype(float)
        
        # 确保factor的格式与输入数据一致
        factor = factor.reindex_like(vwap)
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha154",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(((VWAP - MIN(VWAP, 16))) < (CORR(VWAP, MEAN(vol,180), 18)))",
        "description": "VWAP偏离16日低点的幅度小于VWAP与180日均量之间18日相关系数的股票",
        "lookback_days": 198,
        "method": "Alpha154Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
