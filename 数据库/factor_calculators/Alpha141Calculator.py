from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha141Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        high = data.xs('high', level=0)
        vol = data.xs('vol', level=0)
        
        # 计算横截面排名函数
        def cross_sectional_rank(df):
            """计算横截面排名（分位数形式）"""
            return df.rank(axis=1, pct=True)
        
        # 1. 计算RANK(HIGH)
        rank_high = cross_sectional_rank(high)
        
        # 2. 计算15日成交量移动平均
        mean_vol_15 = vol.rolling(window=15, min_periods=5).mean()
        
        # 3. 计算RANK(MEAN(vol, 15))
        rank_mean_vol = cross_sectional_rank(mean_vol_15)
        
        # 4. 计算9日滚动相关性
        # 这里使用更高效的方法
        corr_9 = pd.DataFrame(index=rank_high.index, columns=rank_high.columns)
        
        # 对每个股票单独计算
        for col in rank_high.columns:
            # 获取该股票的数据
            x_series = rank_high[col]
            y_series = rank_mean_vol[col]
            
            # 计算9日滚动相关性
            correlation = pd.Series(index=x_series.index, dtype=float)
            
            for i in range(8, len(x_series)):  # 从第9天开始
                x_window = x_series.iloc[i-8:i+1]
                y_window = y_series.iloc[i-8:i+1]
                
                # 计算相关性
                valid_mask = (~pd.isna(x_window)) & (~pd.isna(y_window))
                if valid_mask.sum() >= 2:
                    correlation.iloc[i] = x_window[valid_mask].corr(y_window[valid_mask])
                else:
                    correlation.iloc[i] = np.nan
            
            corr_9[col] = correlation
        
        # 5. 对相关性进行横截面排名
        rank_corr = cross_sectional_rank(corr_9)
        
        # 6. 乘以-1得到最终因子
        factor = rank_corr * -1
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha141",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9))* -1)",
        "description": "这个因子衡量的是股票的高价排名与成交量排名之间的负向相关性",
        "lookback_days": 22,
        "method": "Alpha141Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
