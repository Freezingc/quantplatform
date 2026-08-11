from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha108Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","amount","high","low"]  # 指定该因子需要的原始因子

    # 如果VWAP需要更精确计算
    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        high = data.xs('high', level=0)
        vol = data.xs('vol', level=0)
        amount = data.xs('amount', level=0)
        
        # 使用amount/vol计算VWAP，注意单位转换
        vwap = amount / vol * 10
        
     # 4. 计算 HIGH - MIN(HIGH, 2)
        # 使用rolling计算最近2日的最小值
        min_high_2 = high.rolling(window=2, min_periods=2).min()
        high_minus_min = high - min_high_2
        
        # 5. 计算 MEAN(vol,120)
        mean_vol_120 = vol.rolling(window=120, min_periods=120).mean()
        
        # 6. 计算 CORR(VWAP, MEAN(vol,120), 6)
        # 计算6日滚动相关系数
        def rolling_corr(df1, df2, window):
            corr_results = pd.DataFrame(index=df1.index, columns=df1.columns)
            for i in range(window-1, len(df1)):
                window_df1 = df1.iloc[i-window+1:i+1]
                window_df2 = df2.iloc[i-window+1:i+1]
                # 对每个股票计算相关系数
                for col in df1.columns:
                    if not window_df1[col].isna().all() and not window_df2[col].isna().all():
                        corr_val = window_df1[col].corr(window_df2[col])
                        corr_results.iloc[i, df1.columns.get_loc(col)] = corr_val
            return corr_results
        
        corr_6 = rolling_corr(vwap, mean_vol_120, 6)
        
        # 7. 计算RANK
        # 对每个截面进行排名
        def cross_sectional_rank(df):
            # 使用百分比排名，NaN值保持为NaN
            return df.rank(axis=1, pct=True, na_option='keep')
        
        rank_high = cross_sectional_rank(high_minus_min)
        rank_corr = cross_sectional_rank(corr_6)
        
        # 8. 计算 RANK(high_minus_min) ^ RANK(corr)
        # 注意：当底数为负数时，指数需要是整数，这里使用绝对值避免复数
        power_result = rank_high ** rank_corr
        
        # 9. 乘以-1
        factor = power_result * -1
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha108",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "((RANK((HIGH - MIN(HIGH, 2)))^RANK(CORR((VWAP), (MEAN(vol,120)), 6))) * -1)",
        "description": "通过高点的突破强度与量价背离关系的幂运算组合来识别短期反转信号",
        "lookback_days": 124,
        "method": "Alpha108Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
