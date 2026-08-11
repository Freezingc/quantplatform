from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha140Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high","open","low"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        open = data.xs('open', level=0)
        vol = data.xs('vol', level=0)
        
        # 辅助函数定义
        def rank(df):
            """计算横截面排名（0-1之间）"""
            return df.rank(axis=1, pct=True)
        
        def decay_linear(df, window):
            """线性衰减加权平均"""
            if len(df) < window:
                return pd.DataFrame(index=df.index, columns=df.columns)
            
            weights = np.arange(1, window + 1)
            weights = weights / weights.sum()
            
            result = pd.DataFrame(index=df.index, columns=df.columns)
            for i in range(window - 1, len(df)):
                window_data = df.iloc[i - window + 1:i + 1]
                weighted_sum = (window_data.values * weights[:, np.newaxis]).sum(axis=0)
                result.iloc[i] = weighted_sum
            
            return result
        
        def ts_rank(df, window):
            """时间序列排名"""
            if len(df) < window:
                return pd.DataFrame(index=df.index, columns=df.columns)
            
            result = pd.DataFrame(index=df.index, columns=df.columns)
            for i in range(window - 1, len(df)):
                window_data = df.iloc[i - window + 1:i + 1]
                # 计算每个标的在时间窗口内的排名百分比
                result.iloc[i] = window_data.rank(axis=0, pct=True).iloc[-1]
            
            return result
        
        def corr(df1, df2, window):
            """计算滚动相关系数"""
            if len(df1) < window or len(df2) < window:
                return pd.DataFrame(index=df1.index, columns=df1.columns)
            
            result = pd.DataFrame(index=df1.index, columns=df1.columns)
            for i in range(window - 1, len(df1)):
                window_df1 = df1.iloc[i - window + 1:i + 1]
                window_df2 = df2.iloc[i - window + 1:i + 1]
                
                # 计算每列的相关系数
                for col in df1.columns:
                    if col in df2.columns:
                        corr_val = window_df1[col].corr(window_df2[col])
                        result.iloc[i][col] = corr_val if not np.isnan(corr_val) else 0
            
            return result
        
        def mean(df, window):
            """滚动均值"""
            return df.rolling(window=window, min_periods=1).mean()
        
        # 3. 计算第一部分: DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))), 8)
        # 3.1 计算各个排名
        rank_open = rank(open)
        rank_low = rank(low)
        rank_high = rank(high)
        rank_close = rank(close)
        
        # 3.2 计算 (RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))
        part1 = (rank_open + rank_low) - (rank_high + rank_close)
        
        # 3.3 计算 DECAYLINEAR(..., 8)
        decay_part1 = decay_linear(part1, 8)
        
        # 3.4 计算 RANK(DECAYLINEAR(...))
        rank_decay_part1 = rank(decay_part1)
        
        # 4. 计算第二部分: TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME,60), 20), 8), 7), 3)
        # 4.1 计算 TSRANK(CLOSE, 8)
        ts_rank_close = ts_rank(close, 8)
        
        # 4.2 计算 MEAN(VOLUME, 60) 然后 TSRANK(..., 20)
        mean_vol = mean(vol, 60)
        ts_rank_mean_vol = ts_rank(mean_vol, 20)
        
        # 4.3 计算 CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME,60), 20), 8)
        corr_part = corr(ts_rank_close, ts_rank_mean_vol, 8)
        
        # 4.4 计算 DECAYLINEAR(CORR(...), 7)
        decay_corr = decay_linear(corr_part, 7)
        
        # 4.5 计算 TSRANK(DECAYLINEAR(...), 3)
        ts_rank_decay_corr = ts_rank(decay_corr, 3)
        
        # 5. 计算最终因子: MIN(第一部分, 第二部分)
        # 对齐索引
        common_index = rank_decay_part1.index.intersection(ts_rank_decay_corr.index)
        rank_decay_part1 = rank_decay_part1.loc[common_index]
        ts_rank_decay_corr = ts_rank_decay_corr.loc[common_index]
        
        # 计算最小值
        factor = pd.DataFrame(index=common_index, columns=rank_decay_part1.columns)
        for col in factor.columns:
            if col in rank_decay_part1.columns and col in ts_rank_decay_corr.columns:
                factor[col] = pd.concat([rank_decay_part1[col], ts_rank_decay_corr[col]], axis=1).min(axis=1)
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha140",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "MIN(RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))), 8)), TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME,60), 20), 8), 7), 3))",
        "description": "成交量20日标准差",
        "lookback_days": 93,
        "method": "Alpha140Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "波动率因子"
    }
'''
