from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha125Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)  # 假设vol在数据中
        amount = data.xs('amount', level=0)  # 假设amount在数据中
        vwap = amount / vol * 10  # 计算VWAP

        # 第一步: 计算MEAN(VOLUME, 80)
        mean_vol_80 = vol.rolling(window=80, min_periods=1).mean()
        
        # 第二步: 计算CORR((VWAP), MEAN(VOLUME,80), 17)
        # 使用rolling窗口计算相关系数
        def rolling_corr(series1, series2, window):
            corr_df = pd.DataFrame(index=series1.index, columns=series1.columns)
            for col in series1.columns:
                s1 = series1[col]
                s2 = series2[col]
                # 计算滚动相关系数
                corr = s1.rolling(window=window, min_periods=1).corr(s2)
                corr_df[col] = corr
            return corr_df
        
        corr_vwap_vol = rolling_corr(vwap, mean_vol_80, 17)
        
        # 第三步: 计算DECAYLINEAR(..., 20)
        def decay_linear(series, window):
            """
            计算线性衰减权重
            权重: w_i = (window - i) / sum(1:window)
            """
            decayed = pd.DataFrame(index=series.index, columns=series.columns)
            
            for col in series.columns:
                s = series[col]
                result = []
                for i in range(len(s)):
                    if i < window - 1:
                        result.append(np.nan)
                    else:
                        window_data = s.iloc[i - window + 1: i + 1].values
                        # 计算线性衰减权重
                        weights = np.arange(window, 0, -1)
                        weights = weights / weights.sum()
                        # 加权平均
                        decayed_value = np.nansum(window_data * weights)
                        result.append(decayed_value)
                decayed[col] = result
            return decayed
        
        decay_corr = decay_linear(corr_vwap_vol, 20)
        
        # 第四步: 计算RANK(...)
        def rank_data(df):
            """按行计算横截面排名"""
            return df.rank(axis=1, pct=True)
        
        numerator = rank_data(decay_corr)
        
        # 计算分母部分: RANK(DECAYLINEAR(DELTA(((CLOSE * 0.5) + (VWAP * 0.5)), 3), 16))
        
        # 第一步: 计算 ((CLOSE * 0.5) + (VWAP * 0.5))
        close_vwap_avg = close * 0.5 + vwap * 0.5
        
        # 第二步: 计算DELTA(..., 3) - 3日差分
        delta_3 = close_vwap_avg.diff(periods=3)
        
        # 第三步: 计算DECAYLINEAR(..., 16)
        decay_delta = decay_linear(delta_3, 16)
        
        # 第四步: 计算RANK(...)
        denominator = rank_data(decay_delta)
        
        # 第五步: 计算最终因子: 分子 / 分母
        # 避免除零错误
        factor = numerator / denominator.replace(0, np.nan)
        
        # 第六步: 处理异常值
        # 可以使用winsorize或clip方法处理极端值
        factor = factor.clip(lower=factor.quantile(0.01), 
                            upper=factor.quantile(0.99), 
                            axis=1)
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha125",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME,80),17), 20)) / RANK(DECAYLINEAR(DELTA(((CLOSE * 0.5) + (VWAP * 0.5)), 3), 16)))",
        "description": "衡量VWAP与成交量均值的20日衰减相关性排名除以价格混合指标的16日衰减变化排名，来捕捉价格与成交量关系的相对强度相对于价格动量变化的比值信号",
        "lookback_days": 114,
        "method": "Alpha125Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
