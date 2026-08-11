from .base import FactorCalculator
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

class Alpha116Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        close = data.xs('close', level=0)     
           
        # 3. 生成序列（1,2,3,...,20）
        sequence = pd.Series(range(1, 21), index=close.iloc[-20:].index)
        
        # 4. 实现REGBETA因子计算
        def calculate_beta(price_series, sequence, window=20):
            """
            计算线性回归的beta系数
            price_series: 价格序列
            sequence: 序列（1,2,3,...,20）
            window: 滚动窗口大小
            """
            betas = pd.Series(index=price_series.index, dtype=float)
            
            for i in range(window, len(price_series) + 1):
                # 获取当前窗口的数据
                window_prices = price_series.iloc[i-window:i]
                
                # 如果有缺失值，跳过
                if window_prices.isna().any():
                    betas.iloc[i-1] = np.nan
                    continue
                
                try:
                    # 进行线性回归，计算beta
                    x = sequence.values.reshape(-1, 1)  # 转换为二维数组
                    y = window_prices.values.reshape(-1, 1)
                    
                    # 添加截距项
                    x_with_intercept = np.hstack([np.ones((len(x), 1)), x])
                    
                    # 使用最小二乘法计算回归系数
                    result = np.linalg.lstsq(x_with_intercept, y, rcond=None)[0]
                    
                    # beta是斜率系数（第二个系数）
                    betas.iloc[i-1] = result[1][0]
                except:
                    betas.iloc[i-1] = np.nan
            
            return betas
        
        # 5. 对每个股票计算REGBETA
        factor_data = {}
        
        for column in close.columns:
            price_series = close[column]
            beta_series = calculate_beta(price_series, sequence, window=20)
            factor_data[column] = beta_series
        
        # 6. 转换为pivot格式
        factor = pd.DataFrame(factor_data)
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha116",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "REGBETA(CLOSE,SEQUENCE,20)",
        "description": "计算股票收盘价在过去20天内相对于时间趋势的线性回归斜率，反映价格的时间动量",
        "lookback_days": 19,
        "method": "Alpha116Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
