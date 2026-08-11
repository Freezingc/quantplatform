from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha147Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        
        # 3. 因子计算逻辑
        # 计算12日序列 [1, 2, ..., 12]
        sequence_12 = np.arange(1, 13)
        
        def calculate_beta(series):
            """计算单个股票的时间序列与序列[1,2,...,12]的回归系数"""
            if len(series) < 12:
                return np.nan
            
            # 计算12日移动平均
            ma_12 = series.rolling(window=12).mean()
            
            # 为每个窗口计算回归系数
            betas = []
            for i in range(len(series)):
                if i < 11:  # 前11个位置无法计算12日均线
                    betas.append(np.nan)
                    continue
                
                # 获取当前窗口的12日均线值和对应序列
                window_ma = ma_12.iloc[i-11:i+1].values
                x = sequence_12
                
                # 确保没有NaN值
                valid_mask = ~np.isnan(window_ma)
                if valid_mask.sum() < 2:  # 至少需要2个有效点
                    betas.append(np.nan)
                    continue
                
                x_valid = x[valid_mask]
                y_valid = window_ma[valid_mask]
                
                # 计算回归系数 (Beta = Cov(x,y) / Var(x))
                cov_matrix = np.cov(x_valid, y_valid)
                if cov_matrix[0, 0] == 0:
                    betas.append(np.nan)
                else:
                    beta = cov_matrix[0, 1] / cov_matrix[0, 0]
                    betas.append(beta)
            
            return pd.Series(betas, index=series.index)
        
        # 4. 对每个股票计算因子
        factor_df = pd.DataFrame(index=close.index, columns=close.columns)
        
        for col in close.columns:
            factor_df[col] = calculate_beta(close[col])
        
        # 5. 确保factor格式与输入一致
        factor = factor_df
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha147",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "REGBETA(MEAN(CLOSE,12),SEQUENCE(12))",
        "description": "计算的是收盘价12日均线与时间序列[1,2,...,12]的线性回归斜率",
        "lookback_days": 11,
        "method": "Alpha147Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
