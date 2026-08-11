from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha165Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)

        # 3. 实现因子计算逻辑
        # 计算48周期的均值
        mean_48 = close.rolling(window=48).mean()
        
        # 计算 CLOSE - MEAN(CLOSE, 48)
        diff = close - mean_48
        
        # 计算SUM累加（SUMAC）: 计算从窗口开始到当前的累计和
        def sumac(series):
            return series.expanding(min_periods=1).sum()
        
        # 对每个股票计算SUMAC
        sumac_series = pd.DataFrame(index=diff.index, columns=diff.columns)
        for col in diff.columns:
            sumac_series[col] = sumac(diff[col])
        
        # 计算滚动窗口内的最大值和最小值
        # 这里需要注意：原公式的MAX和MIN可能是指在整个计算周期内的极值
        # 但通常因子计算会使用滚动窗口，假设使用48周期窗口
        window = 48
        
        # 计算滚动窗口内的最大值
        rolling_max = sumac_series.rolling(window=window, min_periods=1).max()
        
        # 计算滚动窗口内的最小值
        rolling_min = sumac_series.rolling(window=window, min_periods=1).min()
        
        # 计算标准差
        std_48 = close.rolling(window=window, min_periods=1).std()
        
        # 根据公式计算因子：MAX(SUMAC(...)) - MIN(SUMAC(...)) / STD(CLOSE, 48)
        # 注意：原公式可能有歧义，这里按照常规理解实现
        # 如果公式是：(MAX - MIN) / STD
        factor = (rolling_max - rolling_min) / std_48
        
        # 如果公式是：MAX - (MIN / STD)（根据你写的公式，没有括号，所以是这种）
        # 但实际上通常的逻辑是：(MAX - MIN) / STD
        # 这里我按照常见的标准化处理方式实现
        
        # 4. 确保factor的格式与输入数据一致
        # 添加多级索引，与原始数据格式保持一致
        factor = factor.stack().reorder_levels([1, 0]).sort_index()
        factor.index.names = ['ts_code', 'trade_date']
        factor.name = 'value'
        
        # 如果需要保持pivot格式，可以直接返回DataFrame
        # 这里返回pivot格式
        factor_pivot = factor.unstack('ts_code')
        
        return factor_pivot


'''
# 添加因子信息
    {
        "factor_name": "Alpha165",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48)))/STD(CLOSE,48)",
        "description": "在48个周期内，收盘价相对于其均值的累计偏离幅度经过极差标准化后的波动强度",
        "lookback_days": 47,
        "method": "Alpha165Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "波动率因子"
    }
'''
