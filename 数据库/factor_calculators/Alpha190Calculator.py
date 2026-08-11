from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha190Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        
        # 计算收益率
        returns = close / close.shift(1) - 1
        
        # 计算20日平均收益率
        avg_20d_return = (close / close.shift(20)) ** (1/20) - 1
        
        # 初始化结果
        factor = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        
        # 滑动窗口计算
        window_size = 20
        
        for i in range(window_size, len(close)):
            # 获取窗口数据
            start_idx = i - window_size
            end_idx = i
            
            # 窗口内的收益率和平均收益率
            window_returns = returns.iloc[start_idx:end_idx]
            window_avg = avg_20d_return.iloc[start_idx:end_idx]
            
            # 计算条件
            up_condition = window_returns > window_avg
            down_condition = window_returns < window_avg
            
            # 计算数量
            count_up = up_condition.sum(axis=0)
            count_down = down_condition.sum(axis=0)
            
            # 计算方差和
            # 上涨日的方差和
            up_diff = (window_returns - window_avg) * up_condition
            sum_sq_up = (up_diff ** 2).sum(axis=0)
            
            # 下跌日的方差和
            down_diff = (window_returns - window_avg) * down_condition
            sum_sq_down = (down_diff ** 2).sum(axis=0)
            
            # 计算因子值
            numerator = (count_up - 1) * sum_sq_down
            denominator = count_down * sum_sq_up
            
            # 避免除零错误
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = numerator / denominator
                factor_vals = np.log(ratio)
                factor_vals[(count_down == 0) | (sum_sq_up == 0)] = np.nan
                factor_vals[denominator <= 0] = np.nan
            
            factor.iloc[i] = factor_vals
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha190",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "LOG((COUNT(CLOSE/DELAY(CLOSE)-1>((CLOSE/DELAY(CLOSE,19))^(1/20)-1),20)-1)*(SUMIF(((CLOSE/DELAY(C LOSE)-1-(CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)- 1))/((COUNT((CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1),20))*(SUMIF((CLOSE/DELAY(CLOS E)-1-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1>(CLOSE/DELAY(CLOSE,19))^(1/20)-1))) )",
        "description": "通过比较股票每日收益与20日平均收益的偏离程度，计算上涨日方差与下跌日方差的比值并取对数，衡量收益波动的非对称性特征",
        "lookback_days": 20,
        "method": "Alpha190Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "波动率因子"
    }
'''
