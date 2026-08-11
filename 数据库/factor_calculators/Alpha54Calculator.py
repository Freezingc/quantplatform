from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha54Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "open"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        open_ = data.xs('open', level=0)  # 需要open数据
        
        # 3. 计算因子
        # 计算CLOSE - OPEN
        close_open_diff = close - open_
        
        # 计算STD(ABS(CLOSE - OPEN))，使用滚动窗口标准差，这里假设窗口为10
        std_abs_close_open = close_open_diff.abs().rolling(window=10).std()
        
        # 计算CORR(CLOSE, OPEN, 10)
        corr_close_open = close.rolling(window=10).corr(open_)
        
        # 组合各部分：(STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)
        combined = (std_abs_close_open + close_open_diff) + corr_close_open
        
        # 对combined进行排名
        # 使用rank方法，axis=1表示对每行进行排名
        ranked = combined.rank(axis=1)
        
        # 乘以-1
        factor = -1 * ranked
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha54",
        "creator": "chenxutian",
        "create_date": "2025.09.01",
        "formula": "(-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)))",
        "description": "alpha191因子54",
        "lookback_days": 9,
        "method": "Alpha54Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''
