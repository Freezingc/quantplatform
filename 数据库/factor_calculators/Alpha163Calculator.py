from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha163Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high","amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        vol = data.xs('vol', level=0)  # 成交量
        amount = data.xs('amount', level=0)  # 成交金额
        
        # 3. 因子计算逻辑
        # 计算 RET（收益率），这里假设是日收益率
        ret = close.pct_change(fill_method=None)
        
        # 计算 MEAN(vol, 20) - 20日成交量均值
        mean_vol_20 = vol.rolling(window=20, min_periods=1).mean()
        
        # 计算 VWAP（成交量加权平均价）
        vwap = amount * 10 / vol
        
        # 计算因子中间部分：(-1 * RET) * MEAN(vol,20) * VWAP
        intermediate = (-1 * ret) * mean_vol_20 * vwap
        
        # 计算 (HIGH - CLOSE)
        high_close_diff = high - close
        
        # 计算完整因子表达式
        factor_value = intermediate * high_close_diff
        
        # 4. 计算横截面排名（RANK）
        # 对每个交易日，对所有股票进行排名
        def cross_section_rank(df):
            # 使用百分比排名，值在[0, 1]之间
            return df.rank(axis=1, pct=True)
        
        factor = cross_section_rank(factor_value)
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha163",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "RANK(((((-1 * RET) * MEAN(vol,20)) * VWAP) * (HIGH - CLOSE)))",
        "description": "基于收益率反转、成交量动量、价格位置和盘口强度综合构建的复合反转因子",
        "lookback_days": 19,
        "method": "Alpha163Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
