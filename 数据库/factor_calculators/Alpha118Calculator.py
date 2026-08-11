from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha118Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low","open","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        high = data.xs('high', level=0)
        open = data.xs('open', level=0)  # 注意：原代码中缺少open
        
        # 3. 计算因子 SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*118
        # HIGH - OPEN
        high_minus_open = high - open
        
        # OPEN - LOW
        open_minus_low = open - low
        
        # 计算20日滚动和
        sum_high_open_20 = high_minus_open.rolling(window=20, min_periods=1).sum()
        sum_open_low_20 = open_minus_low.rolling(window=20, min_periods=1).sum()
        
        # 避免除零错误
        sum_open_low_20 = sum_open_low_20.replace(0, np.nan)
        
        # 计算最终因子
        factor = (sum_high_open_20 / sum_open_low_20) * 118
        
        # 4. 确保factor的格式与输入一致
        # 重新构建多级索引
        factor = factor.stack().rename('value').reset_index()
        factor['factor_name'] = 'high_low_ratio'  # 可以自定义因子名称
        
        # 重新pivot为与输入相同的格式
        factor_pivot = factor.pivot(
            index='trade_date', 
            columns='ts_code', 
            values='value'
        )
        
        return factor_pivot


'''
# 添加因子信息
    {
        "factor_name": "Alpha118",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100",
        "description": "衡量了过去20日内价格上涨动能（最高价与开盘价之差的总和）相对于价格下跌保护（开盘价与最低价之差的总和）的比率，反映了多头攻击力度与空头防御能力的相对强弱",
        "lookback_days": 19,
        "method": "Alpha118Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "波动率因子"
    }
'''
