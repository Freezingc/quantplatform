from .base import FactorCalculator
import numpy as np
import pandas as pd

class Amihud_monthly_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["pct_chg","amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        pct_chg = data.xs('pct_chg', level=0)
        amount = data.xs('amount', level=0)
        
        factor = (abs(pct_chg) / amount).rolling(20).mean()

        
        # 6. 处理可能的无穷大和缺失值
        factor = factor.replace([np.inf, -np.inf], np.nan)
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "月度Amihud非流动性",
    "creator": "suyanning",
    "create_date": "2025.08.31",
    "formula": "SMA(ABS(RETURNS)/AMOUNT, 20)",
    "description": "基于换手率和收益率的月度Amihud非流动性指标，衡量股票流动性",
    "lookback_days": 20,
    "method": "Amihud_monthly_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": ""  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
