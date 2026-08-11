from .base import FactorCalculator
import numpy as np
import pandas as pd

class CFOA_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_balance_sheet_factors = ["total_cur_assets", "total_nca"]  # 资产负债表因子
        self.required_cashflow_factors = ["n_cashflow_act"]  # 现金流量表因子

    def _calculate(self, start_time, end_time):
        # 1. 获取资产负债表数据
        balance_sheet_data = self.fetch_balance_sheet_data(start_time, end_time)
        
        # 2. 获取现金流量表数据
        cashflow_data = self.fetch_cashflow_data(start_time, end_time)
        
        # 3. 提取所需因子
        total_cur_assets = balance_sheet_data.xs('total_cur_assets', level=0)  # 流动资产总额
        total_nca = balance_sheet_data.xs('total_nca', level=0)  # 非流动资产总额
        n_cashflow_act = cashflow_data.xs('n_cashflow_act', level=0)  # 经营活动现金流量净额
        
        # 4. 计算总资产
        total_assets = total_cur_assets + total_nca
        
        # 5. 计算CFOA = 经营现金流 / 总资产
        cfoa = n_cashflow_act / total_assets
        
        # 6. 处理无穷大值和无效值
        cfoa = cfoa.replace([np.inf, -np.inf], np.nan)
        
        return cfoa

'''
# 添加因子信息
FACTOR_CONFIG = {


    "factor_name": "CFOA",
    "creator": "zhangyudong",
    "create_date": "2025.12.11",
    "formula": "n_cashflow_act / (total_cur_assets + total_nca)",
    "description": "经营现金流/总资产，衡量企业通过经营活动产生现金流的能力",
    "lookback_days": 1,
    "method": "CFOA_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": ""  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
