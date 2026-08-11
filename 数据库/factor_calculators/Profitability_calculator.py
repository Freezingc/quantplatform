from .base import FactorCalculator
import numpy as np
import pandas as pd

class Profitability_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_balance_sheet_factors = ["total_cur_assets", "total_nca"]  # 资产负债表因子
        self.required_cashflow_factors = ["n_cashflow_act"]  # 现金流量表因子
        self.required_financial_factors = ["roic","roe"] 

    def _calculate(self, start_time, end_time):
        balance_sheet_data = self.fetch_balance_sheet_data(start_time, end_time)
        cashflow_data = self.fetch_cashflow_data(start_time, end_time)
        financial_data = self.fetch_financial_data(start_time, end_time)

       
        total_cur_assets = balance_sheet_data.xs('total_cur_assets', level=0)  # 流动资产总额
        total_nca = balance_sheet_data.xs('total_nca', level=0)  # 非流动资产总额
        n_cashflow_act = cashflow_data.xs('n_cashflow_act', level=0)  # 经营活动现金流量净额
        roic = financial_data.xs('roic', level=0)  # 投资资本回报率
        roe = financial_data.xs('roe', level=0)  # 净资产收益率
        
        total_assets = total_cur_assets + total_nca
    
        cfoa = n_cashflow_act / total_assets
        
        profitability = (roic+roe+cfoa)/3
        profitability = profitability.replace([np.inf, -np.inf], np.nan)
        
        return profitability

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Profitability",
    "creator": "zhangyudong",
    "create_date": "2025.12.11",
    "formula": "(roic+roe+cfoa)/3",
    "description": "盈利能力类因子的复合因子,投资资本回报率、净资产收益率、经营现金流/总资产的平均值",
    "lookback_days": 1,
    "method": "Profitability_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": ""  ,# 可以为空，默认为当日日期，如果为空，请填写空字符串""
    "category": "基本面因子"
}
'''
