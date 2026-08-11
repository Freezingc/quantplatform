from .base import FactorCalculator
import numpy as np


class Profit_Calculator(FactorCalculator):
    """
    综合盈利因子（Profit）：
    Profit = CFOA + ROE + ROIC （等权相加）

    说明：
    - CFOA: 经营活动现金流 / 总资产
    - ROE: 净资产收益率
    - ROIC: 投资资本回报率
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)

        # 所需因子：经营活动现金流（CFOA）、ROE、ROIC
        self.required_cashflow_factors = [
            "c_inf_fr_operate_a"  # 经营活动现金流
        ]
        self.required_financial_factors = [
            "roe",  # 净资产收益率（ROE）
            "roic"  # 投资资本回报率（ROIC）
        ]
        self.required_balance_sheet_factors = ["total_cur_assets", "total_nca"]  # 资产负债表因子

    def _calculate(self, start_time, end_time):
        # 1) 取数
        cashflow_data = self.fetch_cashflow_data(start_time, end_time)
        financial_data = self.fetch_financial_data(start_time, end_time)
        balance_sheet_data = self.fetch_balance_sheet_data(start_time, end_time)

        # 获取经营活动现金流、ROE、ROIC
        cfoa = cashflow_data.xs("c_inf_fr_operate_a", level=0)  # 经营活动现金流
        roe = financial_data.xs("roe", level=0)  # 净资产收益率
        roic = financial_data.xs("roic", level=0)  # 投资资本回报率
        total_cur_assets = balance_sheet_data.xs("total_cur_assets", level=0)
        total_nca = balance_sheet_data.xs("total_nca", level=0)

        # 2) 计算因子：CFOA（经营活动现金流 / 总资产）
        total_assets = total_cur_assets + total_nca
        cfoa_ratio = cfoa / total_assets.replace(0, np.nan)

        # 3) 计算综合盈利因子（等权相加）
        profit = (cfoa_ratio + roe + roic) / 3  # 等权相加

        profit = profit.replace([np.inf, -np.inf], np.nan)  # 替换无效值
        return profit


FACTOR_CONFIG = {
    "factor_name": "Profit",
    "creator": "zhangyudong",
    "create_date": "2026.03.03",
    "formula": "CFOA + ROE + ROIC（等权相加）",
    "description": "综合盈利因子，衡量公司盈利能力的综合表现",
    "lookback_days": 750,
    "method": "Profit_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "盈利因子",
}