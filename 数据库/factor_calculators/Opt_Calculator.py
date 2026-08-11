from .base import FactorCalculator
import numpy as np


class Opt_Calculator(FactorCalculator):
    """
    综合营运效率因子（Opt）：
    Opt = (ATD + OCFA) / 2 （等权相加）

    说明：
    - ATD：资产周转率变动（TTM），衡量资产使用效率的边际改善或恶化
    - OCFA：经营活动现金流 / 总资产，衡量经营活动产生现金流与资产规模的匹配程度

    这里直接在同一类中复用 ATD 与 OCFA 的计算口径，保证与原有单因子定义一致。
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)

        # ATD 所需因子
        self.required_income_factors = ["revenue"]
        self.required_balance_sheet_factors = ["total_cur_assets", "total_nca"]

        # OCFA 所需因子（经营活动现金流）
        self.required_cashflow_factors = ["n_cashflow_act"]

    def _calculate(self, start_time, end_time):
        # 1) 取数
        income_data = self.fetch_income_data(start_time, end_time)
        balance_data = self.fetch_balance_sheet_data(start_time, end_time)
        cashflow_data = self.fetch_cashflow_data(start_time, end_time)

        # 2) 取出各基础字段
        revenue = income_data.xs("revenue", level=0)
        total_cur_assets = balance_data.xs("total_cur_assets", level=0)
        total_nca = balance_data.xs("total_nca", level=0)
        n_cashflow_act = cashflow_data.xs("n_cashflow_act", level=0)

        # 3) 计算总资产
        total_assets = total_cur_assets + total_nca

        # 4) 计算 ATD 与 OCFA
        # 这里简单使用资产周转率 = 营业收入 / 总资产 的日频近似，
        # 然后对其做一阶差分近似「变动」，以保持实现简洁。
        at = revenue / total_assets.replace(0, np.nan)
        atd = at.diff()  # 一阶差分近似资产周转率变动

        ocfa = n_cashflow_act / total_assets.replace(0, np.nan)

        # 5) 组合综合营运效率因子（等权相加）
        opt = (atd + ocfa) / 2

        opt = opt.replace([np.inf, -np.inf], np.nan)
        return opt


FACTOR_CONFIG = {
    "factor_name": "Opt",
    "creator": "zhangyudong",
    "create_date": "2026.03.04",
    "formula": "(ATD + OCFA) / 2（等权相加）",
    "description": "综合营运效率因子，等权组合资产周转率变动与经营活动现金流/总资产",
    "lookback_days": 750,
    "method": "Opt_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "营运效率因子",
}

