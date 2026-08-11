from .base import FactorCalculator, cumulative_to_single_quarter, lag_by_quarters, rolling_sum_quarters
import numpy as np
import pandas as pd

class TOE_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_balance_sheet_factors = [
            "taxes_payable",
            "total_cur_assets",
            "total_nca",
            "total_cur_liab",
            "total_ncl",
        ]
        self.required_cashflow_factors = ["c_paid_for_taxes"]

    def _calculate(self, start_time, end_time):
        # 1. 取数
        balance_sheet_data = self.fetch_balance_sheet_data(start_time, end_time)
        cashflow_data = self.fetch_cashflow_data(start_time, end_time)

        tax_payable = balance_sheet_data.xs("taxes_payable", level=0)
        total_cur_assets = balance_sheet_data.xs("total_cur_assets", level=0)
        total_nca = balance_sheet_data.xs("total_nca", level=0)
        total_cur_liab = balance_sheet_data.xs("total_cur_liab", level=0)
        total_ncl = balance_sheet_data.xs("total_ncl", level=0)
        pay_all_typ_cum = cashflow_data.xs("c_paid_for_taxes", level=0)

        # 2. 净资产
        total_equity = (total_cur_assets + total_nca) - (
            total_cur_liab + total_ncl
        )

        # ===============================
        # 日频近似：单季度缴税 + TTM + 去年同期
        # ===============================
        pay_all_typ = cumulative_to_single_quarter(pay_all_typ_cum)
        pay_all_typ_ttm = rolling_sum_quarters(pay_all_typ, n_quarters=4)
        tax_payable_lag = lag_by_quarters(tax_payable, 4)

        numerator = tax_payable - tax_payable_lag + pay_all_typ_ttm
        toe = numerator / total_equity

        toe = toe.replace([np.inf, -np.inf], np.nan)
        return toe

# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "TOE",
    "creator": "zhangyudong",
    "create_date": "2026.01.04",
    "formula": "(当期应交税费 - 上年同期应交税费 + 缴纳税费现金流TTM) / 净资产",
    "description": "应交税费占比，衡量企业应交税费变动与实际缴税情况相对于净资产的比例。该指标反映企业税务管理效率和纳税能力",
    "lookback_days": 750,
    "method": "TOE_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子"
}


'''
# 说明：
# 1. 数据来源：
#    - taxes_payable: 资产负债表 balance_sheet，应交税费
#    - total_cur_assets: 资产负债表 balance_sheet，流动资产合计
#    - total_nca: 资产负债表 balance_sheet，非流动资产合计
#    - total_cur_liab: 资产负债表 balance_sheet，流动负债合计
#    - total_ncl: 资产负债表 balance_sheet，非流动负债合计
#    - c_paid_for_taxes: 现金流量表 cashflow_table，支付的各项税费
#    - 净资产 = (流动资产 + 非流动资产) - (流动负债 + 非流动负债)
#
# 2. 计算逻辑：
#    - 净资产 = (流动资产 + 非流动资产) - (流动负债 + 非流动负债)
#    - 分子 = 当期应交税费 - 上年同期应交税费 + 缴纳税费现金流TTM
#      * 应交税费增加量：taxes_payable - taxes_payable_{t-4}
#      * 加上实际缴纳的税费（TTM）：c_paid_for_taxes_ttm
#    - 分母 = 净资产
#
# 3. 因子意义：
#    - 正值且较大：企业应交税费增加，且实际缴税较多，可能业务增长或税务管理积极
#    - 负值：应交税费减少，可能享受税收优惠或盈利下降
#    - 接近0：应交税费变化不大，税务情况稳定
#
# 4. 注意事项：
#    - 需要至少4个季度的历史数据才能计算（TTM和同比）
#    - 净资产为负的企业该因子可能异常
#    - 季节性因素可能影响应交税费的季度波动
'''

