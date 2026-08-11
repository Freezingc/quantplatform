from .base import FactorCalculator
import numpy as np
import pandas as pd

class CSR_Calculator(FactorCalculator):
    """
    现金比率（CSR）因子：
    CSR = 现金及现金等价物 / 流动负债
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_balance_sheet_factors = [ "total_cur_liab"]  # 现金及现金等价物和流动负债
        self.required_cashflow_factors = ["n_incr_cash_cash_equ"]  # 现金流量表因子
    def _calculate(self, start_time, end_time):
        balance_data = self.fetch_balance_sheet_data(start_time, end_time)
        cashflow_data = self.fetch_cashflow_data(start_time, end_time)
        cash_equivalents = cashflow_data.xs("n_incr_cash_cash_equ", level=0)
        total_cur_liab = balance_data.xs("total_cur_liab", level=0)
        csr = cash_equivalents / total_cur_liab.replace(0, np.nan)  # 防止除以零
        csr = csr.replace([np.inf, -np.inf], np.nan)  # 替换无效值
        return csr
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "CSR",
    "creator": "zhangyudong",
    "create_date": "2026.01.29",
    "formula": "现金及现金等价物 / 流动负债",
    "description": "现金比率，衡量企业的短期偿债能力",
    "lookback_days": 1,
    "method": "CSR_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子",
}


