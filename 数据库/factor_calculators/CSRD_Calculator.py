from .base import FactorCalculator, lag_by_quarters
import numpy as np


class CSRD_Calculator(FactorCalculator):
    """
    现金比率变动（CSRD）因子：
    CSRD = 当期 CSR - 上期 CSR

    其中：
    CSR = 现金及现金等价物 / 流动负债

    字段说明（与 CSR 保持一致）：
    - n_incr_cash_cash_equ: 现金及现金等价物净增加额（此处按现有 CSR 因子口径使用）
    - total_cur_liab: 流动负债合计
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_balance_sheet_factors = ["total_cur_liab"]
        self.required_cashflow_factors = ["n_incr_cash_cash_equ"]

    def _calculate(self, start_time, end_time):
        balance_data = self.fetch_balance_sheet_data(start_time, end_time)
        cashflow_data = self.fetch_cashflow_data(start_time, end_time)

        cash_equivalents = cashflow_data.xs("n_incr_cash_cash_equ", level=0)
        total_cur_liab = balance_data.xs("total_cur_liab", level=0)

        csr = cash_equivalents / total_cur_liab.replace(0, np.nan)
        csr_lag = lag_by_quarters(csr, 1)
        csrd = csr - csr_lag

        csrd = csrd.replace([np.inf, -np.inf], np.nan)
        return csrd


FACTOR_CONFIG = {
    "factor_name": "CSRD",
    "creator": "zhangyudong",
    "create_date": "2026.03.04",
    "formula": "当期CSR - 上期CSR",
    "description": "现金比率变动，衡量短期偿债能力（现金覆盖）的边际变化",
    "lookback_days": 750,
    "method": "CSRD_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子",
}

