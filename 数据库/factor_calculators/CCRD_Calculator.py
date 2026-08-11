from .base import FactorCalculator, lag_by_quarters
import numpy as np


class CCRD_Calculator(FactorCalculator):
    """
    现金流动负债比率变动（CCRD）因子：
    CCRD = 当期 CCR - 上期 CCR

    其中：
    CCR = 经营净现金流 / 流动负债

    字段说明：
    - n_cashflow_act: 经营活动产生的现金流量净额
    - total_cur_liab: 流动负债合计
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_cashflow_factors = ["n_cashflow_act"]
        self.required_balance_sheet_factors = ["total_cur_liab"]

    def _calculate(self, start_time, end_time):
        cashflow_data = self.fetch_cashflow_data(start_time, end_time)
        balance_data = self.fetch_balance_sheet_data(start_time, end_time)

        ocf = cashflow_data.xs("n_cashflow_act", level=0)
        cur_liab = balance_data.xs("total_cur_liab", level=0)

        ccr = ocf / cur_liab.replace(0, np.nan)

        # 上期（按季度交易日偏移法）
        ccr_lag = lag_by_quarters(ccr, 1)
        ccrd = ccr - ccr_lag

        ccrd = ccrd.replace([np.inf, -np.inf], np.nan)
        return ccrd


FACTOR_CONFIG = {
    "factor_name": "CCRD",
    "creator": "zhangyudong",
    "create_date": "2026.03.04",
    "formula": "当期CCR - 上期CCR",
    "description": "现金流动负债比率变动，衡量经营现金流覆盖短期负债能力的边际变化",
    "lookback_days": 750,
    "method": "CCRD_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子",
}

