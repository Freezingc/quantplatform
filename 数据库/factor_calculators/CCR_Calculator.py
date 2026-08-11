from .base import FactorCalculator
import numpy as np


class CCR_Calculator(FactorCalculator):
    """
    现金流动负债比率（CCR）因子：
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
        ccr = ccr.replace([np.inf, -np.inf], np.nan)
        return ccr


FACTOR_CONFIG = {
    "factor_name": "CCR",
    "creator": "zhangyudong",
    "create_date": "2026.03.04",
    "formula": "经营净现金流 / 流动负债",
    "description": "现金流动负债比率，衡量经营现金流对短期负债的覆盖能力",
    "lookback_days": 1,
    "method": "CCR_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子",
}

