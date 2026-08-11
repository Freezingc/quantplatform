from .base import FactorCalculator
import numpy as np
import pandas as pd


class DTE_Calculator(FactorCalculator):
    """
    产权比率：
    DTE = 总负债 / 归母股东权益

    这里归母股东权益采用资产负债表字段 total_hldr_eqy_exc_min_int。
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_balance_sheet_factors = ["total_liab", "total_hldr_eqy_exc_min_int"]

    def _calculate(self, start_time, end_time):
        balance_data = self.fetch_balance_sheet_data(start_time, end_time)
        total_liab = balance_data.xs("total_liab", level=0)
        equity_parent = balance_data.xs("total_hldr_eqy_exc_min_int", level=0)

        dte = total_liab / equity_parent.replace(0, np.nan)
        dte = dte.replace([np.inf, -np.inf], np.nan)
        return dte


FACTOR_CONFIG = {
    "factor_name": "DTE",
    "creator": "zhangyudong",
    "create_date": "2026.01.29",
    "formula": "总负债 / 归母股东权益",
    "description": "产权比率，衡量负债对股东权益的放大倍数（杠杆程度）",
    "lookback_days": 400,
    "method": "DTE_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子",
}

