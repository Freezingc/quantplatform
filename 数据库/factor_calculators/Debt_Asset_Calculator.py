from .base import FactorCalculator
import numpy as np
import pandas as pd


class Debt_Asset_Calculator(FactorCalculator):
    """
    资产负债比：
    Debt_Asset = 总负债 / 总资产
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_balance_sheet_factors = ["total_liab", "total_cur_assets", "total_nca"]

    def _calculate(self, start_time, end_time):
        balance_data = self.fetch_balance_sheet_data(start_time, end_time)
        total_liab = balance_data.xs("total_liab", level=0)
        total_assets = (
            balance_data.xs("total_cur_assets", level=0)
            + balance_data.xs("total_nca", level=0)
        )

        debt_asset = total_liab / total_assets.replace(0, np.nan)
        debt_asset = debt_asset.replace([np.inf, -np.inf], np.nan)
        return debt_asset


FACTOR_CONFIG = {
    "factor_name": "Debt_Asset",
    "creator": "zhangyudong",
    "create_date": "2026.01.29",
    "formula": "总负债 / 总资产",
    "description": "资产负债比，衡量企业杠杆水平",
    "lookback_days": 400,
    "method": "Debt_Asset_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子",
}

