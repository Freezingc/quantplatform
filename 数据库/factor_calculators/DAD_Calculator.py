from .base import FactorCalculator, lag_by_quarters
import numpy as np
import pandas as pd


class DAD_Calculator(FactorCalculator):
    """
    资产负债率变动：
    DAD = 当期资产负债率 - 上期资产负债率
    其中资产负债率 = 总负债 / 总资产
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

        debt_ratio = total_liab / total_assets.replace(0, np.nan)
        debt_ratio_lag = lag_by_quarters(debt_ratio, 1)
        dad = debt_ratio - debt_ratio_lag

        dad = dad.replace([np.inf, -np.inf], np.nan)
        return dad


FACTOR_CONFIG = {
    "factor_name": "DAD",
    "creator": "zhangyudong",
    "create_date": "2026.01.29",
    "formula": "当期资产负债率 - 上期资产负债率",
    "description": "资产负债率变动，衡量杠杆水平的边际上升或下降",
    "lookback_days": 400,
    "method": "DAD_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子",
}

