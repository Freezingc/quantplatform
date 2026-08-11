from .base import FactorCalculator, lag_by_quarters
import numpy as np
import pandas as pd


class CURD_Calculator(FactorCalculator):
    """
    流动比率变动：
    CURD = 当期流动比率 - 上期流动比率
    其中流动比率 = 流动资产合计 / 流动负债合计
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_balance_sheet_factors = ["total_cur_assets", "total_cur_liab"]

    def _calculate(self, start_time, end_time):
        balance_data = self.fetch_balance_sheet_data(start_time, end_time)
        cur_assets = balance_data.xs("total_cur_assets", level=0)
        cur_liab = balance_data.xs("total_cur_liab", level=0)

        cur_ratio = cur_assets / cur_liab.replace(0, np.nan)
        cur_ratio_lag = lag_by_quarters(cur_ratio, 1)
        curd = cur_ratio - cur_ratio_lag

        curd = curd.replace([np.inf, -np.inf], np.nan)
        return curd


FACTOR_CONFIG = {
    "factor_name": "CURD",
    "creator": "zhangyudong",
    "create_date": "2026.01.29",
    "formula": "当期流动比率 - 上期流动比率",
    "description": "流动比率变动，衡量短期偿债能力的边际改善或恶化",
    "lookback_days": 400,
    "method": "CURD_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子",
}

