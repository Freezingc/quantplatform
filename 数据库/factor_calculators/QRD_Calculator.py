from .base import FactorCalculator, lag_by_quarters
import numpy as np


class QRD_Calculator(FactorCalculator):
    """
    速动比率变动（QRD）因子：
    QRD = 当期速动比率 - 上期速动比率

    其中速动比率 QR 定义为：
    QR = (流动资产 - 存货 - 1年内到期的非流动资产 - 待摊费用 - 预付款) / 流动负债

    字段映射（来自 balance_sheet 表字段，与 QR 保持一致）：
    - total_cur_assets: 流动资产合计
    - inventories: 存货
    - nca_within_1y: 1年内到期的非流动资产
    - amor_exp: 待摊费用
    - prepayment: 预付款
    - total_cur_liab: 流动负债合计
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_balance_sheet_factors = [
            "total_cur_assets",
            "inventories",
            "nca_within_1y",
            "amor_exp",
            "prepayment",
            "total_cur_liab",
        ]

    def _calculate(self, start_time, end_time):
        balance_data = self.fetch_balance_sheet_data(start_time, end_time)

        total_cur_assets = balance_data.xs("total_cur_assets", level=0)
        inventories = balance_data.xs("inventories", level=0)
        nca_within_1y = balance_data.xs("nca_within_1y", level=0)
        amor_exp = balance_data.xs("amor_exp", level=0)
        prepayment = balance_data.xs("prepayment", level=0)
        total_cur_liab = balance_data.xs("total_cur_liab", level=0)

        quick_assets = total_cur_assets - inventories - nca_within_1y - amor_exp - prepayment
        qr = quick_assets / total_cur_liab.replace(0, np.nan)

        qr_lag = lag_by_quarters(qr, 1)
        qrd = qr - qr_lag

        qrd = qrd.replace([np.inf, -np.inf], np.nan)
        return qrd


FACTOR_CONFIG = {
    "factor_name": "QRD",
    "creator": "zhangyudong",
    "create_date": "2026.03.04",
    "formula": "当期速动比率 - 上期速动比率",
    "description": "速动比率变动，衡量短期偿债能力（速动比率）的边际变化",
    "lookback_days": 750,
    "method": "QRD_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子",
}

