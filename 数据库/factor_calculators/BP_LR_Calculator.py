from .base import FactorCalculator
import numpy as np

class BP_LR_Calculator(FactorCalculator):
    """
    市净率倒数（BP_LR）因子：
    BP_LR = 最新财报净资产 / 总市值
    等价于 BP = 1 / PB

    说明：
    - 净资产使用资产负债表最新股东权益（期末值口径）
    - 总市值使用日频行情因子（total_mv）
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)

        # 财报字段：净资产（股东权益合计）
        self.required_balance_sheet_factors = [
            "total_hldr_eqy_inc_min_int"
        ]

        # 行情字段：总市值
        self.required_factors = [
            "total_mv"
        ]

    def _calculate(self, start_time, end_time):
        # 1) 取数
        balance_data = self.fetch_balance_sheet_data(start_time, end_time)
        daily_data = self.fetch_data(start_time, end_time)

        equity = balance_data.xs("total_hldr_eqy_inc_min_int", level=0)  # 净资产
        total_mv = daily_data.xs("total_mv", level=0)                   # 总市值

        # 2) 计算 BP（市净率倒数）
        bp_lr = equity / total_mv.replace(0, np.nan)

        bp_lr = bp_lr.replace([np.inf, -np.inf], np.nan)
        return bp_lr


FACTOR_CONFIG = {
    "factor_name": "BP_LR",
    "creator": "zhangyudong",
    "create_date": "2026.01.29",
    "formula": "最新财报净资产 / 总市值",
    "description": "市净率倒数（BP），衡量估值高低，数值越大代表估值越低",
    "lookback_days": 750,
    "method": "BP_LR_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "估值因子",
}
