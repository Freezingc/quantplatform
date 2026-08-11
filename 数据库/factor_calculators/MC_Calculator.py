from .base import FactorCalculator
import numpy as np


class MC_Calculator(FactorCalculator):
    """
    MC（总市值）因子：
    MC = 总市值（total_mv）

    说明：
    - 直接取日频行情中的总市值
    - 不做TTM、不做变动
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)

        # 行情字段：总市值
        self.required_factors = [
            "total_mv"
        ]

    def _calculate(self, start_time, end_time):
        # 1) 取数
        daily_data = self.fetch_data(start_time, end_time)

        total_mv = daily_data.xs("total_mv", level=0)

        # 2) 简单清洗
        mc = total_mv.replace([np.inf, -np.inf], np.nan)
        return mc


FACTOR_CONFIG = {
    "factor_name": "MC",
    "creator": "zhangyudong",
    "create_date": "2026.02.25",
    "formula": "总市值",
    "description": "公司总市值规模因子（日频）",
    "lookback_days": 1,
    "method": "MC_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "规模因子",
}