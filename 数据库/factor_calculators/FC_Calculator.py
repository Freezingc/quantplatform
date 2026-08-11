from .base import FactorCalculator
import numpy as np


class FC_Calculator(FactorCalculator):
    """
    FC（流通市值）因子：
    FC = 流通市值（float_mv）

    说明：
    - 直接取日频行情中的流通市值
    - 不做TTM、不做变动
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)

        # 行情字段：流通市值
        self.required_factors = [
            "circ_mv"
        ]

    def _calculate(self, start_time, end_time):
        # 1) 取数
        daily_data = self.fetch_data(start_time, end_time)

        circ_mv = daily_data.xs("circ_mv", level=0)

        # 2) 返回（简单清洗）
        fc = circ_mv.replace([np.inf, -np.inf], np.nan)
        return fc


FACTOR_CONFIG = {
    "factor_name": "FC",
    "creator": "zhangyudong",
    "create_date": "2026.02.25",
    "formula": "流通市值",
    "description": "公司流通市值规模因子（日频）",
    "lookback_days": 1,
    "method": "FC_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "规模因子",
}