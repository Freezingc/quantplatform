from .base import FactorCalculator
import numpy as np


class Ln_FC_Calculator(FactorCalculator):
    """
    流通市值对数（Ln_FC）因子：
    Ln_FC = ln(流通市值)

    说明：
    - 使用自然对数
    - 对非正值做NaN处理
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

        # 2) 取对数（仅对正值）
        ln_fc = np.log(circ_mv.where(circ_mv > 0))

        ln_fc = ln_fc.replace([np.inf, -np.inf], np.nan)
        return ln_fc


FACTOR_CONFIG = {
    "factor_name": "Ln_FC",
    "creator": "zhangyudong",
    "create_date": "2026.02.25",
    "formula": "ln(流通市值)",
    "description": "流通市值对数（规模因子常用变换）",
    "lookback_days": 1,
    "method": "Ln_FC_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "规模因子",
}