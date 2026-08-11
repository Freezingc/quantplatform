from .base import FactorCalculator
import numpy as np


class Ln_MC_Calculator(FactorCalculator):
    """
    总市值对数（Ln_MC）因子：
    Ln_MC = ln(总市值)

    说明：
    - 使用自然对数
    - 对非正值做NaN处理
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

        # 2) 取对数（仅对正值）
        ln_mc = np.log(total_mv.where(total_mv > 0))

        ln_mc = ln_mc.replace([np.inf, -np.inf], np.nan)
        return ln_mc


FACTOR_CONFIG = {
    "factor_name": "Ln_MC",
    "creator": "zhangyudong",
    "create_date": "2026.02.25",
    "formula": "ln(总市值)",
    "description": "总市值对数（规模因子常用变换）",
    "lookback_days": 750,
    "method": "Ln_MC_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "规模因子",
}