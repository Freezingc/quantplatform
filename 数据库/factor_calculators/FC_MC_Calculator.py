from .base import FactorCalculator
import numpy as np


class FC_MC_Calculator(FactorCalculator):
    """
    流通市值比总市值（FC_MC）因子：
    FC_MC = 流通市值 / 总市值
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)

        # 行情字段
        self.required_factors = [
            "circ_mv",   # 流通市值
            "total_mv"    # 总市值
        ]

    def _calculate(self, start_time, end_time):
        # 1) 取数
        daily_data = self.fetch_data(start_time, end_time)

        circ_mv = daily_data.xs("circ_mv", level=0)
        total_mv = daily_data.xs("total_mv", level=0)

        # 2) 计算比值
        fc_mc = circ_mv / total_mv.replace(0, np.nan)

        fc_mc = fc_mc.replace([np.inf, -np.inf], np.nan)
        return fc_mc


FACTOR_CONFIG = {
    "factor_name": "FC_MC",
    "creator": "zhangyudong",
    "create_date": "2026.02.25",
    "formula": "流通市值 / 总市值",
    "description": "流通市值占比，衡量公司股份流通程度",
    "lookback_days": 1,
    "method": "FC_MC_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "结构因子",
}