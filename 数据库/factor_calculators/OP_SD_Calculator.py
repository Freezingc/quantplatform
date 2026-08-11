from .base import FactorCalculator, lag_by_quarters
import numpy as np

class OP_SD_Calculator(FactorCalculator):
    """
    营业利润稳健加速度（OP_SD）因子：
    OP_SD = 营业利润稳健增速的一阶差分
    即计算营业利润环比增速的差分（环比增速的变化）

    计算公式：
    1. 首先计算营业利润环比增速（OP_QOQ）。
    2. 然后计算其一阶差分，得到营业利润稳健加速度（OP_SD）。
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)

        # 所需因子：营业利润
        self.required_income_factors = [
            "operate_profit"  # 营业利润（利润表中的数据）
        ]

    def _calculate(self, start_time, end_time):
        # 1) 取数：获取营业利润数据
        income_data = self.fetch_income_data(start_time, end_time)

        operate_profit = income_data.xs("operate_profit", level=0)  # 本季度营业利润

        # 2) 计算上季度营业利润
        operate_profit_lag = lag_by_quarters(operate_profit, 1)  # 上季度营业利润

        # 3) 计算营业利润增速（环比增速）
        op_qoq = (operate_profit - operate_profit_lag) / operate_profit_lag.replace(0, np.nan)

        # 4) 计算营业利润增速的一阶差分（稳健加速度）
        op_sd = op_qoq.diff(periods=1)  # 差分计算

        op_sd = op_sd.replace([np.inf, -np.inf], np.nan)  # 替换无效值
        return op_sd


FACTOR_CONFIG = {
    "factor_name": "OP_SD",
    "creator": "zhangyudong",
    "create_date": "2026.02.25",
    "formula": "营业利润稳健增速的一阶差分",
    "description": "营业利润稳健加速度，衡量营业利润增速变化的敏感度",
    "lookback_days": 750,
    "method": "OP_SD_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "盈利因子",
}