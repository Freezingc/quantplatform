from .base import FactorCalculator, lag_by_quarters
import numpy as np

class DTED_Calculator(FactorCalculator):
    """
    产权比率变动（DTED）因子：
    DTED = 当期产权比率 - 上期产权比率
    其中产权比率 = 股东权益 / 总资产
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_balance_sheet_factors = ["total_hldr_eqy_inc_min_int", "total_assets"]  # 股东权益和总资产

    def _calculate(self, start_time, end_time):
        # 1) 获取股东权益和总资产数据
        balance_data = self.fetch_balance_sheet_data(start_time, end_time)

        total_hldr_eqy = balance_data.xs("total_hldr_eqy_inc_min_int", level=0)
        total_assets = balance_data.xs("total_assets", level=0)

        # 2) 计算产权比率
        equity_ratio = total_hldr_eqy / total_assets.replace(0, np.nan)  # 防止除以零

        # 3) 计算产权比率变动（DTED）
        equity_ratio_lag = lag_by_quarters(equity_ratio, 1)  # 上期产权比率
        tted = equity_ratio - equity_ratio_lag

        tted = tted.replace([np.inf, -np.inf], np.nan)  # 替换无效值
        return tted

# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "DTED",
    "creator": "zhangyudong",
    "create_date": "2026.01.29",
    "formula": "当期产权比率 - 上期产权比率",
    "description": "产权比率变动，衡量企业财务杠杆的边际变化",
    "lookback_days": 750,
    "method": "DTED_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子",
}
