from .base import FactorCalculator, lag_by_quarters
import numpy as np
import pandas as pd

class TA_YOY_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_balance_sheet_factors = [
            "total_cur_assets",
            "total_nca",
        ]

    def _calculate(self, start_time, end_time):
        # 1. 取数
        balance_sheet_data = self.fetch_balance_sheet_data(start_time, end_time)

        total_cur_assets = balance_sheet_data.xs("total_cur_assets", level=0)
        total_nca = balance_sheet_data.xs("total_nca", level=0)

        # ===============================
        # 2. 总资产（日频展示值）
        # ===============================
        total_assets = total_cur_assets + total_nca

        # ===============================
        # 3. 去年同期总资产（交易日偏移法）
        # ===============================
        total_assets_lag = lag_by_quarters(total_assets, 4)

        # ===============================
        # 4. 总资产同比（日频近似）
        # ===============================
        ta_yoy = (total_assets - total_assets_lag) / total_assets_lag.replace(0, np.nan)

        ta_yoy = ta_yoy.replace([np.inf, -np.inf], np.nan)
        return ta_yoy

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "TA_YOY",
    "creator": "zhangyudong",
    "create_date": "2026.01.04",
    "formula": "(当期总资产 - 去年同期总资产) / 去年同期总资产",
    "description": "总资产同比增长率，衡量企业资产规模的扩张速度。正值表示资产增长，负值表示资产萎缩",
    "lookback_days": 400,
    "method": "TA_YOY_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子"
}

# 说明：
# 1. 数据来源：
#    - total_cur_assets: 资产负债表 balance_sheet，流动资产合计
#    - total_nca: 资产负债表 balance_sheet，非流动资产合计
#    - 总资产 = 流动资产 + 非流动资产
#
# 2. 计算逻辑：
#    - 去年同期总资产 = total_assets_{t-4} (4个季度前)
#    - TA_YOY = (total_assets_t - total_assets_{t-4}) / total_assets_{t-4}
#
# 3. 因子意义：
#    - 正值：总资产同比增长，企业规模扩张
#      * >20%：快速扩张期
#      * 10%-20%：稳健增长
#      * 0-10%：缓慢增长
#    - 负值：总资产同比下降，企业规模收缩
#      * 可能是主动收缩、剥离资产或经营困难
#    - 接近0：资产规模保持稳定
#
# 4. 应用场景：
#    - 识别处于快速扩张期的企业
#    - 评估企业成长性
#    - 与营收增速、利润增速结合使用，判断扩张质量
#    - 过高的资产增速需警惕：可能存在并购、盲目扩张风险
#
# 5. 注意事项：
#    - 需要至少4个季度的历史数据才能计算
#    - 并购重组会导致资产大幅波动
#    - 需结合资产周转率、ROA等指标评估资产扩张效率
'''

