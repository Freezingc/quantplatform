from .base import FactorCalculator, lag_by_quarters
import numpy as np
import pandas as pd

from .base import FactorCalculator
import numpy as np
import pandas as pd

class ROE_YOY_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_financial_factors = ["roe"]

    def _calculate(self, start_time, end_time):
        # 1. 取数
        financial_data = self.fetch_financial_data(start_time, end_time)
        roe = financial_data.xs("roe", level=0)

        # ===============================
        # 2. 去年同期 ROE（交易日偏移法）
        # ===============================
        roe_lag = lag_by_quarters(roe, 4)

        # ===============================
        # 3. ROE 同比（日频近似）
        # ===============================
        roe_yoy = (roe - roe_lag) / roe_lag.replace(0, np.nan)

        roe_yoy = roe_yoy.replace([np.inf, -np.inf], np.nan)
        return roe_yoy


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "ROE_YOY",
    "creator": "zhangyudong",
    "create_date": "2026.01.04",
    "formula": "(当期ROE TTM - 去年同期ROE TTM) / 去年同期ROE TTM",
    "description": "净资产收益率同比增长率，衡量企业盈利能力的增长速度。正值表示盈利能力提升，负值表示盈利能力下降",
    "lookback_days": 400,
    "method": "ROE_YOY_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子"
}

# 说明：
# 1. 数据来源：
#    - roe: 财务指标表 financial_metrics_table，净资产收益率（TTM）
#    - ROE = 净利润 / 净资产
#
# 2. 计算逻辑：
#    - 去年同期 ROE = roe_{t-4} (4个季度前)
#    - ROE_YOY = (roe_t - roe_{t-4}) / roe_{t-4}
#
# 3. 因子意义：
#    - 正值：ROE 同比增长，盈利能力改善
#      * >50%：盈利能力大幅提升
#      * 20%-50%：盈利能力显著改善
#      * 0-20%：盈利能力小幅提升
#    - 负值：ROE 同比下降，盈利能力恶化
#    - 接近0：ROE 保持稳定
#
# 4. 应用场景：
#    - 识别盈利能力快速改善的企业
#    - 评估企业经营质量的变化趋势
#    - 与 ROE 绝对值结合使用：
#      * 高 ROE + 高 ROE_YOY：优质成长股
#      * 低 ROE + 高 ROE_YOY：困境反转股
#      * 高 ROE + 低 ROE_YOY：成熟稳定股
#      * 低 ROE + 低 ROE_YOY：避免投资
#
# 5. 注意事项：
#    - 需要至少4个季度的历史数据才能计算
#    - ROE 基数效应：低 ROE 基数下的高增速可能不具有持续性
#    - 非经常性损益、并购等会导致 ROE 大幅波动
#    - 负 ROE 的增速解读需谨慎（如从 -5% 到 -2% 是改善还是恶化？）
'''

