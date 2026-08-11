from .base import FactorCalculator, cumulative_to_single_quarter, lag_by_quarters
import numpy as np
import pandas as pd

class NP_Q_YOY_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_income_factors = ["n_income_attr_p"]

    def _calculate(self, start_time, end_time):
        # 1. 取数
        income_data = self.fetch_income_data(start_time, end_time)
        n_income_cumsum = income_data.xs("n_income_attr_p", level=0)

        # ===============================
        # 2. 单季度净利润（日频近似）
        # ===============================
        n_income_quarter = cumulative_to_single_quarter(n_income_cumsum)

        # ===============================
        # 3. 去年同季度（交易日偏移法）
        # ===============================
        n_income_quarter_lag = lag_by_quarters(n_income_quarter, 4)

        # ===============================
        # 4. 单季度同比（日频近似）
        # ===============================
        np_q_yoy = (
            n_income_quarter - n_income_quarter_lag
        ) / n_income_quarter_lag.replace(0, np.nan)

        np_q_yoy = np_q_yoy.replace([np.inf, -np.inf], np.nan)
        return np_q_yoy

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "NP_Q_YOY",
    "creator": "zhangyudong",
    "create_date": "2026.01.04",
    "formula": "(本季度净利润 - 去年同季度净利润) / 去年同季度净利润",
    "description": "净利润单季度同比增长率，衡量企业单季度盈利能力的增长速度。正值表示盈利增长，负值表示盈利下降",
    "lookback_days": 400,
    "method": "NP_Q_YOY_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子"
}

# 说明：
# 1. 数据来源：
#    - n_income_attr_p: 利润表 company_income_table，归属于母公司所有者的净利润（累计值）
#
# 2. 计算逻辑：
#    - 财报中的净利润是累计值（年初至今）
#    - 单季度净利润 = 当期累计净利润 - 上期累计净利润
#      * Q1: 直接用累计值
#      * Q2: Q2累计 - Q1累计
#      * Q3: Q3累计 - Q2累计
#      * Q4: Q4累计 - Q3累计（或直接用年报全年值）
#    - 去年同季度净利润 = 单季度净利润_{t-4}
#    - NP_Q_YOY = (单季度净利润_t - 单季度净利润_{t-4}) / 单季度净利润_{t-4}
#
# 3. 因子意义：
#    - 正值：单季度净利润同比增长
#      * >50%：业绩爆发式增长
#      * 20%-50%：业绩快速增长
#      * 0-20%：业绩平稳增长
#    - 负值：单季度净利润同比下降
#    - 接近0：单季度业绩保持稳定
#
# 4. 应用场景：
#    - 捕捉企业最新的盈利变化趋势（相比TTM更及时）
#    - 识别业绩拐点：
#      * 连续多个季度正增长：业绩上行周期
#      * 由负转正：困境反转
#      * 由正转负：业绩见顶
#    - 季度业绩超预期/低于预期分析
#
# 5. 与其他因子对比：
#    - NP_Q_YOY（单季度同比）：反映最新季度变化，更及时
#    - NP_TTM_YOY（TTM同比）：平滑季节性波动，更稳定
#
# 6. 注意事项：
#    - 需要至少5个季度的历史数据（计算单季度+同比）
#    - 单季度数据波动较大，受季节性因素影响
#    - 去年同期为负、本期为正时，增速会异常（需谨慎解读）
#    - 非经常性损益会导致单季度业绩大幅波动
'''

