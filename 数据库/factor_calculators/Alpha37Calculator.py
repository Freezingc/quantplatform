from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha37Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "open"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        open = data.xs('open', level=0)  # 获取开盘价数据

        # 3. 计算收益率RET (使用收盘价计算日收益率)
        ret = close.pct_change()

        # 4. 计算SUM(OPEN, 5) - 5日开盘价之和
        sum_open_5 = open.rolling(window=5).sum()

        # 5. 计算SUM(RET, 5) - 5日收益率之和
        sum_ret_5 = ret.rolling(window=5).sum()

        # 6. 计算SUM(OPEN, 5) * SUM(RET, 5)
        product = sum_open_5 * sum_ret_5

        # 7. 计算DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10) - 滞后10期
        delayed_product = product.shift(10)

        # 8. 计算差值
        diff = product - delayed_product

        # 9. 计算排名（横截面排名，使用rank(axis=1)）
        # 对每行（每个时间点）的所有股票进行排名
        rank_diff = diff.rank(axis=1, pct=False)

        # 10. 乘以-1
        factor = -1 * rank_diff

        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha37",
    "creator": "chenxutian",
    "create_date": "2025.08.26",
    "formula": "(-1 * RANK(((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10))))",
    "description": "alpha191因子37",
    "lookback_days": 15,
    "method": "Alpha37Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''