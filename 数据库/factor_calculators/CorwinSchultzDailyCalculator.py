from .CorwinSchultzCalculatorBase import CorwinSchultzCalculatorBase


class CorwinSchultzDailyCalculator(CorwinSchultzCalculatorBase):
    """基于相邻两个交易日最高价和最低价的原始 CS 价差。"""

    output_name = "daily"

