from .CorwinSchultzCalculatorBase import CorwinSchultzCalculatorBase


class CorwinSchultzMean21Calculator(CorwinSchultzCalculatorBase):
    """日频 CS 价差的过去21个交易日均值。"""

    output_name = "mean_21"

