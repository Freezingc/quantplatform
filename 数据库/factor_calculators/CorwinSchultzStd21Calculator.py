from .CorwinSchultzCalculatorBase import CorwinSchultzCalculatorBase


class CorwinSchultzStd21Calculator(CorwinSchultzCalculatorBase):
    """日频 CS 价差的过去21个交易日样本标准差。"""

    output_name = "std_21"

