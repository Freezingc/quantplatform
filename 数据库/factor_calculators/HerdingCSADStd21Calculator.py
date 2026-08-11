from .HerdingCSADCalculatorBase import HerdingCSADCalculatorBase
from .HerdingCSADCore import CSAD_STD_21


class HerdingCSADStd21Calculator(HerdingCSADCalculatorBase):
    """相关股票群组过去21日 CSAD 的样本标准差。"""

    output_name = CSAD_STD_21

