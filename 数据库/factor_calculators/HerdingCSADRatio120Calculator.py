from .HerdingCSADCalculatorBase import HerdingCSADCalculatorBase
from .HerdingCSADCore import RATIO_120


class HerdingCSADRatio120Calculator(HerdingCSADCalculatorBase):
    """120日 CSAD 波动率与中心股票收益波动率之比。"""

    output_name = RATIO_120

