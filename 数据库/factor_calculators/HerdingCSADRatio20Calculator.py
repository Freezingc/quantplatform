from .HerdingCSADCalculatorBase import HerdingCSADCalculatorBase
from .HerdingCSADCore import RATIO_20


class HerdingCSADRatio20Calculator(HerdingCSADCalculatorBase):
    """20日 CSAD 波动率与中心股票收益波动率之比。"""

    output_name = RATIO_20

