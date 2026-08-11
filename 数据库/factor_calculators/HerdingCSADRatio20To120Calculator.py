from .HerdingCSADCalculatorBase import HerdingCSADCalculatorBase
from .HerdingCSADCore import RATIO_20_TO_120


class HerdingCSADRatio20To120Calculator(HerdingCSADCalculatorBase):
    """20日与120日 CSAD/收益波动率比值的短长周期比。"""

    output_name = RATIO_20_TO_120

