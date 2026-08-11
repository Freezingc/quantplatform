from .base import FactorCalculator


class ExprDemoCalculator(FactorCalculator):
    """
    示例因子：演示如何仅用表达式就完成计算。

    """

    expression = "RANK(DELTA($vwap, 5)) - RANK(DELTA($open, 5))"
