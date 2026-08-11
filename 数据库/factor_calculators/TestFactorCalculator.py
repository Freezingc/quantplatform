from .base import FactorCalculator
from factor_calculators.Alpha15Calculator import Alpha15Calculator

class TestFactorCalculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "pct_chg"]  # 指定该因子需要的原始因子
    
    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)

        alpha15calculator = Alpha15Calculator(self.s, 1) # 这里的第二个参数是该因子的lookback_days，需要正确填写
        alpha15 = alpha15calculator.calculate(start_time, end_time)

        # 2. 提取所需因子
        close = data.xs('close',level=0)
        pct_chg = data.xs('pct_chg',level=0)
        # 3. 计算市场平均收益率
        market = close.mean(axis=1).pct_change() * 100

        # 4. 计算惊恐度Terri
        bias = abs(pct_chg.sub(market, axis=0))
        bench = abs(pct_chg).add(abs(market), axis=0) + 0.1
        terri = bias / bench

        # 5. 计算原始惊恐
        terri_ret = (terri * pct_chg).rolling(self.lookback_days).mean().iloc[self.lookback_days:]  # 惊恐收益
        terri_std = (terri * pct_chg).rolling(self.lookback_days).std().iloc[self.lookback_days:]  # 惊恐波动
        terri_ori = terri_ret + terri_std  # 原始惊恐

        return terri_ori
    
'''
    {
        "factor_name": "TestFactor",
        "creator": "chenxutian",
        "create_date": "2025.08.27",
        "formula": "test formula",
        "description": "这是一个用来测试pipeline的因子",
        "lookback_days": 5,
        "method": "TestFactorCalculator",
        "needed_factor": [],
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    },
'''