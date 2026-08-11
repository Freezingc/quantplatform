from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha10Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)

        # 3. 计算收益率
        ret = close.pct_change()
        # 4. 计算20日收益率标准差
        std_ret = ret.rolling(20).std()
        # 5. 实现条件逻辑: RET < 0 时取STD(RET,20)，否则取CLOSE
        condition = (ret < 0)
        intermediate = np.where(condition, std_ret, close)
        
        # 6. 平方运算
        squared = np.power(intermediate, 2)
        
        # 7. 计算5日最大值
        max_5 = pd.DataFrame(squared).rolling(5).max()
        # 8. 计算横截面排名
        def rank(df):
            return df.rank(axis=1, pct=True)
        
        factor = max_5.groupby(max_5.index).apply(rank)
        
        # 确保factor的格式与输入数据一致
        factor = pd.DataFrame(factor.values, index=close.index, columns=close.columns).fillna(0)
        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha10",
    "creator": "chenxutian",
    "create_date": "2025.08.21",
    "formula": "(RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2),5))",
    "description": "alpha191因子10",
    "lookback_days": 23,
    "method": "Alpha10Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''