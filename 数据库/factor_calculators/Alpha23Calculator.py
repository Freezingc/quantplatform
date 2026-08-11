from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha23Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)

        # 3. 实现因子计算逻辑
        # 计算前一日收盘价
        delay_close = close.shift(1)
        
        # 计算20日收盘价的标准差
        std_close_20 = close.rolling(window=20).std()
        
        # 条件1: CLOSE > DELAY(CLOSE, 1)
        condition1 = close > delay_close
        
        # 条件2: CLOSE <= DELAY(CLOSE, 1)
        condition2 = close <= delay_close
        
        # 计算分子部分: SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
        numerator_data = pd.DataFrame(np.where(condition1, std_close_20, 0), 
                                    index=close.index, columns=close.columns)
        numerator = numerator_data.rolling(window=20, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*19)/20 if len(x) == 20 else x.mean()
        )

        # 计算分母的第一部分: SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
        denominator_part1 = numerator
        
        # 计算分母的第二部分: SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
        denominator_part2_data = pd.DataFrame(np.where(condition2, std_close_20, 0), 
                                            index=close.index, columns=close.columns)

        denominator_part2 = denominator_part2_data.rolling(window=20, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*19)/20 if len(x) == 20 else x.mean()
        )
        
        # 计算完整分母
        denominator = denominator_part1 + denominator_part2
        
        # 避免除零错误
        denominator = denominator.replace(0, np.nan)
        
        # 计算最终因子
        factor = (numerator / denominator) * 100
        
        # 确保factor的格式与输入数据一致
        factor = factor.reindex(index=close.index, columns=close.columns)

        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha23",
    "creator": "chenxutian",
    "create_date": "2025.08.22",
    "formula": "SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE:20),0),20,1)/(SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1 )+SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100",
    "description": "alpha191因子23​",
    "lookback_days": 38,
    "method": "Alpha23Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
