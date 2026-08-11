from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha117Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high","low"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        vol = data.xs('vol', level=0)  # 假设成交量列名为'vol'
        
        # 3. 计算收益率RET
        # 假设RET是日收益率
        ret = close.pct_change()
        
        # 4. 定义TSRANK函数
        def tsrank(series, window):
            """计算时间序列排名（百分比排名）"""
            if series.empty:
                return pd.Series(index=series.index)
            
            # 对每个滚动窗口内的数据进行百分比排名
            return series.rolling(window).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == window else np.nan,
                raw=False
            )
        
        # 5. 计算各个TSRANK分量
        # 5.1 计算TSRANK(VOLUME, 32)
        tsrank_vol = tsrank(vol, 32)
        
        # 5.2 计算中间表达式: (CLOSE + HIGH) - LOW
        close_high_low = (close + high) - low
        
        # 5.3 计算TSRANK(((CLOSE + HIGH) - LOW), 16)
        tsrank_chl = tsrank(close_high_low, 16)
        
        # 5.4 计算TSRANK(RET, 32)
        tsrank_ret = tsrank(ret, 32)
        
        # 6. 组合因子
        # 因子公式: (TSRANK(VOLUME, 32) * (1 - TSRANK(((CLOSE + HIGH) - LOW), 16))) * (1 - TSRANK(RET, 32))
        
        # 6.1 计算第一部分: (1 - TSRANK(((CLOSE + HIGH) - LOW), 16))
        part1 = 1 - tsrank_chl
        
        # 6.2 计算第二部分: (1 - TSRANK(RET, 32))
        part2 = 1 - tsrank_ret
        
        # 6.3 计算最终因子
        factor = tsrank_vol * part1 * part2
        
        # 7. 确保factor的格式与输入数据一致
        # 转换为与low, high, close相同的pivot格式
        factor = factor.where(pd.notnull(factor))
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha117",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "((TSRANK(VOLUME, 32) * (1 - TSRANK(((CLOSE + HIGH) - LOW), 16))) * (1 - TSRANK(RET, 32)))",
        "description": "衡量“在成交量相对高位、价格呈现窄幅整理且近期表现不佳的股票”的特征",
        "lookback_days": 31,
        "method": "Alpha117Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
