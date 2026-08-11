from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha157Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        close = data.xs('close', level=0)
        
        # 3. 因子计算
        
        # 先计算第一部分: MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1 * RANK(DELTA((CLOSE - 1), 5))))), 2), 1)))), 1), 5)
        
        # 3.1 计算 DELTA((CLOSE - 1), 5)
        close_minus_1 = close - 1
        delta_close = close_minus_1 - close_minus_1.shift(5)
        
        # 3.2 计算 RANK(DELTA(...))
        rank_delta = delta_close.rank(axis=1, pct=True)
        
        # 3.3 计算 -1 * RANK(...)
        neg_rank = -1 * rank_delta
        
        # 3.4 计算 RANK(RANK(...)) 
        rank_neg_rank = neg_rank.rank(axis=1, pct=True)
        rank_rank_neg_rank = rank_neg_rank.rank(axis=1, pct=True)
        
        # 3.5 计算 TSMIN(RANK(RANK(...)), 2)
        # TSMIN通常表示时间序列上的最小值，这里可能是rolling最小值
        tsmin_rank = rank_rank_neg_rank.rolling(window=2, min_periods=1).min()
        
        # 3.6 计算 SUM(TSMIN(...), 1) - 这里SUM可能是rolling sum
        sum_tsmin = tsmin_rank.rolling(window=1, min_periods=1).sum()
        
        # 3.7 计算 LOG(SUM(...)) - 添加小值防止log(0)
        log_sum = np.log(sum_tsmin + 1e-10)
        
        # 3.8 计算 RANK(RANK(LOG(...)))
        rank_log = log_sum.rank(axis=1, pct=True)
        rank_rank_log = rank_log.rank(axis=1, pct=True)
        
        # 3.9 计算 PROD(RANK(RANK(...)), 1) - 这里PROD可能是rolling product
        # 但由于window=1，prod的结果就是自身
        prod_rank = rank_rank_log.rolling(window=1, min_periods=1).apply(np.prod, raw=True)
        
        # 3.10 计算 MIN(PROD(...), 5) - 这里可能是rolling min
        min_prod = prod_rank.rolling(window=5, min_periods=1).min()
        
        # 第一部分结果
        part1 = min_prod
        
        # 4. 计算第二部分: TSRANK(DELAY((-1 * RET), 6), 5)
        
        # 4.1 计算RET (收益率)
        ret = close.pct_change()
        
        # 4.2 计算 -1 * RET
        neg_ret = -1 * ret
        
        # 4.3 计算 DELAY((-1 * RET), 6)
        delay_neg_ret = neg_ret.shift(6)
        
        # 4.4 计算 TSRANK(DELAY(...), 5) - 时间序列排名
        def tsrank(series, window=5):
            """计算时间序列排名（百分比）"""
            if len(series) < window:
                return pd.Series(index=series.index, data=np.nan)
            
            result = pd.Series(index=series.index, dtype=float)
            
            for i in range(window-1, len(series)):
                # 获取当前窗口的数据
                window_data = series.iloc[i-window+1:i+1]
                # 计算当前值在窗口中的百分位排名
                if window_data.notna().all():
                    # 对窗口内的值进行排名
                    ranks = window_data.rank(pct=True)
                    # 取最后一个值的排名（当前值）
                    result.iloc[i] = ranks.iloc[-1]
                else:
                    result.iloc[i] = np.nan
            
            return result
        
        # 对每个股票计算TSRANK
        tsrank_results = pd.DataFrame(index=delay_neg_ret.index, columns=delay_neg_ret.columns)
        
        for col in delay_neg_ret.columns:
            tsrank_results[col] = tsrank(delay_neg_ret[col], window=5)
        
        part2 = tsrank_results
        
        # 5. 合并两部分
        factor = part1 + part2
        
        # 6. 确保factor的格式与输入数据一致
        factor = factor.reindex(index=close.index, columns=close.columns)
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha157",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1 * RANK(DELTA((CLOSE - 1), 5))))), 2), 1)))), 1), 5) + TSRANK(DELAY((-1 * RET), 6), 5))",
        "description": "前半部分捕捉5日价格变化的复杂非线性关系，后半部分评估过去6天负收益率在近5天内的相对表现",
        "lookback_days": 16,
        "method": "Alpha157Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
