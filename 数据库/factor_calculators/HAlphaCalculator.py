from .base import FactorCalculator
import numpy as np
import pandas as pd
import statsmodels.api as sm

class HAlphaCalculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close_hfq"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close_hfq = data.xs('close_hfq', level=0)
        
        # 3. 计算
        ret = close_hfq.pct_change()
        mkt_ret = close_hfq.mean(axis=1).pct_change()

        # 4. 计算滚动均值、协方差、方差
        window = 60
        mean_stock = ret.rolling(window).mean()
        mean_mkt = mkt_ret.rolling(window).mean()

        cov = ret.rolling(window).cov(mkt_ret)
        var_mkt = mkt_ret.rolling(window).var()

        beta = cov.div(var_mkt, axis=0)

        halpha = mean_stock.sub(beta.mul(mean_mkt, axis=0), axis=0)
        
        return halpha

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "halpha",
    "creator": "jiangyushi",
    "create_date": "2025.10.23",
    "formula": "halpha = MEAN(daily_ret, 60) - COV(daily_ret, mkt_ret, 60) / VAR(mkt_ret, 60) * MEAN(mkt_ret, 60)",
    "description": "个股 60 个日收益与市场收益线性回归的截矩项",
    "lookback_days": 60,
    "method": "HAlphaCalculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    "category": "动量反转因子"
}
'''
