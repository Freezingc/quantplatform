from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha172Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        
        # 确保索引按日期升序排列
        low = low.sort_index()
        high = high.sort_index()
        close = close.sort_index()
        
        # 3. 计算 LD 和 HD
        prev_close = close.shift(1)  # 前一日收盘价
        LD_raw = prev_close - low    # 下跌波动
        HD_raw = high - prev_close   # 上涨波动
        
        # 4. 计算 TR（真实波幅）
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        TR = np.maximum(np.maximum(tr1, tr2), tr3)
        
        # 5. 滚动窗口大小
        window1 = 14  # 第一个求和窗口
        window2 = 6   # 移动平均窗口
        
        # 6. 计算条件
        condition1 = (LD_raw > 0) & (LD_raw > HD_raw)  # LD>0 & LD>HD
        condition2 = (HD_raw > 0) & (HD_raw > LD_raw)  # HD>0 & HD>LD
        
        # 7. 应用条件
        LD_cond = LD_raw.where(condition1, 0)
        HD_cond = HD_raw.where(condition2, 0)
        
        # 8. 计算滚动和
        sum_LD_14 = LD_cond.rolling(window=window1, min_periods=window1).sum()
        sum_HD_14 = HD_cond.rolling(window=window1, min_periods=window1).sum()
        sum_TR_14 = TR.rolling(window=window1, min_periods=window1).sum()
        
        # 9. 计算 a 和 b
        a = sum_LD_14 * 100 / sum_TR_14
        b = sum_HD_14 * 100 / sum_TR_14
        
        # 10. 计算中间值
        denominator = a + b
        value = abs(a - b) / denominator * 100
        value[denominator == 0] = np.nan  # 处理分母为零的情况
        
        # 11. 计算因子（6日移动平均）
        factor = value.rolling(window=window2, min_periods=window2).mean()
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha172",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)",
        "description": "这是一个基于价格相对强度与真实波幅比值的动量因子，通过比较上涨力量与下跌力量在14日窗口内的累积相对强度差异，并进行6日平滑处理。",
        "lookback_days": 20,
        "method": "Alpha172Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
