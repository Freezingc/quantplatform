from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha186Calculator(FactorCalculator):
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
        
        # 3. 重新解读公式中的LD和HD
        # 从公式看：SUM((LD>0 & LD>HD)?LD:0,14)
        # 这似乎是一种动量或相对强度指标
        
        # 3.1 计算TR（真实波幅）
        hl = high - low
        hc = abs(high - close.shift(1))
        lc = abs(low - close.shift(1))
        tr = pd.DataFrame(np.maximum(np.maximum(hl.values, hc.values), lc.values),
                        index=low.index, columns=low.columns)
        
        # 3.2 重新理解LD和HD
        # 可能LD表示"上涨动量"，HD表示"下跌动量"
        # 尝试使用收盘价变化来计算动量
        
        # 计算价格变化
        price_change = close - close.shift(1)
        
        # LD: 上涨动量（如果价格上涨且价格>0）
        ld_up = pd.DataFrame(np.where((price_change > 0) & (price_change > 0), 
                                    price_change, 0),  # 第二个条件可能是price_change > 某种阈值
                            index=low.index, columns=low.columns)
        
        # HD: 下跌动量（如果价格下跌且abs(价格变化)>0）
        hd_down = pd.DataFrame(np.where((price_change < 0) & (abs(price_change) > 0), 
                                    abs(price_change), 0),
                            index=low.index, columns=low.columns)
        
        # 3.3 计算14日滚动和
        sum_ld_14 = ld_up.rolling(window=14, min_periods=14).sum()
        sum_hd_14 = hd_down.rolling(window=14, min_periods=14).sum()
        sum_tr_14 = tr.rolling(window=14, min_periods=14).sum()
        
        # 避免除零错误
        epsilon = 1e-10
        
        # 3.4 计算百分比（确保分母不为零）
        ld_percent = sum_ld_14 * 100 / (sum_tr_14 + epsilon)
        hd_percent = sum_hd_14 * 100 / (sum_tr_14 + epsilon)
        
        # 3.5 计算绝对差值
        abs_diff = abs(ld_percent - hd_percent)
        
        # 3.6 计算总和百分比
        sum_percent = ld_percent + hd_percent
        sum_percent_safe = sum_percent.where(sum_percent != 0, epsilon)
        
        # 3.7 计算比率
        ratio = abs_diff * 100 / sum_percent_safe
        
        # 3.8 计算6日移动平均
        mean_ratio_6 = ratio.rolling(window=6, min_periods=6).mean()
        
        # 3.9 计算延迟6期的值
        delay_mean_ratio_6 = mean_ratio_6.shift(6)
        
        # 3.10 计算最终因子
        factor = (mean_ratio_6 + delay_mean_ratio_6) / 2
        
        # 3.11 处理可能的NaN值（保持原始格式）
        factor = factor.fillna(0)
        
        return factor

'''
# 添加因子信息
    {
        "factor_name": "Alpha186",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)+DELAY(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6),6))/2",
        "description": "基于价格动量分歧的相对波动指标，通过比较特定条件下的高低点差异变化率来衡量市场买卖力量的动态平衡状态",
        "lookback_days": 30,
        "method": "Alpha186Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
