from .base import FactorCalculator
import numpy as np
import pandas as pd

class ROA_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        # 直接使用 financial_metrics_table 中已有的 ROA 数据
        self.required_financial_factors = ["roa"]  # 总资产报酬率（已经是TTM数据）

    def _calculate(self, start_time, end_time):
        # 1. 直接获取财务指标表中的 ROA 数据
        # financial_metrics_table 中的 roa 字段已经是计算好的总资产报酬率
        financial_data = self.fetch_financial_data(start_time, end_time)
        
        # 调试：检查数据结构
        if financial_data is None or financial_data.empty:
            print(f"警告：未获取到任何财务数据")
            return pd.DataFrame()
        
        print(f"调试信息 - 列索引: {financial_data.columns[:5]}")  # 打印前5个列
        print(f"调试信息 - 列索引类型: {type(financial_data.columns)}")
        print(f"调试信息 - 数据形状: {financial_data.shape}")
        
        # 2. 提取 ROA 因子
        try:
            roa = financial_data.xs('roa', level=0)
        except KeyError as e:
            print(f"错误：无法找到 'roa' 因子，可用的因子: {financial_data.columns.get_level_values(0).unique().tolist()}")
            raise
        
        # 3. 处理无穷大值和无效值
        roa = roa.replace([np.inf, -np.inf], np.nan)
        
        return roa

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "ROA",
    "creator": "zhangyudong",
    "create_date": "2025.12.17",
    "formula": "直接使用 financial_metrics_table 中的 roa 字段（总资产报酬率）",
    "description": "资产回报率，衡量企业利用全部资产获取利润的能力。数据来源于 Tushare fina_indicator 接口",
    "lookback_days": 1,
    "method": "ROA_Calculator",
    "start_time": "",  # 可以为空，默认从2020.01.01开始
    "end_time": "",  # 可以为空，默认为当日日期
    "category": "基本面因子",
}

# 说明：
# financial_metrics_table 中还有其他相关的 ROA 字段可供选择：
# - roa: 总资产报酬率（EBIT/平均总资产）
# - roa2_yearly: 年化总资产报酬率
# - roa_yearly: 年化总资产净利率（净利润/平均总资产）
# - roa_dp: 总资产净利率(杜邦分析)
# 
# 如需使用其他字段，修改 self.required_financial_factors = ["字段名"] 即可
'''

