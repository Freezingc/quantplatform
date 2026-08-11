
# 添加因子信息
FACTOR_CONFIG = [
{
    "factor_name": "PeterLynch_IdioCap_21std",
    "creator": "zhangyudong",
    "create_date": "2026.05.24",
    "formula": "Cross-sectional OLS: log(market_cap) ~ log_cf_pos + log_cf_neg + log_rev_pos + log_rev_neg + log_non_cur_liab + debt_to_assets + log_inventory; residual; ts_std(residual, 21)",
    "description": "彼得林奇特质市值：用彼得林奇偏好基本面变量（净现金流、营业收入正负拆分、非流动负债、资产负债率、存货）做截面回归取残差，再取21日滚动标准差。参考公众号文章《用彼得林奇的思想改进特异市值，IC居然真的能提升？》",
    "lookback_days": 21,
    "method": "PeterLynch_IdioCap_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子",
},
]



