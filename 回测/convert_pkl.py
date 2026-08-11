import pandas as pd
import os
import re  

# 读取pkl
pkl_path = '/data/zhangyudong/FDU/Quant-Platform-Prod-new/模型/data/result/STGATim/alpha1_None_seq30_horizon5_train_begin_date2015-01-02_train_end_date2025-01-01.pkl'
df = pd.read_pickle(pkl_path)

# 新增：转换instrument格式（SH600000 → 600000.SH）
def convert_ts_code(code):
    if pd.isna(code):
        return code
    code = str(code).strip()
    # 匹配 SH/SZ + 数字 → 数字.SH/SZ
    match_sh = re.match(r'(SH)(\d+)', code)
    match_sz = re.match(r'(SZ)(\d+)', code)
    if match_sh:
        return match_sh.group(2) + '.SH'
    if match_sz:
        return match_sz.group(2) + '.SZ'
    # 如果已是标准格式（如000001.SZ），保持
    if '.' in code:
        return code
    return code  # 其他保持

df['instrument'] = df['instrument'].apply(convert_ts_code)



# 清洗datetime → YYYYMMDD
df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y%m%d')

# Pivot宽表
dta = df.pivot(index='instrument', columns='datetime', values='score').fillna(0)

# 插入ts_code列
dta.insert(0, 'ts_code', dta.index)
dta = dta.reset_index(drop=True)

# 保存
name = 'STGATim_stockes'
os.makedirs('input', exist_ok=True)
csv_path = f'input/{name}.csv'
dta.to_csv(csv_path, index=False)
print(f"转换完成：{csv_path}")
print(f"形状：{dta.shape}")
print("前5行ts_code预览：")
print(dta['ts_code'].head())
print("检查'SH600000'是否转换：", '600000.SH' in dta['ts_code'].values)