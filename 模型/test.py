import pandas as pd; 
# df = pd.read_parquet('/data/zhangyudong/FDU/Quant-Platform-Prod-new/模型/test_data.parquet'); 
# print(df.tail(30))

import pandas as pd; 
df = pd.read_parquet('test_data_files/label.parquet'); 
df.to_csv('test_data_files/label.csv', index=False);
print(f'转换完成! 文件已保存为 test_data_files/label.csv, 共 {len(df)} 行数据')