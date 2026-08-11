import dolphindb as ddb
import pandas as pd

s = ddb.session()
s.connect("10.176.55.211", 8848, "liuziyang", "123456")

query = f'''
    db = database("dfs://YesLab")
    t1 = loadTable(db, "factor_summary_full")
    select factor_name from t1 where IC均值 >= 0.025 or IC均值 <= -0.025
    '''
result = s.run(query)
with open('feature_columns.txt', 'w') as f:
    for col in result.iloc[:, 0].tolist():
        f.write(col + '\n')