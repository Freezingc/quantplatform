import pandas as pd
import dolphindb as ddb
import numpy as np
import os
import time

class DataFetcher:
    def __init__(self, session):
        self.session = session
        self.db_name = None
        self.table_name = None
        self.date_name = None

    def _get_store_table_name(self, factor_name):
        query = f'''
        db = database("dfs://YesLab")
        t = loadTable(db, "factor_metadata")
        select store_factor from t where factor_name = '{factor_name}'
        '''
        result = self.session.run(query)
        if result is None or result.empty:
            raise ValueError(f"因子 {factor_name} 在 factor_metadata 表中不存在。")
        return result.iloc[0, 0]

    def __get_time_range(self, date_name, is_start=True):
        """获取最早或最晚时间戳"""
        time_field = f"min({date_name})" if is_start else f"max({date_name})"
        query = f'''
        db = database("{self.db_name}")
        t = loadTable(db, "{self.table_name}")
        select {time_field} from t
        '''
        result = self.session.run(query)
        date_value = result.iloc[0, 0]
        date_dt = pd.to_datetime(date_value)
        return date_dt.strftime('%Y.%m.%d')

    def __set_db_and_table(self, db_name, table_name):
        """设置数据库和表名"""
        if not db_name or not table_name:
            raise ValueError("Database name and table name must be provided.")
        self.db_name = db_name
        self.table_name = table_name

    def get_factor_data(self, factor_name: str, start_time="", end_time=""):
        """获取因子数据"""
        self.__set_db_and_table("dfs://YesLab", "factor_daily_data")
        store_table = self._get_store_table_name(factor_name)
        
        # 获取时间范围
        if not start_time:
            start_time = self.__get_time_range("trade_date", is_start=True)
        if not end_time:
            end_time = self.__get_time_range("trade_date", is_start=False)

        query = f'''
        db = database("{self.db_name}")
        t = loadTable(db, "{store_table}")
        select ts_code, trade_date, value as "{factor_name}" from t
        where factor_name == "{factor_name}"
            and trade_date between date(`{start_time}) and date(`{end_time})
        '''
        result = self.session.run(query)
        result = result.drop_duplicates(keep='first')
        if result is None or result.empty:
            raise ValueError(f"因子 {factor_name} 查询结果为空，请检查数据表 {store_table} 或时间区间。")
        if result[factor_name].notna().sum() == 0:
            raise ValueError(f"因子 {factor_name} 查询结果全为空值，请检查数据表 {store_table}。")

        return result
    
    def get_all_factor_names(self):
        """获取所有因子名称"""
        self.__set_db_and_table("dfs://YesLab", "factor_daily_data")
        query = f'''
        db = database("{self.db_name}")
        t = loadTable(db, "{self.table_name}")
        select distinct factor_name from t
        '''
        result = self.session.run(query)
        return result.iloc[:, 0].tolist()
    
    def save_all_factors_to_directory(self, output_dir, start_time="", end_time="", factor_names = []):
        """
        获取所有因子数据并以长表格式保存到指定目录
        """
        start_timestamp = time.time()
        os.makedirs(output_dir, exist_ok=True)
        
        all_factors = factor_names or self.get_all_factor_names()
        total_factors = len(all_factors)
        
        print(f"开始处理 {total_factors} 个因子...")
        
        for i, factor_name in enumerate(all_factors, 1):
            try:
                factor_data = self.get_factor_data(factor_name, start_time, end_time)
                if factor_data.empty:
                    raise ValueError(f"因子 {factor_name} 导出结果为空。")
                
                file_path = os.path.join(output_dir, f"factor_data_{factor_name}.csv")
                # factor_data.to_csv(file_path, index=False)
                factor_data.to_parquet(os.path.join(output_dir, f"factor_data_{factor_name}.parquet"), index=False)
                
                # 打印进度
                print(f"[{i}/{total_factors}] 已保存因子 {factor_name} 到 {file_path}, rows={len(factor_data)}, non_null={int(factor_data[factor_name].notna().sum())}")
                
            except Exception as e:
                print(f"处理因子 {factor_name} 时出错: {e}")

        print(f"所有因子数据已保存到 {output_dir}, 耗时 {time.time() - start_timestamp:.2f} 秒")


if __name__ == "__main__":
    # 配置信息
    import argparse
    username = "jiangyushi"
    password = "123456"
    host = "10.176.55.211"
    port = 8848

    # 输出目录
    output_directory = "test_data_files"
    # 指定因子名，获取对应数据（不指定则获取全部）
    parser = argparse.ArgumentParser(description="处理数据文件")
    parser.add_argument('--start_time', type=str, required=False, help='YYYY.MM.DD', default="2021.01.01")
    parser.add_argument('--end_time', type=str, required=False, help='YYYY.MM.DD', default="2026.03.01")
    parser.add_argument('--label_len', type=int, required=False, default=5)

    args = parser.parse_args()

    start_time = args.start_time
    end_time = args.end_time
    label_len = args.label_len

    with open('feature_columns.txt', 'r') as f:
        factor_names = [line.strip() for line in f]  # strip() 去除每行末尾的换行符 \n
    
    # 创建会话并连接
    session = ddb.session()
    try:
        session.connect(host, port, username, password)
        fetcher = DataFetcher(session)
        fetcher.save_all_factors_to_directory(output_directory, start_time, end_time, factor_names)
        
    except ValueError as e:
        print(f"参数错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 确保会话关闭
        session.close()

    #获取开盘价并生成标签
    import dolphindb as ddb
    import pandas as pd
    
    s = ddb.session()
    s.connect("10.176.55.211", 8848, "liuziyang", "123456")
    
    query = f'''
    db = database("dfs://YesLab")
    t1 = loadTable(db, "stock_factor_daily")
    select trade_date, ts_code, open_qfq from t1 
    where trade_date between {start_time} and {end_time}
    '''
    df = s.run(query)  
    df = df.sort_values(['trade_date','ts_code'])
    
    def add_forward_return(group):
        group = group.copy()
        group['open_t1'] = group['open_qfq'].shift(-1)
        group['open_t6'] = group['open_qfq'].shift(-(label_len+1))
        group['label'] = group['open_t6'] / group['open_t1'] - 1
        return group
    
    df = df.groupby('ts_code', group_keys=False).apply(add_forward_return)
    df[['trade_date', 'ts_code', 'label']].to_parquet(f'{output_directory}/label.parquet', index=False)

    import pyarrow.parquet as pq
    from pathlib import Path
    import pandas as pd

    folder_path = Path(f'{output_directory}')
    parquet_files = sorted(folder_path.glob("*.parquet"))

    factor_series = []

    for file in parquet_files:
        factor_name = file.stem
        print(f"Processing: {factor_name}...")
    
        df = pq.read_table(str(file)).to_pandas()
    
        value_col = [col for col in df.columns if col not in ['ts_code', 'trade_date']]
        if len(value_col) != 1:
            raise ValueError(f"{file.name} should have one value column, got {value_col}")
        value_col = value_col[0]
        if factor_name != 'label' and df[value_col].notna().sum() == 0:
            raise ValueError(f"{file.name} 的因子值全为空，停止合并，请先修复取数脚本。")
    
        series = df.set_index(['ts_code', 'trade_date'])[value_col]
        series.name = factor_name  
        factor_series.append(series)
    
    merged = pd.concat(factor_series, axis=1) 
    non_label_cols = [col for col in merged.columns if col != 'label']
    if len(non_label_cols) == 0:
        raise ValueError("合并结果中没有任何特征列。")
    if merged[non_label_cols].notna().sum().sum() == 0:
        raise ValueError("合并结果的全部特征值均为空，请检查因子取数逻辑。")
    
    merged = merged.reset_index().sort_values(by=['trade_date', 'ts_code'])
    
    merged.to_parquet("test_data.parquet", engine='pyarrow', compression='snappy')
        
# nohup python -u scripts/get_all_factor.py > get_all_factor.log 2>&1 &