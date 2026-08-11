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
        
        # 获取时间范围
        if not start_time:
            start_time = self.__get_time_range("trade_date", is_start=True)
        if not end_time:
            end_time = self.__get_time_range("trade_date", is_start=False)

        query = f'''
        db = database("{self.db_name}")
        t = loadTable(db, "{self.table_name}")
        select ts_code, trade_date, value as "{factor_name}" from t
        where factor_name == "{factor_name}"
            and trade_date between date(`{start_time}) and date(`{end_time})
        '''
        result = self.session.run(query)
        result = result.drop_duplicates(keep='first')

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
    
    def get_existing_factor_names(self, output_dir):
        """获取已有的因子名称"""
        existing_factor = []

        if not os.path.exists(output_dir):
            return existing_factor

        for filename in os.listdir(output_dir):
            if filename.startswith("factor_data_") and filename.endswith(".csv"):
                factor_name = filename[len("factor_data_"):-len(".csv")]
                existing_factor.append(factor_name)
        return existing_factor

    def save_all_factors_to_directory(self, output_dir, start_time="", end_time="", factor_names = []):
        """
        获取所有因子数据并以长表格式保存到指定目录
        """
        start_timestamp = time.time()
        os.makedirs(output_dir, exist_ok=True)
        
        all_factors = factor_names or self.get_all_factor_names()
        existing_factors = self.get_existing_factor_names(output_dir)
        all_factors = [factor for factor in all_factors if factor not in existing_factors]
        total_factors = len(all_factors)
        
        print(f"开始处理 {total_factors} 个因子...")
        
        for i, factor_name in enumerate(all_factors, 1):
            try:
                factor_data = self.get_factor_data(factor_name, start_time, end_time)
                
                file_path = os.path.join(output_dir, f"factor_data_{factor_name}.csv")
                factor_data.to_csv(file_path, index=False)
                
                # 打印进度
                print(f"[{i}/{total_factors}] 已保存因子 {factor_name} 到 {file_path}")
                
            except Exception as e:
                print(f"处理因子 {factor_name} 时出错: {e}")

        print(f"所有因子数据已保存到 {output_dir}, 耗时 {time.time() - start_timestamp:.2f} 秒")


if __name__ == "__main__":
    # 配置信息
    username = "jiangyushi"
    password = "123456"
    host = "10.176.55.211"
    port = 8848

    # 输出目录
    output_directory = "factor_data_files"
    
    # 时间范围
    start_time = "2020.01.01"
    end_time = "2025.09.01"
    
    # 指定因子名，获取对应数据（不指定则获取全部）
    factor_names = []
    # factor_names = ['Alpha1']
    
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


# nohup python -u scripts/get_all_factor.py > get_all_factor.log 2>&1 &