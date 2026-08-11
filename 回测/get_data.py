import pandas as pd
import dolphindb as ddb
import numpy as np
from pandas.tseries.offsets import QuarterEnd

class DataFetcher:
    def __init__(self, host="10.176.55.211", port=8848, user=None, password=None):
        if user is None or password is None:
            raise ValueError("User and password must be provided.")
        
        # 创建数据库连接
        self.session = ddb.session()
        self.session.connect(host, port, user, password)
        self.db_name = None
        self.table_name = None
        self.date_name = None

    def __get_time_range(self, date_name, is_start=True):
        # 获取最早或最晚时间戳
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
        if not db_name or not table_name:
            raise ValueError("Database name and table name must be provided.")
        self.db_name = db_name
        self.table_name = table_name

    #获得因子
    def get_factor_matrix(self, factor_names, start_time="", end_time=""):
        self.__set_db_and_table("dfs://dayFactorDB", "dayFactorTB")
        # 如果开始时间为空，获取最早时间戳
        if not start_time:
            start_time = self.__get_time_range("tradetime", is_start=True)
        # 如果结束时间为空，获取最晚时间戳
        if not end_time:
            end_time = self.__get_time_range("tradetime", is_start=False)
        # 初始化factor_data
        factor_data = pd.DataFrame()

        for factor_name in factor_names:
            # 构建查询语句
            query = f'''
            db = database("{self.db_name}")
            t = loadTable(db, "{self.table_name}")
            select tradetime, securityid, value from t
            where factorname == "{factor_name}"
                and tradetime between date(`{start_time}) and date(`{end_time})
            '''
            # 执行查询并获取结果
            result = self.session.run(query)
            result = result.drop_duplicates(keep='first')
            result = result.rename(columns={'securityid': 'ts_code', 'tradetime': 'trade_date'})
            matrix = result.pivot(index='ts_code', columns='trade_date', values='value')
            matrix.columns = pd.MultiIndex.from_product([[factor_name], matrix.columns])
            factor_data = pd.concat([factor_data, matrix], axis=1)

        return factor_data

    #获得指数每日数据
    def get_index_daily_matrix(self, factor_names, start_time="", end_time=""):
        self.__set_db_and_table("dfs://YesLab", "index_daily_data")
        # 如果开始时间为空，获取最早时间戳
        if not start_time:
            start_time = self.__get_time_range("trade_date", is_start=True)
        # 如果结束时间为空，获取最晚时间戳
        if not end_time:
            end_time = self.__get_time_range("trade_date", is_start=False)
        # 初始化factor_data
        factor_data = pd.DataFrame()

        get_factor_query = f'''
        db = database("{self.db_name}")
        t = loadTable(db, "{self.table_name}")
        select trade_date, ts_code, {",".join(factor_names)} from t
        where trade_date between date(`{start_time}) and date(`{end_time})
        '''
        # print(get_factor_query)
        result = self.session.run(get_factor_query)
        result = result.drop_duplicates(keep='first')
        factor_names = [col for col in result.columns if col not in ["trade_date", "ts_code"]]

        factor_data = pd.DataFrame()

        for factor in factor_names:
            # 对每个因子生成独立的矩阵（格式为 DataFrame）
            matrix = result.pivot(index="ts_code", columns="trade_date", values=factor)
            
            # 添加多级列索引：第一层为因子名称，第二层为日期
            matrix.columns = pd.MultiIndex.from_product([[factor], matrix.columns])
            
            # 将当前因子的矩阵按列合并到总 DataFrame 中
            factor_data = pd.concat([factor_data, matrix], axis=1)

        return factor_data

    #获得股票每日数据
    def get_stock_daily_matrix(self, factor_names, start_time="", end_time=""):
        self.__set_db_and_table("dfs://YesLab", "stock_factor_daily")
        # 如果开始时间为空，获取最早时间戳
        if not start_time:
            start_time = self.__get_time_range("trade_date", is_start=True)
        # 如果结束时间为空，获取最晚时间戳
        if not end_time:
            end_time = self.__get_time_range("trade_date", is_start=False)
        # 初始化factor_data
        factor_data = pd.DataFrame()

        get_factor_query = f'''
        db = database("{self.db_name}")
        t = loadTable(db, "{self.table_name}")
        select trade_date, ts_code, {",".join(factor_names)} from t
        where trade_date between date(`{start_time}) and date(`{end_time})
        '''
        # print(get_factor_query)
        result = self.session.run(get_factor_query)
        result = result.drop_duplicates(keep='first')
        factor_names = [col for col in result.columns if col not in ["trade_date", "ts_code"]]

        factor_data = pd.DataFrame()

        for factor in factor_names:
            # 对每个因子生成独立的矩阵（格式为 DataFrame）
            matrix = result.pivot(index="ts_code", columns="trade_date", values=factor)
            
            # 添加多级列索引：第一层为因子名称，第二层为日期
            matrix.columns = pd.MultiIndex.from_product([[factor], matrix.columns])
            
            # 将当前因子的矩阵按列合并到总 DataFrame 中
            factor_data = pd.concat([factor_data, matrix], axis=1)

        return factor_data

    #获得财报数据,仅获得ann_date公告日期，而不获得具体数值
    def get_financial_metrics_matrix_anndate(self, start_time="", end_time=""):
        self.__set_db_and_table("dfs://YesLab", "financial_metrics_table")
        # 如果开始时间为空，获取最早时间戳
        if not start_time:
            start_time = self.__get_time_range( "end_date", is_start=True)
        # 如果结束时间为空，获取最晚时间戳
        if not end_time:
            end_time = self.__get_time_range( "end_date", is_start=False)

        get_factor_query = f'''
            db = database("{self.db_name}")
            t = loadTable(db, "{self.table_name}")
            select ann_date, ts_code ,update_flag from t
            where ann_date >= date(`{start_time}) and ann_date <= date(`{end_time}) 
        '''
        # and update_flag == 0
        
        result_get_factor_query = self.session.run(get_factor_query)
        result_get_factor_query = result_get_factor_query.drop_duplicates(keep='first')
        
        get_ts_code_query = f'''
            db = database("dfs://YesLab")
            t = loadTable(db, "financial_metrics_table")
            select distinct ts_code from t
        '''
        ts_code_df = self.session.run(get_ts_code_query)
        unique_ts_codes = ts_code_df['ts_code'].astype(str).tolist()
        #print(f"不重复的ts_code数量: {len(unique_ts_codes)}")

        # 2. 创建日期范围
        date_range = pd.date_range(start=start_time, end=end_time, freq='D')

        # 3. 创建空的DataFrame，行是日期，列是ts_code
        factor_data = pd.DataFrame(0, index=date_range, columns=unique_ts_codes)

        # 4. 填充矩阵
        for _, row in result_get_factor_query.iterrows():
            date = pd.to_datetime(row['ann_date'])
            ts_code = row['ts_code']
            factor_data.loc[date, ts_code] = 1

        return factor_data.T

    #获得财报数据
    def get_financial_metrics_matrix(self, factor_names, start_time="", end_time=""):
        self.__set_db_and_table("dfs://YesLab", "financial_metrics_table")
        # 如果开始时间为空，获取最早时间戳
        if not start_time:
            start_time = self.__get_time_range( "end_date", is_start=True)
        # 如果结束时间为空，获取最晚时间戳
        if not end_time:
            end_time = self.__get_time_range( "end_date", is_start=False)

        # 初始化factor_data
        factor_data = pd.DataFrame()

        get_ts_code_query = f'''
            db = database("dfs://YesLab")
            t = loadTable(db, "financial_metrics_table")
            select distinct ts_code from t
        '''
        ts_code_df = self.session.run(get_ts_code_query)
        unique_ts_codes = ts_code_df['ts_code'].astype(str).tolist()
    
        for factor_name in factor_names:
            get_factor_query = f'''
                db = database("{self.db_name}")
                t = loadTable(db, "{self.table_name}")
                select ann_date, end_date, ts_code, {factor_name} from t
                where end_date >= date(`{start_time}) and end_date <= date(`{end_time}) 
            '''

            #print(get_factor_query)
            result_get_factor_query = self.session.run(get_factor_query)
            # 去重后的数据
            result_get_factor_query = result_get_factor_query.drop_duplicates(keep='first')
            idx = result_get_factor_query.groupby(['end_date', 'ts_code'])['ann_date'].idxmax()
            result_get_factor_query = result_get_factor_query.loc[idx]
            #print(f"去重后数据量: {len(result_get_factor_query)}")
            #print(result_get_factor_query)

            # 创建日期范围
            date_range = pd.date_range(
                start=start_time, 
                end=end_time, 
                freq=QuarterEnd(n=1)  # n=1 表示每季度一次
            )
            #print(f"日期范围: {date_range}")
            data_factor = pd.DataFrame(np.nan, index=date_range, columns=unique_ts_codes)
            for _, row in result_get_factor_query.iterrows():
                date = pd.to_datetime(row['end_date'])
                ts_code = row['ts_code']
                data_factor.loc[date, ts_code] = row[factor_name]

            data_factor = data_factor.T
            # 添加因子名称列
            data_factor.columns = pd.MultiIndex.from_product([[factor_name], data_factor.columns])
            
            # 将当前因子的数据合并到总表
            factor_data = pd.concat([factor_data, data_factor], axis=1)
        
        return factor_data
    
    #获得ST数据,值为1代表ST，值不为1代表非ST（可能退市等）
    def get_stock_is_ST(self, start_time="",end_time=""):
        def process_tscode_records(df, start_time, end_time, ts_code):
            # 处理单个ts_code的记录
            df = df.sort_values('start_date').reset_index(drop=True)
            # 创建全0时间线
            timeline = []
            dates = pd.date_range(start=start_time, end=end_time,freq='D')
            timeline.append(pd.DataFrame({
                'date': dates,
                'st_status': 0
            }))
            start_time = pd.to_datetime(start_time, format='%Y.%m.%d')
            end_time = pd.to_datetime(end_time, format='%Y.%m.%d')

            for i in range(len(df)):
                # 获取当前记录的有效期
                start = pd.to_datetime(df.at[i, 'start_date'])
                end = pd.to_datetime(df.at[i, 'end_date']) if pd.notna(df.at[i, 'end_date']) else None
                if end is not None and end < start_time:
                    continue
                if start is not None and start> end_time:
                    continue
                if 'ST' not in df.at[i, 'name']:
                    continue
                # 处理没有开始日期、结束日期的记录
                if start is None:
                    start = start_time
                if end is None:
                    end = end_time

                # 生成日期序列
                dates = pd.date_range(start=max(start,start_time), end=min(end,end_time),freq='D')
                
                timeline.append(pd.DataFrame({
                    'date': dates,
                    'st_status': 1
                }))
            # 合并时间线并去重（保留最后出现的记录）
            full_timeline = pd.concat(timeline).drop_duplicates('date', keep='last')
            return full_timeline.set_index('date')['st_status']
        
        self.__set_db_and_table("dfs://YesLab", "stock_used_name")
        # 如果开始时间为空，获取最早时间戳
        if not start_time:
            start_time = self.__get_time_range("start_date", is_start=True)
        # 如果结束时间为空，获取最晚时间戳
        if not end_time:
            end_time = self.__get_time_range("end_date", is_start=False)
        # 初始化factor_data
        get_code_list_query = f'''
        db = database("{self.db_name}")
        t = loadTable(db, "{self.table_name}")
        select distinct ts_code from t
        '''
        
        ts_code_df  = self.session.run(get_code_list_query)
        ts_code_list = ts_code_df['ts_code'].astype(str).tolist()

        date_index = pd.date_range(start=start_time, end=end_time, freq='D')
        final_df = pd.DataFrame(
            data=0,  # 初始值设为0
            index=date_index,
            columns=ts_code_list,
            dtype=np.int8  # 节省内存
        )
        #print(final_df)
        
        for ts_code in ts_code_list:
            get_result_query = f'''
                db = database("{self.db_name}")
                t = loadTable(db, "{self.table_name}")
                select * from t
                where ts_code = "{ts_code}"
            '''
            records = self.session.run(get_result_query)
            records = (
                records
                # 将非空end_date的行排在前面
                .sort_values(by=['ts_code', 'name', 'start_date', 'end_date'], 
                            na_position='last')
                # 按关键字段去重，保留第一个（非空end_date）
                .drop_duplicates(
                    subset=['ts_code', 'name', 'start_date', 'ann_date', 'change_reason'],
                    keep='first'
                )
            )

            records['start_date'] = pd.to_datetime(records['start_date'])
            records['end_date'] = pd.to_datetime(records['end_date'], errors='coerce')
            ts_series = process_tscode_records(records, start_time, end_time, ts_code)
            final_df[ts_code] = ts_series.astype(np.int8)
        final_df = final_df.T
        #print(final_df)
        return final_df
    
    #获得股票是否停牌，值为1代表停牌，值不为1仅代表不停牌
    def get_stock_is_suspended(self, start_time="",end_time=""):
        def process_tscode_records(df, start_time, end_time, ts_code):
            # 处理单个ts_code的记录
            df = df.sort_values('trade_date').reset_index(drop=True)
            # 创建全0时间线
            timeline = []
            dates = pd.date_range(start=start_time, end=end_time,freq='D')
            timeline.append(pd.DataFrame({
                'date': dates,
                'st_status': 0
            }))
            start_time = pd.to_datetime(start_time, format='%Y.%m.%d')
            end_time = pd.to_datetime(end_time, format='%Y.%m.%d')

            for i in range(len(df)):
                # 获取当前记录的有效期
                trade_date = pd.to_datetime(df.at[i, 'trade_date'])
                
                if trade_date is not None and (trade_date < start_time or trade_date > end_time):
                    continue

                if 'S' not in df.at[i, 'suspend_type']:
                    continue

                # 生成日期序列
                dates = pd.date_range(start=trade_date, end=trade_date,freq='D')
                
                timeline.append(pd.DataFrame({
                    'date': dates,
                    'st_status': 1
                }))
            # 合并时间线并去重（保留最后出现的记录）
            full_timeline = pd.concat(timeline).drop_duplicates('date', keep='last')
            return full_timeline.set_index('date')['st_status']
        
        self.__set_db_and_table("dfs://YesLab", "stock_trading_suspension_info")
        # 如果开始时间为空，获取最早时间戳
        if not start_time:
            start_time = self.__get_time_range("trade_date", is_start=True)
        # 如果结束时间为空，获取最晚时间戳
        if not end_time:
            end_time = self.__get_time_range("trade_date", is_start=False)
        # 初始化factor_data
        get_code_list_query = f'''
        db = database("{self.db_name}")
        t = loadTable(db, "{self.table_name}")
        select distinct ts_code from t
        '''
        
        ts_code_df  = self.session.run(get_code_list_query)
        ts_code_list = ts_code_df['ts_code'].astype(str).tolist()

        date_index = pd.date_range(start=start_time, end=end_time, freq='D')
        final_df = pd.DataFrame(
            data=0,  # 初始值设为0
            index=date_index,
            columns=ts_code_list,
            dtype=np.int8  # 节省内存
        )
        #print(final_df)
        
        for ts_code in ts_code_list:
            get_result_query = f'''
                db = database("{self.db_name}")
                t = loadTable(db, "{self.table_name}")
                select * from t
                where ts_code = "{ts_code}"
            '''
            records = self.session.run(get_result_query)
            records['trade_date'] = pd.to_datetime(records['trade_date'])
            ts_series = process_tscode_records(records, start_time, end_time, ts_code)
            final_df[ts_code] = ts_series.astype(np.int8)
        final_df = final_df.T
        #print(final_df)
        return final_df

    #获得交易日历，exchange代表交易所，SSE上交所，SZSE深交所，默认为SSE
    def get_trade_calendar(self, exchange = "SSE", start_time="",end_time=""):
        self.__set_db_and_table("dfs://YesLab", "trade_calendar")
        # 如果开始时间为空，获取最早时间戳
        if not start_time:
            start_time = self.__get_time_range("cal_date", is_start=True)
        # 如果结束时间为空，获取最晚时间戳
        if not end_time:
            end_time = self.__get_time_range("cal_date", is_start=False)
        # 初始化factor_data
        factor_data = pd.DataFrame()
        get_factor_query = f'''
        db = database("{self.db_name}")
        t = loadTable(db, "{self.table_name}")
        select cal_date, is_open from t
        where cal_date between date(`{start_time}) and date(`{end_time}) and exchange = "{exchange}"
        '''
        result = self.session.run(get_factor_query)
        result = result.set_index('cal_date').T
        return result