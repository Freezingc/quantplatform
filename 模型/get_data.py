import pandas as pd
import dolphindb as ddb
import numpy as np
from pandas.tseries.offsets import QuarterEnd
import os

class DataFetcher:
    def __init__(self, s):
        self.session = s
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

    #获得旧因子，已弃用
    def get_factor_matrix_old(self, factor_names, start_time="", end_time=""):
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

    #获得新因子
    def get_factor_matrix_new(self, factor_names=[], start_time="", end_time=""):
        
        def get_table_name(factor_name):
            query = f'''
                db = database("dfs://YesLab")
                t = loadTable(db, "factor_metadata")
                select store_factor from t where factor_name = '{factor_name}'
            '''
            result = self.session.run(query)
            if result is None or result.empty:
                raise ValueError(f"因子 {factor_name} 在 factor_metadata 表中不存在。")
            return result.iloc[0, 0]

        self.__set_db_and_table("dfs://YesLab", "factor_daily_data")
        # 如果开始时间为空，获取最早时间戳
        if not start_time:
            start_time = self.__get_time_range("trade_date", is_start=True)
        # 如果结束时间为空，获取最晚时间戳
        if not end_time:
            end_time = self.__get_time_range("trade_date", is_start=False)
        # 初始化factor_data
        factor_data = pd.DataFrame()
        factor_dfs = []
        if factor_names == []:
            query = f'''
                db = database("{self.db_name}")
                t = loadTable(db, "factor_metadata")
                select factor_name from t
            '''
            result = self.session.run(query)
            factor_names = result['factor_name'].tolist()
            if "test_factor" in factor_names:
                factor_names.remove("test_factor")
            print(f"Fetching factors: {factor_names}")
        for factor_name in factor_names:
            table_name = get_table_name(factor_name)
            print(factor_name,table_name)
            print(f"Fetching factor: {factor_name}")
            # 构建查询语句
            query = f'''
            db = database("{self.db_name}")
            t = loadTable(db, "{table_name}")
            select trade_date, ts_code, value from t
            where factor_name == "{factor_name}"
                and trade_date between date(`{start_time}) and date(`{end_time})
            '''
            # 执行查询并获取结果
            result = self.session.run(query)
            result = result.drop_duplicates(keep='first')
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

    def get_financial_metrics_matrix(self, factor_names, start_time="", end_time=""):
        self.__set_db_and_table("dfs://YesLab", "financial_metrics_table")
        # 如果开始时间为空，获取最早时间戳
        if not start_time:
            start_time = self.__get_time_range( "end_date", is_start=True)
        start_time_dt = pd.to_datetime(start_time)
        start_time_oneyearbefore_dt = start_time_dt - pd.DateOffset(years=1)
        start_time_oneyearbefore = start_time_oneyearbefore_dt.strftime("%Y.%m.%d")

        # 如果结束时间为空，获取最晚时间戳
        if not end_time:
            end_time = self.__get_time_range( "end_date", is_start=False)

        get_factor_query = f'''
            db = database("{self.db_name}")
            t = loadTable(db, "{self.table_name}")
            select ann_date, end_date, ts_code ,{",".join(factor_names)} from t
            where ann_date >= date(`{start_time_oneyearbefore}) and ann_date <= date(`{end_time}) 
        '''
        
        result = self.session.run(get_factor_query)

        # 先排序，确保处理顺序正确
        result = result.sort_values(['ts_code', 'ann_date', 'end_date'])
        
        # 使用 groupby 和 forward fill
        def process_group(df):
            if len(df) > 1:
                # 对分组进行前向填充
                df_filled = df.ffill()
                # 只取第一行
                return df_filled.iloc[[0]]
            return df

        group_keys = ['ts_code', 'ann_date', 'end_date']
        try:
            result = result.groupby(
                group_keys,
                as_index=False,
                sort=False,
                include_groups=False,
            ).apply(process_group)
        except TypeError:
            result = result.groupby(
                group_keys,
                as_index=False,
                sort=False,
            ).apply(process_group)
        result = result.reset_index(drop=True)

        result = result.sort_values('end_date')
        result = result.drop_duplicates(subset=['ts_code', 'ann_date'], keep='last')        
        cols_to_fill = [col for col in result.columns if col not in ('ann_date', 'end_date', 'ts_code')]
        result[cols_to_fill] = result[cols_to_fill].fillna(-2147483647)

        factor_data = pd.DataFrame()

        for factor in factor_names:
            # 对每个因子生成独立的矩阵（格式为 DataFrame）
            matrix = result.pivot(index="ts_code", columns="ann_date", values=factor)
            # 添加多级列索引：第一层为因子名称，第二层为日期
            matrix.columns = pd.MultiIndex.from_product([[factor], matrix.columns])
            matrix = matrix.ffill(axis=1)
            matrix = matrix.replace(-2147483647, np.nan)
            # 现在按日期筛选：只保留从start_time到end_time范围内的数据
        # 新增：获取交易日列表
            trade_days = self.get_trade_days(exchange="SSE", start_time=start_time, end_time=end_time)
            
            if not matrix.empty and len(trade_days) > 0:
                # 构造新的 MultiIndex columns：(factor, 交易日)
                new_columns = pd.MultiIndex.from_product([[factor], trade_days])
                
                # reindex 到交易日 + 前向填充
                matrix = matrix.reindex(columns=new_columns, method='ffill')
                
                # 可选：如果开头有 NaN（因为 start_time 前历史数据不足），可以后向填充
                # matrix = matrix.bfill(axis=1)   # 根据需求开启

            # 继续 concat
            factor_data = pd.concat([factor_data, matrix], axis=1)
        return factor_data
    
    #获得资产负债表数据
    def get_balance_sheet_matrix(self, factor_names, start_time="", end_time=""):
        self.__set_db_and_table("dfs://YesLab", "balance_sheet")
        # 如果开始时间为空，获取最早时间戳
        if not start_time:
            start_time = self.__get_time_range("end_date", is_start=True)
        start_time_dt = pd.to_datetime(start_time)
        start_time_oneyearbefore_dt = start_time_dt - pd.DateOffset(years=1)
        start_time_oneyearbefore = start_time_oneyearbefore_dt.strftime("%Y.%m.%d")

        # 如果结束时间为空，获取最晚时间戳
        if not end_time:
            end_time = self.__get_time_range("end_date", is_start=False)

        get_factor_query = f'''
            db = database("{self.db_name}")
            t = loadTable(db, "{self.table_name}")
            select ann_date, end_date, ts_code ,{",".join(factor_names)} from t
            where ann_date >= date(`{start_time_oneyearbefore}) and ann_date <= date(`{end_time}) 
        '''
        
        result = self.session.run(get_factor_query)

        # 先排序，确保处理顺序正确
        result = result.sort_values(['ts_code', 'ann_date', 'end_date'])
        
        # 使用 groupby 和 forward fill
        def process_group(df):
            if len(df) > 1:
                # 对分组进行前向填充
                df_filled = df.ffill()
                # 只取第一行
                return df_filled.iloc[[0]]
            return df

        result = result.groupby(['ts_code', 'ann_date', 'end_date'], as_index=False, sort=False).apply(process_group)
        result = result.reset_index(drop=True)

        result = result.sort_values('end_date')
        result = result.drop_duplicates(subset=['ts_code', 'ann_date'], keep='last')        
        cols_to_fill = [col for col in result.columns if col not in ('ann_date', 'end_date', 'ts_code')]
        result[cols_to_fill] = result[cols_to_fill].fillna(-2147483647)

        factor_data = pd.DataFrame()

        for factor in factor_names:
            # 对每个因子生成独立的矩阵（格式为 DataFrame）
            matrix = result.pivot(index="ts_code", columns="ann_date", values=factor)
            # 添加多级列索引：第一层为因子名称，第二层为日期
            matrix.columns = pd.MultiIndex.from_product([[factor], matrix.columns])
            matrix = matrix.ffill(axis=1)
            matrix = matrix.replace(-2147483647, np.nan)
            # 现在按日期筛选：只保留从start_time到end_time范围内的数据
                   # 新增：获取交易日列表
            trade_days = self.get_trade_days(exchange="SSE", start_time=start_time, end_time=end_time)
            
            if not matrix.empty and len(trade_days) > 0:
                # 构造新的 MultiIndex columns：(factor, 交易日)
                new_columns = pd.MultiIndex.from_product([[factor], trade_days])
                
                # reindex 到交易日 + 前向填充
                matrix = matrix.reindex(columns=new_columns, method='ffill')
                
                # 可选：如果开头有 NaN（因为 start_time 前历史数据不足），可以后向填充
                # matrix = matrix.bfill(axis=1)   # 根据需求开启

            # 继续 concat
            factor_data = pd.concat([factor_data, matrix], axis=1)
        return factor_data
    
    #获得现金流量表数据
    def get_cashflow_matrix(self, factor_names, start_time="", end_time=""):
        self.__set_db_and_table("dfs://YesLab", "cashflow_table")
        # 如果开始时间为空，获取最早时间戳
        if not start_time:
            start_time = self.__get_time_range("end_date", is_start=True)
        start_time_dt = pd.to_datetime(start_time)
        start_time_oneyearbefore_dt = start_time_dt - pd.DateOffset(years=1)
        start_time_oneyearbefore = start_time_oneyearbefore_dt.strftime("%Y.%m.%d")

        # 如果结束时间为空，获取最晚时间戳
        if not end_time:
            end_time = self.__get_time_range("end_date", is_start=False)

        get_factor_query = f'''
            db = database("{self.db_name}")
            t = loadTable(db, "{self.table_name}")
            select ann_date, end_date, ts_code ,{",".join(factor_names)} from t
            where ann_date >= date(`{start_time_oneyearbefore}) and ann_date <= date(`{end_time}) 
        '''
        
        result = self.session.run(get_factor_query)

        # 先排序，确保处理顺序正确
        result = result.sort_values(['ts_code', 'ann_date', 'end_date'])
        
        # 使用 groupby 和 forward fill
        def process_group(df):
            if len(df) > 1:
                # 对分组进行前向填充
                df_filled = df.ffill()
                # 只取第一行
                return df_filled.iloc[[0]]
            return df

        result = result.groupby(['ts_code', 'ann_date', 'end_date'], as_index=False, sort=False).apply(process_group)
        result = result.reset_index(drop=True)

        result = result.sort_values('end_date')
        result = result.drop_duplicates(subset=['ts_code', 'ann_date'], keep='last')        
        cols_to_fill = [col for col in result.columns if col not in ('ann_date', 'end_date', 'ts_code')]
        result[cols_to_fill] = result[cols_to_fill].fillna(-2147483647)

        factor_data = pd.DataFrame()

        for factor in factor_names:
            # 对每个因子生成独立的矩阵（格式为 DataFrame）
            matrix = result.pivot(index="ts_code", columns="ann_date", values=factor)
            # 添加多级列索引：第一层为因子名称，第二层为日期
            matrix.columns = pd.MultiIndex.from_product([[factor], matrix.columns])
            matrix = matrix.ffill(axis=1)
            matrix = matrix.replace(-2147483647, np.nan)
            # 现在按日期筛选：只保留从start_time到end_time范围内的数据
                   # 新增：获取交易日列表
            trade_days = self.get_trade_days(exchange="SSE", start_time=start_time, end_time=end_time)
            
            if not matrix.empty and len(trade_days) > 0:
                # 构造新的 MultiIndex columns：(factor, 交易日)
                new_columns = pd.MultiIndex.from_product([[factor], trade_days])
                
                # reindex 到交易日 + 前向填充
                matrix = matrix.reindex(columns=new_columns, method='ffill')
                
                # 可选：如果开头有 NaN（因为 start_time 前历史数据不足），可以后向填充
                # matrix = matrix.bfill(axis=1)   # 根据需求开启

            # 继续 concat
            factor_data = pd.concat([factor_data, matrix], axis=1)
        return factor_data
    
    #获得利润表数据
    def get_income_matrix(self, factor_names, start_time="", end_time=""):
        self.__set_db_and_table("dfs://YesLab", "company_income_table")
        # 如果开始时间为空，获取最早时间戳
        if not start_time:
            start_time = self.__get_time_range("end_date", is_start=True)
        start_time_dt = pd.to_datetime(start_time)
        start_time_oneyearbefore_dt = start_time_dt - pd.DateOffset(years=1)
        start_time_oneyearbefore = start_time_oneyearbefore_dt.strftime("%Y.%m.%d")

        # 如果结束时间为空，获取最晚时间戳
        if not end_time:
            end_time = self.__get_time_range("end_date", is_start=False)

        get_factor_query = f'''
            db = database("{self.db_name}")
            t = loadTable(db, "{self.table_name}")
            select ann_date, end_date, ts_code ,{",".join(factor_names)} from t
            where ann_date >= date(`{start_time_oneyearbefore}) and ann_date <= date(`{end_time}) 
        '''
        
        result = self.session.run(get_factor_query)

        # 先排序，确保处理顺序正确
        result = result.sort_values(['ts_code', 'ann_date', 'end_date'])
        
        # 使用 groupby 和 forward fill
        def process_group(df):
            if len(df) > 1:
                # 对分组进行前向填充
                df_filled = df.ffill()
                # 只取第一行
                return df_filled.iloc[[0]]
            return df

        result = result.groupby(['ts_code', 'ann_date', 'end_date'], as_index=False, sort=False).apply(process_group)
        result = result.reset_index(drop=True)

        result = result.sort_values('end_date')
        result = result.drop_duplicates(subset=['ts_code', 'ann_date'], keep='last')        
        cols_to_fill = [col for col in result.columns if col not in ('ann_date', 'end_date', 'ts_code')]
        result[cols_to_fill] = result[cols_to_fill].fillna(-2147483647)

        factor_data = pd.DataFrame()

        for factor in factor_names:
            # 对每个因子生成独立的矩阵（格式为 DataFrame）
            matrix = result.pivot(index="ts_code", columns="ann_date", values=factor)
            # 添加多级列索引：第一层为因子名称，第二层为日期
            matrix.columns = pd.MultiIndex.from_product([[factor], matrix.columns])
            matrix = matrix.ffill(axis=1)
            matrix = matrix.replace(-2147483647, np.nan)
            # 现在按日期筛选：只保留从start_time到end_time范围内的数据
                   # 新增：获取交易日列表
            trade_days = self.get_trade_days(exchange="SSE", start_time=start_time, end_time=end_time)
            
            if not matrix.empty and len(trade_days) > 0:
                # 构造新的 MultiIndex columns：(factor, 交易日)
                new_columns = pd.MultiIndex.from_product([[factor], trade_days])
                
                # reindex 到交易日 + 前向填充
                matrix = matrix.reindex(columns=new_columns, method='ffill')
                
                # 可选：如果开头有 NaN（因为 start_time 前历史数据不足），可以后向填充
                # matrix = matrix.bfill(axis=1)   # 根据需求开启

            # 继续 concat
            factor_data = pd.concat([factor_data, matrix], axis=1)
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

    def get_trade_days(self, exchange="SSE", start_time="", end_time=""):
        """
        获取指定范围内的交易日列表（只返回 is_open == 1 的 cal_date）
        返回：pd.DatetimeIndex（推荐用于 reindex）
        """
        self.__set_db_and_table("dfs://YesLab", "trade_calendar")
        
        if not start_time:
            start_time = self.__get_time_range("cal_date", is_start=True)
        if not end_time:
            end_time = self.__get_time_range("cal_date", is_start=False)
        
        query = f'''
        db = database("{self.db_name}")
        t = loadTable(db, "{self.table_name}")
        select cal_date 
        from t 
        where cal_date between date(`{start_time}) and date(`{end_time}) 
        and exchange = "{exchange}" 
        and is_open == 1
        order by cal_date asc
        '''
        result = self.session.run(query)
        
        # 转成 pandas datetime
        trade_days = pd.to_datetime(result['cal_date'].values)
        return trade_days

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()  # 默认加载当前目录的 .env
    
    username = "zhangyudong"
    password = "123456"

    # 使用示例
    #日期可以不填
    start_time = "2023.12.06"
    end_time = "2023.12.12"
    factor_names = ['Alpha1','Alpha60']

    # 量价数据获取用get_stock_daily_matrix方法
    # 旧因子数据（主要是alpha191因子）获取用get_factor_matrix_old方法,该方法已弃用，建议使用get_factor_matrix_new方法
    # 新因子数据（目前主要的因子库数据）获取用get_factor_matrix_new方法
    # 指数数据获取用get_index_daily_matrix方法
    # 获得股票是否ST用get_stock_is_ST方法
    # 获得股票是否停牌用get_stock_is_suspended方法
    # 获得交易日历用get_trade_calendar方法
    # 获得财报数据（根据ann_date公告日期）用get_financial_metrics_matrix_anndate方法
    # 获得财报具体数据（根据end_date日期）用
    # 

    
    s = ddb.session()
    s.connect("10.176.55.211", 8848, username, password)

    try:
        fetcher = DataFetcher(s = s)
        data = fetcher.get_factor_matrix_new(factor_names=factor_names,start_time = start_time, end_time = end_time)
        print(data)
        
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"发生错误: {e}")

    s.close()