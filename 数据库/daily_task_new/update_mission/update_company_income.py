from tqdm import tqdm
import dolphindb as ddb
import pandas as pd
import time
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import tushare as ts

def update_company_income():
    load_dotenv()  # 默认加载当前目录的 .env
    username = os.getenv('dolphinDB_USERNAME')  # 从环境变量读取
    password = os.getenv('dolphinDB_PASSWORD')

    # 连接 DolphinDB 数据库
    s = ddb.session()
    s.connect("10.176.55.211", 8848, username, password)

    # 连接 Tushare API
    pro = ts.pro_api(os.getenv('TUSHARE_TOKEN'))

    code_list = pro.stock_basic(fields = 'ts_code')

    time_list = pro.trade_cal(exchange='', is_open='', 
                            start_date='',
                            end_date='',
                            fields='cal_date')

    # 获取数据库中已有数据的最大日期
    max_date_query = '''
        db = database("dfs://YesLab")
        t = loadTable(db,'company_income_table')
        select max(ann_date) from t
    '''
    max_date_result = s.run(max_date_query)

    # 如果数据库中有数据，获取最大日期并增加一天
    if not pd.isna(max_date_result['max_ann_date'].iloc[0]):
        max_date = pd.to_datetime(max_date_result['max_ann_date'][0])
        start_date = (max_date + pd.Timedelta(days=1)).strftime('%Y%m%d')  # 转换为 YYYY.MM.DD 格式
    else:
        # 如果数据库为空，设定默认的开始日期
        start_date = ''

    # 获取当前日期
    today = datetime.today()
    # 格式化为 YYYY.MM.DD 的格式
    end_date = today.strftime("%Y%m%d")

    print("更新起始日期:",start_date)
    print("更新截止日期:",end_date)

    time_list = pro.trade_cal(exchange='', is_open='', 
                            start_date=start_date,
                            end_date=end_date,
                            fields='cal_date')
    # print(time_list)

    # 每次获取的数据条目
    chunk_size = 1000

    time_list = time_list.sort_values(by='cal_date')
    date_ranges = []
    num_chunks = (len(time_list) + chunk_size - 1) // chunk_size  # 向上取整
    # 遍历每个子集并记录日期范围
    for i in range(num_chunks):
        start_date = time_list['cal_date'].iloc[i * chunk_size]
        end_date = time_list['cal_date'].iloc[min((i + 1) * chunk_size - 1, len(time_list) - 1)]
        date_ranges.append((start_date, end_date))
    # 将日期转换为 pd 的 datetime 格式
    date_ranges = [(pd.to_datetime(start).strftime('%Y%m%d'), pd.to_datetime(end).strftime('%Y%m%d')) for start, end in date_ranges]
    # 打印结果
    # print(date_ranges)

    fields = [
        "ts_code",
        "ann_date",
        "f_ann_date",
        "end_date",
        "report_type",
        "comp_type",
        "end_type",
        "basic_eps",
        "diluted_eps",
        "total_revenue",
        "revenue",
        "int_income",
        "prem_earned",
        "comm_income",
        "n_commis_income",
        "n_oth_income",
        "n_oth_b_income",
        "prem_income",
        "out_prem",
        "une_prem_reser",
        "reins_income",
        "n_sec_tb_income",
        "n_sec_uw_income",
        "n_asset_mg_income",
        "oth_b_income",
        "fv_value_chg_gain",
        "invest_income",
        "ass_invest_income",
        "forex_gain",
        "total_cogs",
        "oper_cost",
        "int_exp",
        "comm_exp",
        "biz_tax_surchg",
        "sell_exp",
        "admin_exp",
        "fin_exp",
        "assets_impair_loss",
        "prem_refund",
        "compens_payout",
        "reser_insur_liab",
        "div_payt",
        "reins_exp",
        "oper_exp",
        "compens_payout_refu",
        "insur_reser_refu",
        "reins_cost_refund",
        "other_bus_cost",
        "operate_profit",
        "non_oper_income",
        "non_oper_exp",
        "nca_disploss",
        "total_profit",
        "income_tax",
        "n_income",
        "n_income_attr_p",
        "minority_gain",
        "oth_compr_income",
        "t_compr_income",
        "compr_inc_attr_p",
        "compr_inc_attr_m_s",
        "ebit",
        "ebitda",
        "insurance_exp",
        "undist_profit",
        "distable_profit",
        "rd_exp",
        "fin_exp_int_exp",
        "fin_exp_int_inc",
        "transfer_surplus_rese",
        "transfer_housing_imprest",
        "transfer_oth",
        "adj_lossgain",
        "withdra_legal_surplus",
        "withdra_legal_pubfund",
        "withdra_biz_devfund",
        "withdra_rese_fund",
        "withdra_oth_ersu",
        "workers_welfare",
        "distr_profit_shrhder",
        "prfshare_payable_dvd",
        "comshare_payable_dvd",
        "capit_comstock_div",
        "continued_net_profit",
        "update_flag"
    ]
    df = pd.DataFrame(columns = fields)
    df["end_date"] = pd.to_datetime(df["end_date"])
    df["f_ann_date"] = pd.to_datetime(df["f_ann_date"])
    df["ann_date"] = pd.to_datetime(df["ann_date"])
    # 拉取并写入数据
    for index, row in tqdm(code_list.iterrows(), total=len(code_list)):
        for i, (start, end) in enumerate(date_ranges):
            error_count = 0  # 初始化错误计数器
            max_errors = 10  # 最大允许错误次数
            while True:
                try:
                    df_new = pro.income(**{
                        "ts_code": row.ts_code,
                        "ann_date": "",
                        "f_ann_date": "",
                        "start_date": start,
                        "end_date": end,
                        "period": "",
                        "report_type": "",
                        "comp_type": "",
                        "is_calc": "",
                        "limit": "",
                        "offset": ""
                    }, fields=fields)
                    # 将 'trade_date' 转换为 datetime 格式
                    df_new["end_date"] = pd.to_datetime(df_new["end_date"])
                    df_new["f_ann_date"] = pd.to_datetime(df_new["f_ann_date"])
                    df_new["ann_date"] = pd.to_datetime(df_new["ann_date"])
                    df = pd.concat([df, df_new], ignore_index=True)
                    break
                except Exception as e:
                    error_count += 1  # 错误计数加1
                    print(f"发生异常: {e}, 1min后重新尝试... (错误次数: {error_count}/{max_errors})")
                    
                    if error_count >= max_errors:
                        print(f"已达到最大错误次数 {max_errors}，停止尝试...")
                        import sys
                        sys.exit(1)
                        
                    time.sleep(60)  # 休息1min
    return "", df