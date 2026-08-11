from tqdm import tqdm
import dolphindb as ddb
import pandas as pd
import time
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import tushare as ts

def update_financial_metrics_table():
    load_dotenv()  # 默认加载当前目录的 .env
    username = os.getenv('dolphinDB_USERNAME')  # 从环境变量读取
    password = os.getenv('dolphinDB_PASSWORD')

    # 连接 DolphinDB 数据库
    s = ddb.session()
    s.connect("10.176.55.211", 8848, username, password)

    # 连接 Tushare API
    pro = ts.pro_api(os.getenv('TUSHARE_TOKEN'))

    code_list = pro.stock_basic(fields = 'ts_code')


    #根据end_date更新！
    # 获取数据库中已有数据的最大日期
    max_date_query = '''
        db = database("dfs://YesLab")
        t = loadTable(db,'financial_metrics_table')
        select max(end_date) from t
    '''
    max_date_result = s.run(max_date_query)

    # 如果数据库中有数据，获取最大日期并增加一天
    if not pd.isna(max_date_result['max_end_date'].iloc[0]):
        max_date = pd.to_datetime(max_date_result['max_end_date'][0])
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
    print(time_list)

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
        "end_date",
        "eps",
        "dt_eps",
        "total_revenue_ps",
        "revenue_ps",
        "capital_rese_ps",
        "surplus_rese_ps",
        "undist_profit_ps",
        "extra_item",
        "profit_dedt",
        "gross_margin",
        "current_ratio",
        "quick_ratio",
        "cash_ratio",
        "ar_turn",
        "ca_turn",
        "fa_turn",
        "assets_turn",
        "op_income",
        "ebit",
        "ebitda",
        "fcff",
        "fcfe",
        "current_exint",
        "noncurrent_exint",
        "interestdebt",
        "netdebt",
        "tangible_asset",
        "working_capital",
        "networking_capital",
        "invest_capital",
        "retained_earnings",
        "diluted2_eps",
        "bps",
        "ocfps",
        "retainedps",
        "cfps",
        "ebit_ps",
        "fcff_ps",
        "fcfe_ps",
        "netprofit_margin",
        "grossprofit_margin",
        "cogs_of_sales",
        "expense_of_sales",
        "profit_to_gr",
        "saleexp_to_gr",
        "adminexp_of_gr",
        "finaexp_of_gr",
        "impai_ttm",
        "gc_of_gr",
        "op_of_gr",
        "ebit_of_gr",
        "roe",
        "roe_waa",
        "roe_dt",
        "roa",
        "npta",
        "roic",
        "roe_yearly",
        "roa2_yearly",
        "debt_to_assets",
        "assets_to_eqt",
        "dp_assets_to_eqt",
        "ca_to_assets",
        "nca_to_assets",
        "tbassets_to_totalassets",
        "int_to_talcap",
        "eqt_to_talcapital",
        "currentdebt_to_debt",
        "longdeb_to_debt",
        "ocf_to_shortdebt",
        "debt_to_eqt",
        "eqt_to_debt",
        "eqt_to_interestdebt",
        "tangibleasset_to_debt",
        "tangasset_to_intdebt",
        "tangibleasset_to_netdebt",
        "ocf_to_debt",
        "turn_days",
        "roa_yearly",
        "roa_dp",
        "fixed_assets",
        "profit_to_op",
        "q_saleexp_to_gr",
        "q_gc_to_gr",
        "q_roe",
        "q_dt_roe",
        "q_npta",
        "q_ocf_to_sales",
        "basic_eps_yoy",
        "dt_eps_yoy",
        "cfps_yoy",
        "op_yoy",
        "ebt_yoy",
        "netprofit_yoy",
        "dt_netprofit_yoy",
        "ocf_yoy",
        "roe_yoy",
        "bps_yoy",
        "assets_yoy",
        "eqt_yoy",
        "tr_yoy",
        "or_yoy",
        "q_sales_yoy",
        "q_op_qoq",
        "equity_yoy",
        "invturn_days",
        "arturn_days",
        "inv_turn",
        "valuechange_income",
        "interst_income",
        "daa",
        "roe_avg",
        "opincome_of_ebt",
        "investincome_of_ebt",
        "n_op_profit_of_ebt",
        "tax_to_ebt",
        "dtprofit_to_profit",
        "salescash_to_or",
        "ocf_to_or",
        "ocf_to_opincome",
        "capitalized_to_da",
        "ocf_to_interestdebt",
        "ocf_to_netdebt",
        "ebit_to_interest",
        "longdebt_to_workingcapital",
        "ebitda_to_debt",
        "profit_prefin_exp",
        "non_op_profit",
        "op_to_ebt",
        "nop_to_ebt",
        "ocf_to_profit",
        "cash_to_liqdebt",
        "cash_to_liqdebt_withinterest",
        "op_to_liqdebt",
        "op_to_debt",
        "roic_yearly",
        "total_fa_trun",
        "q_opincome",
        "q_investincome",
        "q_dtprofit",
        "q_eps",
        "q_netprofit_margin",
        "q_gsprofit_margin",
        "q_exp_to_sales",
        "q_profit_to_gr",
        "q_adminexp_to_gr",
        "q_finaexp_to_gr",
        "q_impair_to_gr_ttm",
        "q_op_to_gr",
        "q_opincome_to_ebt",
        "q_investincome_to_ebt",
        "q_dtprofit_to_profit",
        "q_salescash_to_or",
        "q_ocf_to_or",
        "q_gr_yoy",
        "q_gr_qoq",
        "q_sales_qoq",
        "q_op_yoy",
        "q_profit_yoy",
        "q_profit_qoq",
        "q_netprofit_yoy",
        "q_netprofit_qoq",
        "rd_exp",
        "update_flag"
    ]
    df = pd.DataFrame(columns = fields)
    df["end_date"] = pd.to_datetime(df["end_date"])
    df["ann_date"] = pd.to_datetime(df["ann_date"])
    df["update_flag"] = df["update_flag"].astype(int)
    # 拉取并写入数据
    for index, row in tqdm(code_list.iterrows(), total=len(code_list)):
        for i, (start, end) in enumerate(date_ranges):
            error_count = 0  # 初始化错误计数器
            max_errors = 10  # 最大允许错误次数
            while True:
                try:
                    df_new = pro.fina_indicator(**{
                        "ts_code": row.ts_code,
                        "ann_date": "",
                        "start_date": start,
                        "end_date": end,
                        "period": "",
                        "update_flag": "",
                        "limit": "",
                        "offset": ""
                    }, fields=fields)
                    # 将 'trade_date' 转换为 datetime 格式
                    df_new["end_date"] = pd.to_datetime(df_new["end_date"])
                    df_new["ann_date"] = pd.to_datetime(df_new["ann_date"])
                    df_new["update_flag"] = df_new["update_flag"].astype(int)
                    # 向数据库中追加数据
                    df = pd.concat([df, df_new], ignore_index=True)
                    #print(date, num)
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