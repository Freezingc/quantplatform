from tqdm import tqdm
import dolphindb as ddb
import pandas as pd
import time
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import tushare as ts

def update_balance_sheet():
    load_dotenv()  # 默认加载当前目录的 .env
    username = os.getenv('dolphinDB_USERNAME')  # 从环境变量读取
    password = os.getenv('dolphinDB_PASSWORD')

    # 连接 DolphinDB 数据库
    s = ddb.session()
    s.connect("10.176.55.211", 8848, username, password)

    # 连接 Tushare API
    pro = ts.pro_api(os.getenv('TUSHARE_TOKEN'))

    code_list = pro.stock_basic(fields = 'ts_code')

    # 获取数据库中已有数据的最大日期
    max_date_query = '''
        db = database("dfs://YesLab")
        t = loadTable(db,'balance_sheet')
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
        "total_share",
        "cap_rese",
        "undistr_porfit",
        "surplus_rese",
        "special_rese",
        "money_cap",
        "trad_asset",
        "notes_receiv",
        "accounts_receiv",
        "oth_receiv",
        "prepayment",
        "div_receiv",
        "int_receiv",
        "inventories",
        "amor_exp",
        "nca_within_1y",
        "sett_rsrv",
        "loanto_oth_bank_fi",
        "premium_receiv",
        "reinsur_receiv",
        "reinsur_res_receiv",
        "pur_resale_fa",
        "oth_cur_assets",
        "total_cur_assets",
        "fa_avail_for_sale",
        "htm_invest",
        "lt_eqt_invest",
        "invest_real_estate",
        "time_deposits",
        "oth_assets",
        "lt_rec",
        "fix_assets",
        "cip",
        "const_materials",
        "fixed_assets_disp",
        "produc_bio_assets",
        "oil_and_gas_assets",
        "intan_assets",
        "r_and_d",
        "goodwill",
        "lt_amor_exp",
        "defer_tax_assets",
        "decr_in_disbur",
        "oth_nca",
        "total_nca",
        "cash_reser_cb",
        "depos_in_oth_bfi",
        "prec_metals",
        "deriv_assets",
        "rr_reins_une_prem",
        "rr_reins_outstd_cla",
        "rr_reins_lins_liab",
        "rr_reins_lthins_liab",
        "refund_depos",
        "ph_pledge_loans",
        "refund_cap_depos",
        "indep_acct_assets",
        "client_depos",
        "client_prov",
        "transac_seat_fee",
        "invest_as_receiv",
        "total_assets",
        "lt_borr",
        "st_borr",
        "cb_borr",
        "depos_ib_deposits",
        "loan_oth_bank",
        "trading_fl",
        "notes_payable",
        "acct_payable",
        "adv_receipts",
        "sold_for_repur_fa",
        "comm_payable",
        "payroll_payable",
        "taxes_payable",
        "int_payable",
        "div_payable",
        "oth_payable",
        "acc_exp",
        "deferred_inc",
        "st_bonds_payable",
        "payable_to_reinsurer",
        "rsrv_insur_cont",
        "acting_trading_sec",
        "acting_uw_sec",
        "non_cur_liab_due_1y",
        "oth_cur_liab",
        "total_cur_liab",
        "bond_payable",
        "lt_payable",
        "specific_payables",
        "estimated_liab",
        "defer_tax_liab",
        "defer_inc_non_cur_liab",
        "oth_ncl",
        "total_ncl",
        "depos_oth_bfi",
        "deriv_liab",
        "depos",
        "agency_bus_liab",
        "oth_liab",
        "prem_receiv_adva",
        "depos_received",
        "ph_invest",
        "reser_une_prem",
        "reser_outstd_claims",
        "reser_lins_liab",
        "reser_lthins_liab",
        "indept_acc_liab",
        "pledge_borr",
        "indem_payable",
        "policy_div_payable",
        "total_liab",
        "treasury_share",
        "ordin_risk_reser",
        "forex_differ",
        "invest_loss_unconf",
        "minority_int",
        "total_hldr_eqy_exc_min_int",
        "total_hldr_eqy_inc_min_int",
        "total_liab_hldr_eqy",
        "lt_payroll_payable",
        "oth_comp_income",
        "oth_eqt_tools",
        "oth_eqt_tools_p_shr",
        "lending_funds",
        "acc_receivable",
        "st_fin_payable",
        "payables",
        "hfs_assets",
        "hfs_sales",
        "cost_fin_assets",
        "fair_value_fin_assets",
        "contract_assets",
        "contract_liab",
        "accounts_receiv_bill",
        "accounts_pay",
        "oth_rcv_total",
        "fix_assets_total",
        "cip_total",
        "oth_pay_total",
        "long_pay_total",
        "debt_invest",
        "oth_debt_invest",
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
                    df_new = pro.balancesheet(**{
                        "ts_code": row.ts_code,
                        "ann_date": "",
                        "f_ann_date": "",
                        "start_date": start,
                        "end_date": end,
                        "period": "",
                        "report_type": "",
                        "comp_type": "",
                        "limit": "",
                        "offset": ""
                    }, fields=fields)
                    # 将 'trade_date' 转换为 datetime 格式
                    df_new["end_date"] = pd.to_datetime(df_new["end_date"])
                    df_new["f_ann_date"] = pd.to_datetime(df_new["f_ann_date"])
                    df_new["ann_date"] = pd.to_datetime(df_new["ann_date"])
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