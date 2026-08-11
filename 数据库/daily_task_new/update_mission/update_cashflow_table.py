from tqdm import tqdm
import dolphindb as ddb
import pandas as pd
import time
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import tushare as ts

def update_cashflow_table():
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
        t = loadTable(db,'cashflow_table')
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
    print(date_ranges)

    fields = [
        "ts_code",
        "ann_date",
        "f_ann_date",
        "end_date",
        "comp_type",
        "report_type",
        "end_type",
        "net_profit",
        "finan_exp",
        "c_fr_sale_sg",
        "recp_tax_rends",
        "n_depos_incr_fi",
        "n_incr_loans_cb",
        "n_inc_borr_oth_fi",
        "prem_fr_orig_contr",
        "n_incr_insured_dep",
        "n_reinsur_prem",
        "n_incr_disp_tfa",
        "ifc_cash_incr",
        "n_incr_disp_faas",
        "n_incr_loans_oth_bank",
        "n_cap_incr_repur",
        "c_fr_oth_operate_a",
        "c_inf_fr_operate_a",
        "c_paid_goods_s",
        "c_paid_to_for_empl",
        "c_paid_for_taxes",
        "n_incr_clt_loan_adv",
        "n_incr_dep_cbob",
        "c_pay_claims_orig_inco",
        "pay_handling_chrg",
        "pay_comm_insur_plcy",
        "oth_cash_pay_oper_act",
        "st_cash_out_act",
        "n_cashflow_act",
        "oth_recp_ral_inv_act",
        "c_disp_withdrwl_invest",
        "c_recp_return_invest",
        "n_recp_disp_fiolta",
        "n_recp_disp_sobu",
        "stot_inflows_inv_act",
        "c_pay_acq_const_fiolta",
        "c_paid_invest",
        "n_disp_subs_oth_biz",
        "oth_pay_ral_inv_act",
        "n_incr_pledge_loan",
        "stot_out_inv_act",
        "n_cashflow_inv_act",
        "c_recp_borrow",
        "proc_issue_bonds",
        "oth_cash_recp_ral_fnc_act",
        "stot_cash_in_fnc_act",
        "free_cashflow",
        "c_prepay_amt_borr",
        "c_pay_dist_dpcp_int_exp",
        "incl_dvd_profit_paid_sc_ms",
        "oth_cashpay_ral_fnc_act",
        "stot_cashout_fnc_act",
        "n_cash_flows_fnc_act",
        "eff_fx_flu_cash",
        "n_incr_cash_cash_equ",
        "c_cash_equ_beg_period",
        "c_cash_equ_end_period",
        "c_recp_cap_contrib",
        "incl_cash_rec_saims",
        "uncon_invest_loss",
        "prov_depr_assets",
        "depr_fa_coga_dpba",
        "amort_intang_assets",
        "lt_amort_deferred_exp",
        "decr_deferred_exp",
        "incr_acc_exp",
        "loss_disp_fiolta",
        "loss_scr_fa",
        "loss_fv_chg",
        "invest_loss",
        "decr_def_inc_tax_assets",
        "incr_def_inc_tax_liab",
        "decr_inventories",
        "decr_oper_payable",
        "incr_oper_payable",
        "others",
        "im_net_cashflow_oper_act",
        "conv_debt_into_cap",
        "conv_copbonds_due_within_1y",
        "fa_fnc_leases",
        "im_n_incr_cash_equ",
        "net_dism_capital_add",
        "net_cash_rece_sec",
        "credit_impa_loss",
        "use_right_asset_dep",
        "oth_loss_asset",
        "end_bal_cash",
        "beg_bal_cash",
        "end_bal_cash_equ",
        "beg_bal_cash_equ",
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
                    df_new = pro.cashflow(**{
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