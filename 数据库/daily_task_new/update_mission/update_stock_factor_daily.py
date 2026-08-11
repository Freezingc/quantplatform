from tqdm import tqdm
import dolphindb as ddb
import pandas as pd
import time
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import tushare as ts

def update_stock_factor_daily():
    load_dotenv()  # 默认加载当前目录的 .env
    username = os.getenv('dolphinDB_USERNAME')  # 从环境变量读取
    password = os.getenv('dolphinDB_PASSWORD')

    # 连接 DolphinDB 数据库
    s = ddb.session()
    s.connect("10.176.55.211", 8848, username, password)

    # 连接 Tushare API
    pro = ts.pro_api(os.getenv('TUSHARE_TOKEN'))

    # 获取数据库中已有数据的最大日期
    max_date_query = '''
        db = database("dfs://YesLab")
        t = loadTable(db,'stock_factor_daily')
        select max(trade_date) from t
    '''
    max_date_result = s.run(max_date_query)

    # 如果数据库中有数据，获取最大日期并增加一天
    if not pd.isna(max_date_result['max_trade_date'].iloc[0]):
        max_date = pd.to_datetime(max_date_result['max_trade_date'][0])
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

    # 获取交易日历，确保从最大日期之后的日期开始
    time_list = pro.trade_cal(exchange='', is_open=1, 
                            start_date = start_date, 
                            end_date = end_date, 
                            fields ='cal_date')
    # print(time_list)
    # 拉取并写入数据

    fields=[
        "ts_code",
        "trade_date",
        "open",
        "open_hfq",
        "open_qfq",
        "high",
        "high_hfq",
        "high_qfq",
        "low",
        "low_hfq",
        "low_qfq",
        "close",
        "close_hfq",
        "close_qfq",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "ps_ttm",
        "dv_ratio",
        "dv_ttm",
        "total_share",
        "float_share",
        "free_share",
        "total_mv",
        "circ_mv",
        "adj_factor",
        "asi_bfq",
        "asi_hfq",
        "asi_qfq",
        "asit_bfq",
        "asit_hfq",
        "asit_qfq",
        "atr_bfq",
        "atr_hfq",
        "atr_qfq",
        "bbi_bfq",
        "bbi_hfq",
        "bbi_qfq",
        "bias1_bfq",
        "bias1_hfq",
        "bias1_qfq",
        "bias2_bfq",
        "bias2_hfq",
        "bias2_qfq",
        "bias3_bfq",
        "bias3_hfq",
        "bias3_qfq",
        "boll_lower_bfq",
        "boll_lower_hfq",
        "boll_lower_qfq",
        "boll_mid_bfq",
        "boll_mid_hfq",
        "boll_mid_qfq",
        "boll_upper_bfq",
        "boll_upper_hfq",
        "boll_upper_qfq",
        "brar_ar_bfq",
        "brar_ar_hfq",
        "brar_ar_qfq",
        "brar_br_bfq",
        "brar_br_hfq",
        "brar_br_qfq",
        "cci_bfq",
        "cci_hfq",
        "cci_qfq",
        "cr_bfq",
        "cr_hfq",
        "cr_qfq",
        "dfma_dif_bfq",
        "dfma_dif_hfq",
        "dfma_dif_qfq",
        "dfma_difma_bfq",
        "dfma_difma_hfq",
        "dfma_difma_qfq",
        "dmi_adx_bfq",
        "dmi_adx_hfq",
        "dmi_adx_qfq",
        "dmi_adxr_bfq",
        "dmi_adxr_hfq",
        "dmi_adxr_qfq",
        "dmi_mdi_bfq",
        "dmi_mdi_hfq",
        "dmi_mdi_qfq",
        "dmi_pdi_bfq",
        "dmi_pdi_hfq",
        "dmi_pdi_qfq",
        "downdays",
        "updays",
        "dpo_bfq",
        "dpo_hfq",
        "dpo_qfq",
        "madpo_bfq",
        "madpo_hfq",
        "madpo_qfq",
        "ema_bfq_10",
        "ema_bfq_20",
        "ema_bfq_250",
        "ema_bfq_30",
        "ema_bfq_5",
        "ema_bfq_60",
        "ema_bfq_90",
        "ema_hfq_10",
        "ema_hfq_20",
        "ema_hfq_250",
        "ema_hfq_30",
        "ema_hfq_5",
        "ema_hfq_60",
        "ema_hfq_90",
        "ema_qfq_10",
        "ema_qfq_20",
        "ema_qfq_250",
        "ema_qfq_30",
        "ema_qfq_5",
        "ema_qfq_60",
        "ema_qfq_90",
        "emv_bfq",
        "emv_hfq",
        "emv_qfq",
        "maemv_bfq",
        "maemv_hfq",
        "maemv_qfq",
        "expma_12_bfq",
        "expma_12_hfq",
        "expma_12_qfq",
        "expma_50_bfq",
        "expma_50_hfq",
        "expma_50_qfq",
        "kdj_bfq",
        "kdj_hfq",
        "kdj_qfq",
        "kdj_d_bfq",
        "kdj_d_hfq",
        "kdj_d_qfq",
        "kdj_k_bfq",
        "kdj_k_hfq",
        "kdj_k_qfq",
        "ktn_down_bfq",
        "ktn_down_hfq",
        "ktn_down_qfq",
        "ktn_mid_bfq",
        "ktn_mid_hfq",
        "ktn_mid_qfq",
        "ktn_upper_bfq",
        "ktn_upper_hfq",
        "ktn_upper_qfq",
        "lowdays",
        "topdays",
        "ma_bfq_10",
        "ma_bfq_20",
        "ma_bfq_250",
        "ma_bfq_30",
        "ma_bfq_5",
        "ma_bfq_60",
        "ma_bfq_90",
        "ma_hfq_10",
        "ma_hfq_20",
        "ma_hfq_250",
        "ma_hfq_30",
        "ma_hfq_5",
        "ma_hfq_60",
        "ma_hfq_90",
        "ma_qfq_10",
        "ma_qfq_20",
        "ma_qfq_250",
        "ma_qfq_30",
        "ma_qfq_5",
        "ma_qfq_60",
        "ma_qfq_90",
        "macd_bfq",
        "macd_hfq",
        "macd_qfq",
        "macd_dea_bfq",
        "macd_dea_hfq",
        "macd_dea_qfq",
        "macd_dif_bfq",
        "macd_dif_hfq",
        "macd_dif_qfq",
        "mass_bfq",
        "mass_hfq",
        "mass_qfq",
        "ma_mass_bfq",
        "ma_mass_hfq",
        "ma_mass_qfq",
        "mfi_bfq",
        "mfi_hfq",
        "mfi_qfq",
        "mtm_bfq",
        "mtm_hfq",
        "mtm_qfq",
        "mtmma_bfq",
        "mtmma_hfq",
        "mtmma_qfq",
        "obv_bfq",
        "obv_hfq",
        "obv_qfq",
        "psy_bfq",
        "psy_hfq",
        "psy_qfq",
        "psyma_bfq",
        "psyma_hfq",
        "psyma_qfq",
        "roc_bfq",
        "roc_hfq",
        "roc_qfq",
        "maroc_bfq",
        "maroc_hfq",
        "maroc_qfq",
        "rsi_bfq_12",
        "rsi_bfq_24",
        "rsi_bfq_6",
        "rsi_hfq_12",
        "rsi_hfq_24",
        "rsi_hfq_6",
        "rsi_qfq_12",
        "rsi_qfq_24",
        "rsi_qfq_6",
        "taq_down_bfq",
        "taq_down_hfq",
        "taq_down_qfq",
        "taq_mid_bfq",
        "taq_mid_hfq",
        "taq_mid_qfq",
        "taq_up_bfq",
        "taq_up_hfq",
        "taq_up_qfq",
        "trix_bfq",
        "trix_hfq",
        "trix_qfq",
        "trma_bfq",
        "trma_hfq",
        "trma_qfq",
        "vr_bfq",
        "vr_hfq",
        "vr_qfq",
        "wr_bfq",
        "wr_hfq",
        "wr_qfq",
        "wr1_bfq",
        "wr1_hfq",
        "wr1_qfq",
        "xsii_td1_bfq",
        "xsii_td1_hfq",
        "xsii_td1_qfq",
        "xsii_td2_bfq",
        "xsii_td2_hfq",
        "xsii_td2_qfq",
        "xsii_td3_bfq",
        "xsii_td3_hfq",
        "xsii_td3_qfq",
        "xsii_td4_bfq",
        "xsii_td4_hfq",
        "xsii_td4_qfq"
    ]
    df = pd.DataFrame(columns = fields)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for date in tqdm(time_list["cal_date"]):
        error_count = 0  # 初始化错误计数器
        max_errors = 10  # 最大允许错误次数
        while True:
            try:
                df_new = pro.stk_factor_pro(**{
                    "ts_code": "",
                    "start_date": "",
                    "end_date": "",
                    "trade_date": date,
                    "limit": "",
                    "offset": ""
                }, fields=fields)
                # 将 'trade_date' 转换为 datetime 格式
                df_new["trade_date"] = pd.to_datetime(df_new["trade_date"])
                # 向数据库中追加数据
                df = pd.concat([df, df_new], ignore_index=True)
                break
                
            except Exception as e:
                error_count += 1  # 错误计数加1
                print(f"发生异常: {e}, 1min后重新尝试... (错误次数: {error_count}/{max_errors})")
                
                if error_count >= max_errors:
                    print(f"已达到最大错误次数 {max_errors}，停止尝试...")
                    import sys
                    sys.exit(1)
                    
                time.sleep(1)
    return "", df