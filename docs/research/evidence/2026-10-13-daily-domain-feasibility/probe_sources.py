import os, re, sys, time
import pandas as pd
tok = re.search(r'QRV2_TUSHARE_TOKEN=(\S+)', open('/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/.env').read()).group(1)
import tushare as ts
pro = ts.pro_api(tok)
for api, kw in (("moneyflow", dict(trade_date="20240315")),
                ("cyq_perf", dict(trade_date="20240315")),
                ("margin_detail", dict(trade_date="20240315")),
                ("top_list", dict(trade_date="20240315")),
                ("report_rc", dict(start_date="20240301", end_date="20240315"))):
    try:
        t0 = time.time()
        df = getattr(pro, api)(**kw)
        print(f"{api:14s} OK rows={len(df):6d} cols={len(df.columns):2d} {time.time()-t0:.2f}s  {list(df.columns)[:9]}")
    except Exception as e:
        print(f"{api:14s} FAIL {str(e)[:130]}")
    time.sleep(0.4)
