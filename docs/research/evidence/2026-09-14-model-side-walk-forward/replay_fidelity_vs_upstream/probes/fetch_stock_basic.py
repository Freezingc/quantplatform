"""Fetch list_date AND delist_date for all three listing states, cached locally. Read-only vendor call."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ENV = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/.env")

ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
tok = [l.split("=", 1)[1].strip() for l in ENV.read_text().splitlines() if l.startswith("QRV2_TUSHARE_TOKEN=")][0]
import tushare as ts
pro = ts.pro_api(tok)
frames = [pro.stock_basic(list_status=s, fields="ts_code,name,list_date,delist_date,list_status")
          for s in ("L", "D", "P")]
df = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code").set_index("ts_code")
df.to_json(a.out, orient="table")
print("rows", len(df), "columns", list(df.columns))
print("with delist_date:", int(df["delist_date"].notna().sum()))
print(df["list_status"].value_counts().to_dict())
