"""Independently compute expected values for the B8 real golden sample.

This script deliberately does NOT call the production builder or cleaner: it
re-implements the cleaning/history rules with plain Python so the golden test
compares two independent computations.  It also extends the extract past the
window end to derive mature T+1..T+6 adjusted-open labels for window signals.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import date
from pathlib import Path

import dolphindb as ddb
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "tests" / "golden" / "data"
EXTENSION_END = date(2025, 12, 15)


def _connect() -> ddb.session:
    session = ddb.session()
    connected = session.connect(
        os.getenv("QRV2_DDB_HOST", "127.0.0.1"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", ""),
        os.getenv("QRV2_DDB_PASSWORD", ""),
    )
    if not connected:
        raise SystemExit("cannot connect to the configured DolphinDB source database")
    return session


def independent_clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Hand-written cleaning rules mirroring the frozen contract, not the code.

    Row validation, ordering, deduplication and adjusted_open are implemented
    here from scratch.  The rolling history features share pandas' rolling mean
    as a documented numeric primitive (bit-level agreement with the production
    pipeline) while the surrounding assembly stays independent.
    """
    rows = []
    for record in frame.sort_values(["ts_code", "trade_date"], kind="mergesort").itertuples(
        index=False
    ):
        open_price = float(record.open)
        high = float(record.high)
        low = float(record.low)
        close = float(record.close)
        vol = float(record.vol)
        amount = float(record.amount)
        adj_factor = float(record.adj_factor)
        if min(open_price, high, low, close) <= 0:
            raise SystemExit(f"OHLC must be positive for {record.ts_code} {record.trade_date}")
        if high < max(open_price, low, close) or low > min(open_price, high, close):
            raise SystemExit(f"OHLC relationship invalid for {record.ts_code} {record.trade_date}")
        if vol < 0 or amount < 0 or adj_factor <= 0:
            raise SystemExit(f"vol/amount/adj invalid for {record.ts_code} {record.trade_date}")
        rows.append(
            {
                "ts_code": record.ts_code,
                "trade_date": record.trade_date,
                "adjusted_open": open_price * adj_factor,
                "valid_trade_day": (vol > 0) and (amount > 0),
                "vol": vol,
                "amount": amount,
            }
        )
    cleaned = pd.DataFrame(rows)
    cleaned = cleaned.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(
        drop=True
    )
    grouped = cleaned.groupby("ts_code", sort=False)
    cleaned["valid_days_20"] = grouped["valid_trade_day"].transform(
        lambda values: values.rolling(20, min_periods=1).sum()
    ).astype("int64")
    valid_amount = cleaned["amount"].where(cleaned["valid_trade_day"])
    cleaned["adv20"] = valid_amount.groupby(cleaned["ts_code"], sort=False).transform(
        lambda values: values.rolling(20, min_periods=15).mean()
    )
    return cleaned


def main() -> int:
    manifest = json.loads((DATA_DIR / "yeslab_market_50x60_manifest.json").read_text())
    extract = pd.read_parquet(DATA_DIR / "yeslab_market_50x60.parquet")
    window_days = [date.fromisoformat(day) for day in manifest["window"]["trading_days"]]

    cleaned = independent_clean(extract)

    codes = sorted(extract["ts_code"].unique())
    code_list = ",".join(f"'{code}'" for code in codes)
    session = _connect()
    extension = session.run(
        "select ts_code, trade_date, open, adj_factor "
        "from loadTable('dfs://YesLab', 'stock_factor_daily') "
        f"where trade_date between {window_days[0].strftime('%Y.%m.%d')} "
        f"and {EXTENSION_END.strftime('%Y.%m.%d')} and ts_code in [{code_list}]"
    )
    extension["trade_date"] = pd.to_datetime(extension["trade_date"]).dt.date

    calendar_rows = session.run(
        "select cal_date from loadTable('dfs://YesLab', 'trade_calendar') "
        "where exchange='SSE' and is_open=1 and cal_date between 2025.08.01 and 2025.12.31 "
        "order by cal_date"
    )
    calendar = [pd.Timestamp(value).date() for value in calendar_rows["cal_date"]]
    day_index = {day: position for position, day in enumerate(calendar)}
    prices: dict[tuple[str, date], float] = {
        (row.ts_code, row.trade_date): float(row.open) * float(row.adj_factor)
        for row in extension.itertuples(index=False)
    }

    label_rows = []
    member_pairs = sorted(
        {
            (record.trade_date, record.ts_code)
            for record in extract.loc[:, ["trade_date", "ts_code"]].itertuples(index=False)
        }
    )
    for signal_date, code in member_pairs:
        if signal_date not in day_index:
            continue
        position = day_index[signal_date]
        if position + 6 >= len(calendar):
            continue
        entry_date = calendar[position + 1]
        exit_date = calendar[position + 6]
        entry_price = prices.get((code, entry_date))
        exit_price = prices.get((code, exit_date))
        if entry_price is None or exit_price is None or entry_price <= 0:
            continue
        label_rows.append(
            {
                "signal_date": signal_date.isoformat(),
                "ts_code": code,
                "entry_date": entry_date.isoformat(),
                "exit_date": exit_date.isoformat(),
                "raw_return_5d": exit_price / entry_price - 1.0,
            }
        )
    labels = pd.DataFrame(label_rows)

    cleaned_out = cleaned.sort_values(
        ["trade_date", "ts_code"], kind="mergesort"
    ).reset_index(drop=True)
    clean_payload = [
        {
            "ts_code": row.ts_code,
            "trade_date": row.trade_date.isoformat()
            if hasattr(row.trade_date, "isoformat")
            else str(row.trade_date),
            "adjusted_open": round(float(row.adjusted_open), 9),
            "valid_trade_day": bool(row.valid_trade_day),
            "valid_days_20": int(row.valid_days_20),
            "adv20": None if pd.isna(row.adv20) else round(float(row.adv20), 3),
        }
        for row in cleaned_out.itertuples(index=False)
    ]
    clean_hash = hashlib.sha256(
        json.dumps(clean_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    label_payload = labels.to_dict(orient="records")
    label_hash = hashlib.sha256(
        json.dumps(label_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    expected = {
        "description": (
            "Independently recomputed cleaning and label expectations for "
            "yeslab_market_50x60 (no production builder involved). Canonical "
            "floats: adjusted_open rounded to 9 decimals, adv20 to 3 decimals "
            "(ULP absorption at currency magnitudes)."
        ),
        "clean_rows_sha256": clean_hash,
        "clean_row_count": len(clean_payload),
        "label_rows_sha256": label_hash,
        "label_row_count": len(label_payload),
        "window_days": len(window_days),
        "calendar": [day.isoformat() for day in calendar],
    }
    out_path = DATA_DIR / "expected_yeslab_market_50x60.json"
    out_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n")
    extension_sorted = extension.sort_values(
        ["ts_code", "trade_date"], kind="mergesort"
    ).reset_index(drop=True)
    extension_sorted.to_parquet(
        DATA_DIR / "yeslab_label_extension_50x60.parquet", engine="pyarrow", index=False
    )
    print(json.dumps(expected, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
