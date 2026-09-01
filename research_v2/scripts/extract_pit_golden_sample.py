"""Extract the B8 golden sample: 50 securities x 60 trading days with boundary coverage.

Read-only against the YesLab source database.  The selection is deterministic:
liquidity-ranked candidates per board plus forced boundary securities (ST
transition, suspension, new listing, ex-rights).  Writes the market extract,
the selection manifest with boundary tags, and the source watermark record to
``tests/golden/data/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import dolphindb as ddb
import pandas as pd

WINDOW_START = date(2025, 9, 1)
WINDOW_TARGET_DAYS = 60
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "tests" / "golden" / "data"

FORCED_BOUNDARY = {
    # ST removal inside the window (ended 2025-09-07).
    "000711.SZ": ("st_transition",),
    # ST across the whole window (until 2025-12-22).
    "002586.SZ": ("st_active",),
    # New main-board listing inside the window (2025-09-05).
    "603370.SH": ("new_listing",),
    # New BJ listing inside the window (2025-09-08) with ex-rights history.
    "920100.BJ": ("new_listing", "ex_rights"),
    # Three adj_factor changes within the window across boards.
    "000062.SZ": ("ex_rights",),
    "300272.SZ": ("ex_rights",),
    "601133.SH": ("ex_rights",),
    "920007.BJ": ("ex_rights", "new_listing_recent"),
}

BOARD_QUOTAS = {
    "main_sh": 12,
    "main_sz": 12,
    "chinext": 8,
    "star": 8,
    "bj": 8,
}


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


def _window_days(session) -> list[date]:
    frame = session.run(
        "select cal_date from loadTable('dfs://YesLab', 'trade_calendar') "
        "where exchange='SSE' and is_open=1 "
        f"and cal_date >= {WINDOW_START.strftime('%Y.%m.%d')} order by cal_date"
    )
    days = [pd.Timestamp(value).date() for value in frame["cal_date"]]
    return days[:WINDOW_TARGET_DAYS]


def _board_of(ts_code: str, market: str) -> str | None:
    if ts_code.endswith(".BJ"):
        return "bj"
    if market == "创业板":
        return "chinext"
    if market == "科创板":
        return "star"
    if market == "主板":
        return "main_sh" if ts_code.endswith(".SH") else "main_sz"
    return None


def main() -> int:
    session = _connect()
    window_days = _window_days(session)
    if len(window_days) != WINDOW_TARGET_DAYS:
        raise SystemExit(f"expected {WINDOW_TARGET_DAYS} window days, got {len(window_days)}")
    window_lo = window_days[0].strftime("%Y.%m.%d")
    window_hi = window_days[-1].strftime("%Y.%m.%d")

    info = session.run(
        "select ts_code, name, market, list_date, list_status "
        "from loadTable('dfs://YesLab', 'stock_info_table')"
    )
    info["list_date"] = pd.to_datetime(info["list_date"])
    info["board"] = [
        _board_of(code, market)
        for code, market in zip(info["ts_code"], info["market"], strict=True)
    ]
    established = info.loc[
        (info["list_status"] == "L")
        & (info["board"].notna())
        & (info["list_date"] < pd.Timestamp(WINDOW_START))
    ]

    liquidity = session.run(
        "select ts_code, avg(amount) as avg_amount "
        "from loadTable('dfs://YesLab', 'stock_factor_daily') "
        f"where trade_date between {window_lo} and {window_hi} "
        "group by ts_code"
    )
    liquidity_index = dict(zip(liquidity["ts_code"], liquidity["avg_amount"], strict=True))

    selected: dict[str, tuple[str, ...]] = {
        code: tags for code, tags in FORCED_BOUNDARY.items() if code in set(info["ts_code"])
    }
    for board, quota in BOARD_QUOTAS.items():
        pool = established.loc[established["board"] == board, "ts_code"].tolist()
        ranked = sorted(pool, key=lambda code: (-liquidity_index.get(code, 0.0), code))
        for code in ranked:
            current = sum(
                1 for selected_code in selected if _board_matches(selected_code, board)
            )
            if current >= quota:
                break
            selected.setdefault(code, ("liquidity",))

    # Top up to exactly fifty securities with the most liquid remaining names so
    # the golden sample size is stable regardless of forced-boundary overlap.
    remaining = [
        code for code in established["ts_code"] if code not in selected
    ]
    ranked_remaining = sorted(
        remaining, key=lambda code: (-liquidity_index.get(code, 0.0), code)
    )
    for code in ranked_remaining:
        if len(selected) >= 50:
            break
        selected.setdefault(code, ("liquidity",))

    codes = sorted(selected)
    code_list = ",".join(f"'{code}'" for code in codes)
    extract = session.run(
        "select ts_code, trade_date, open, high, low, close, vol, amount, adj_factor "
        "from loadTable('dfs://YesLab', 'stock_factor_daily') "
        f"where trade_date between {window_lo} and {window_hi} and ts_code in [{code_list}]"
    )
    extract["trade_date"] = pd.to_datetime(extract["trade_date"]).dt.date
    extract = extract.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(
        drop=True
    )

    suspensions = session.run(
        "select ts_code, trade_date, suspend_type "
        "from loadTable('dfs://YesLab', 'stock_trading_suspension_info') "
        f"where trade_date between {window_lo} and {window_hi} and ts_code in [{code_list}]"
    )
    suspended_codes = set(suspensions["ts_code"].astype(str)) if len(suspensions) else set()

    names = session.run(
        "select ts_code, name, start_date, end_date "
        "from loadTable('dfs://YesLab', 'stock_used_name') "
        f"where ts_code in [{code_list}] and name like '%ST%'"
    )

    boundary_tags: dict[str, list[str]] = {code: list(tags) for code, tags in selected.items()}
    for code in suspended_codes:
        if code in boundary_tags and "suspension" not in boundary_tags[code]:
            boundary_tags[code].append("suspension")
    st_codes = set(names["ts_code"].astype(str)) if len(names) else set()
    for code in st_codes:
        if code in boundary_tags and "st_history" not in boundary_tags[code]:
            boundary_tags[code].append("st_history")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extract_path = OUTPUT_DIR / "yeslab_market_50x60.parquet"
    extract.to_parquet(extract_path, engine="pyarrow", index=False)
    content_sha256 = hashlib.sha256(extract_path.read_bytes()).hexdigest()

    manifest = {
        "extracted_at": datetime.now(UTC).isoformat(),
        "source_database": "dfs://YesLab",
        "source_table": "stock_factor_daily",
        "window": {
            "start": window_days[0].isoformat(),
            "end": window_days[-1].isoformat(),
            "trading_day_count": len(window_days),
            "trading_days": [day.isoformat() for day in window_days],
        },
        "security_count": len(codes),
        "row_count": len(extract),
        "boundary_tags": {code: sorted(tags) for code, tags in sorted(boundary_tags.items())},
        "content_sha256": content_sha256,
    }
    manifest_path = OUTPUT_DIR / "yeslab_market_50x60_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"window: {window_days[0]} .. {window_days[-1]} ({len(window_days)} days)")
    print(f"securities: {len(codes)}, rows: {len(extract)}")
    print(f"parquet sha256: {content_sha256}")
    tag_summary: dict[str, int] = {}
    for tags in boundary_tags.values():
        for tag in tags:
            tag_summary[tag] = tag_summary.get(tag, 0) + 1
    print("boundary coverage:", dict(sorted(tag_summary.items())))
    return 0


def _board_matches(ts_code: str, board: str) -> bool:
    if board == "bj":
        return ts_code.endswith(".BJ")
    if board == "main_sh":
        return ts_code.endswith(".SH") and not ts_code.startswith("688")
    if board == "star":
        return ts_code.startswith("688")
    if board == "chinext":
        return ts_code.startswith(("300", "301"))
    if board == "main_sz":
        return ts_code.endswith(".SZ") and not ts_code.startswith(("300", "301"))
    return False


if __name__ == "__main__":
    sys.exit(main())
