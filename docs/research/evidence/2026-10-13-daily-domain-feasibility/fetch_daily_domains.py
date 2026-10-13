"""Fetch three daily non-price-volume panels (2022-2025) for a feasibility screen.

Sources (all daily, full-market, verified accessible on 2026-10-13):
  moneyflow     per-stock main/large/medium/small order flows      (published evening of T)
  cyq_perf      chip-distribution cost quantiles + winner_rate     (published evening of T)
  margin_detail per-stock margin financing/securities lending      (published T+1 ~08:30)

Window 2022-01-01..2025-12-31 = the four folds' TEST years (F1-F4). 2026 is never requested.
Writes raw/<source>_<year>.parquet, resumable at year granularity, append-only progress log.
"""
from __future__ import annotations

import re
import time
import pathlib
import sys

import pandas as pd

OUT = pathlib.Path(__file__).resolve().parent
RAW = OUT / "raw"
RAW.mkdir(parents=True, exist_ok=True)
TOKEN = re.search(r"QRV2_TUSHARE_TOKEN=(\S+)",
                  pathlib.Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/.env").read_text()).group(1)
SOURCES = ("moneyflow", "cyq_perf", "margin_detail")
YEARS = (2022, 2023, 2024, 2025)
SLEEP = 0.15


def log(msg: str) -> None:
    line = f"{pd.Timestamp.now(tz='Asia/Shanghai').isoformat()} {msg}"
    print(line, flush=True)
    with (OUT / "logs" / "fetch_progress.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def trading_days(pro, year: int) -> list[str]:
    """Trading days for `year`, preferring the already-fetched moneyflow panel as the calendar.

    pro.trade_cal intermittently returns an empty frame (observed 2026-10-13: KeyError 'cal_date'
    killed the run), so it is only a fallback and is retried.
    """
    ref = RAW / f"moneyflow_{year}.parquet"
    if ref.exists():
        return sorted(pd.read_parquet(ref, columns=["trade_date"])["trade_date"].astype(str).unique().tolist())
    for attempt in range(6):
        cal = pro.trade_cal(exchange="SSE", start_date=f"{year}0101", end_date=f"{year}1231", is_open="1")
        if cal is not None and "cal_date" in getattr(cal, "columns", []) and len(cal):
            return sorted(cal["cal_date"].astype(str).tolist())
        log(f"  trade_cal {year} empty (attempt {attempt}), retrying")
        time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"trade_cal returned no rows for {year}")


def fetch_day(pro, source: str, day: str, retries: int = 6) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            return getattr(pro, source)(trade_date=day)
        except Exception as exc:  # rate limit / transient network
            wait = min(2 ** attempt, 30)
            log(f"  retry {source} {day} attempt={attempt} wait={wait}s err={str(exc)[:90]}")
            time.sleep(wait)
    raise RuntimeError(f"gave up on {source} {day}")


def main() -> int:
    import tushare as ts
    pro = ts.pro_api(TOKEN)
    for source in SOURCES:
        for year in YEARS:
            dest = RAW / f"{source}_{year}.parquet"
            if dest.exists():
                log(f"[skip] {dest.name} already present")
                continue
            days = trading_days(pro, year)
            t0, frames = time.time(), []
            for i, day in enumerate(days, 1):
                frames.append(fetch_day(pro, source, day))
                time.sleep(SLEEP)
                if i % 60 == 0:
                    log(f"  {source} {year}: {i}/{len(days)} days, {sum(len(f) for f in frames):,} rows, {time.time()-t0:.0f}s")
            df = pd.concat(frames, ignore_index=True)
            df.to_parquet(dest, index=False)
            log(f"[done] {source} {year}: {len(df):,} rows, {df['trade_date'].nunique()} days, "
                f"{time.time()-t0:.0f}s -> {dest.name}")
    log("ALL_FETCH_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
