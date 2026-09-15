#!/usr/bin/env python3
"""For every residual record, count the absent days inside its trailing 60-trading-day window.

`probe_206_absence_cause.py` established that the residual securities are seasoned (median 2320 trading
days since listing) and were trading normally on the day they were judged to lack yeslab_206, which rules
out both IPO seasoning and same-day suspension. What is left is a gap EARLIER in the window that the
factor's 60-day rolling statistic has to span. This counts, for each residual record, how many of the 60
trading days ending on that day have no row at all for that security in the local cache, and how many have
a zero volume. Absent rows are how a suspension appears in tushare's daily table, so the two counts
separate "no trading day" from "traded nothing".

Reads one existing measurement file plus the local caches. No database, no weights, no repository import.
The control that makes this number meaningful is `probe_206_hole_control.py`, which measures the same
statistic on securities that DO carry all nine factors; run them together.

Usage: python probe_206_history_hole.py --absence <probe_206_absence_cause.json> --out <json>
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

CACHE = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/artifacts/tushare_cache")
WINDOW = 60


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--absence", type=Path, required=True)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    d = json.loads(args.absence.read_text())
    all_dates = sorted({pd.Timestamp(p.stem.replace("daily_", "")) for p in CACHE.glob("daily_[0-9]*.json")})
    pos = {t: i for i, t in enumerate(all_dates)}
    memo: dict[pd.Timestamp, tuple[set, dict]] = {}

    def daysets(t: pd.Timestamp):
        if t not in memo:
            p = CACHE / f"daily_{t:%Y%m%d}.json"
            if not p.exists():
                memo[t] = (set(), {})
            else:
                raw = json.loads(p.read_text())
                recs = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
                df = pd.DataFrame(recs)
                df["ts_code"] = df["ts_code"].astype(str)
                df = df.drop_duplicates("ts_code", keep="last")
                v = pd.to_numeric(df["vol"], errors="coerce")
                memo[t] = (set(df.ts_code), dict(zip(df.ts_code, v)))
        return memo[t]

    rows = []
    for r in d["per_day"]:
        dt = pd.Timestamp(r["day"])
        i = pos[dt]
        win = all_dates[max(0, i - args.window + 1):i + 1]
        sets = [daysets(t) for t in win]
        for x in r["records"]:
            c = x["ts_code"]
            absent = sum(1 for s, _ in sets if c not in s)
            zero = sum(1 for s, v in sets if c in s and (v.get(c) or 0) == 0)
            rows.append({"day": r["day"], "ts_code": c, "missing": x["missing"],
                         "window_trading_days": len(win), "rows_absent_in_window": absent,
                         "zero_volume_days_in_window": zero,
                         "age_trading_days": x["trading_days_since_listing"]})
        print(f"  {r['day']}: {len(r['records'])} records", flush=True)

    df = pd.DataFrame(rows)
    df["sig"] = df.missing.apply(lambda m: "+".join(x.replace("yeslab_", "") for x in m))
    by_sig = df.groupby("sig").agg(
        n=("ts_code", "size"),
        share_with_a_hole=("rows_absent_in_window", lambda s: float((s > 0).mean())),
        median_absent=("rows_absent_in_window", "median"),
        max_absent=("rows_absent_in_window", "max"),
        share_with_zero_volume=("zero_volume_days_in_window", lambda s: float((s > 0).mean())),
    ).sort_values("n", ascending=False)

    share = float((df.rows_absent_in_window > 0).mean())
    out = {
        "question": "Do the residual securities lack yeslab_206 because their trailing 60-trading-day "
                    "history has a hole?",
        "window_trading_days": args.window,
        "records": len(df), "distinct_securities": int(df.ts_code.nunique()), "days": int(df.day.nunique()),
        "records_with_an_absent_row_in_the_window": int((df.rows_absent_in_window > 0).sum()),
        "records_with_a_zero_volume_day_in_the_window": int((df.zero_volume_days_in_window > 0).sum()),
        "share_with_an_absent_row": share,
        "absent_rows_in_window": {"min": int(df.rows_absent_in_window.min()),
                                  "median": float(df.rows_absent_in_window.median()),
                                  "p90": float(df.rows_absent_in_window.quantile(0.9)),
                                  "max": int(df.rows_absent_in_window.max())},
        "by_missing_signature": by_sig.reset_index().to_dict("records"),
        "verdict": None,
        "note": "This measures the residual side only. Without probe_206_hole_control.py's control group the "
                "percentage below is uninformative -- a 100% hit rate means nothing if complete-window "
                "securities also have holes.",
        "database_writes": False, "weights_written": False, "cache_writes": False,
    }
    out["verdict"] = (
        f"{out['records_with_an_absent_row_in_the_window']} of {len(df)} residual records "
        f"({share:.1%}) have at least one absent row in the trailing {args.window}-trading-day window "
        f"(median {out['absent_rows_in_window']['median']:.0f}, max {out['absent_rows_in_window']['max']}), "
        f"while {out['records_with_a_zero_volume_day_in_the_window']} have a zero-volume day -- so the gaps are "
        f"absent trading days, not days that traded nothing.")
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n" + out["verdict"])
    print(by_sig.head(8).to_string())


if __name__ == "__main__":
    main()
