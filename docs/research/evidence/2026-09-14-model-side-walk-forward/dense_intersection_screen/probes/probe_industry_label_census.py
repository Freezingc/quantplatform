"""A complete census of the PIT industry label dependency: every day it drops, not just the one we found.

Three failures of this dependency are already documented, each found by chasing something else: a
21-month board-wide absence of STAR-Market membership (2022-01..2023-09, which cost 24 OOS days), a
31-day incomplete left edge (2014-01-02..2014-02-20, 17 of 28 industries until 912 securities
register on 2014-02-21), and a one-day hole on PIT_CUTOVER itself (2021-12-10, where 535 memberships
expire under a right-exclusive interval while their 393 replacements start on 2021-12-13).

Three anecdotes found by accident is not a census. This walks every business day from 2014 to 2026
for BOTH halves of the cache and reports every day the active membership drops, so the question "is
the cutover hole unique?" has an answer, and so a future fold whose test year contains one of these
dates can be checked against a list instead of against luck. Only the OOS window can move a published
gate, so that window is reported separately.

Reads local JSON only. No database, no weights, no writes outside --out.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DEFAULT = Path(
    "/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/artifacts/tushare_cache"
)
OUT_DEFAULT = Path("/data/zhangyudong/qrv2_research_artifacts/industry_label_census.json")
SOURCES = {
    "SW2014": "pit_industry_sw2014_members_*_SI.json",
    "SW2021": "pit_industry_members_*_SI.json",
}
STAR_PREFIXES = ("688", "689")
# A drop this large is worth naming. The 2021-12-10 hole is -378 members; the 2014-02-21 step is +912.
DROP_THRESHOLD = 50
OOS_WINDOW = ("2022-01-01", "2025-12-31")


def load(cache: Path, pattern: str) -> tuple[pd.DataFrame, dict]:
    files = sorted(glob.glob(str(cache / pattern)))
    if not files:
        raise SystemExit(f"no files match {pattern} under {cache}")
    digest = hashlib.sha256()
    frames = []
    for path in files:
        raw = Path(path).read_bytes()
        digest.update(hashlib.sha256(raw).digest())
        frame = pd.read_json(Path(path), orient="table")
        frames.append(frame)
    members = pd.concat(frames, ignore_index=True)
    members["con_code"] = members["con_code"].astype(str)
    members["in_parsed"] = pd.to_datetime(members["in_date"], errors="coerce")
    out = pd.to_datetime(members["out_date"], errors="coerce", format="%Y%m%d")
    if out.isna().all():
        out = pd.to_datetime(members["out_date"], errors="coerce")
    members["out_parsed"] = out
    return members, {
        "pattern": pattern,
        "files": len(files),
        "rows": int(len(members)),
        "combined_sha256": digest.hexdigest(),
    }


def daily_series(members: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Active members per day by sweep: active(d) = #(in_date <= d) - #(out_date <= d).

    The interval is right-exclusive, which is exactly why 535 records with out_date 2021-12-10 are
    already gone on that day; the arithmetic here uses the same rule the provider does.
    """
    ins = np.sort(members["in_parsed"].dropna().to_numpy())
    outs = np.sort(members["out_parsed"].dropna().to_numpy())
    day_values = days.to_numpy()
    started = np.searchsorted(ins, day_values, side="right")
    ended = np.searchsorted(outs, day_values, side="right")
    star = members.loc[members["con_code"].str.startswith(STAR_PREFIXES)]
    star_ins = np.sort(star["in_parsed"].dropna().to_numpy())
    star_outs = np.sort(star["out_parsed"].dropna().to_numpy())
    star_started = np.searchsorted(star_ins, day_values, side="right")
    star_ended = np.searchsorted(star_outs, day_values, side="right")
    frame = pd.DataFrame(
        {
            "day": days,
            "active": started - ended,
            "active_star": star_started - star_ended,
            "registrations": np.searchsorted(ins, day_values, side="right")
            - np.searchsorted(ins, day_values - np.timedelta64(1, "D"), side="right"),
            "expirations": np.searchsorted(outs, day_values, side="right")
            - np.searchsorted(outs, day_values - np.timedelta64(1, "D"), side="right"),
        }
    )
    frame["delta"] = frame["active"].diff().fillna(0).astype(int)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE_DEFAULT)
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--threshold", type=int, default=DROP_THRESHOLD)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()

    days = pd.bdate_range(args.start, args.end)
    result: dict = {
        "cache_dir": str(args.cache),
        "window": [args.start, args.end],
        "business_days": int(len(days)),
        "drop_threshold": args.threshold,
        "oos_window": list(OOS_WINDOW),
        "interval_rule": "active(d) = in_date <= d < out_date (right-exclusive)",
        "sources": {},
    }
    for label, pattern in SOURCES.items():
        members, provenance = load(args.cache, pattern)
        series = daily_series(members, days)
        drops = series.loc[series["delta"] <= -args.threshold]
        gains = series.loc[series["delta"] >= args.threshold]
        oos = series.loc[
            (series["day"] >= pd.Timestamp(OOS_WINDOW[0]))
            & (series["day"] <= pd.Timestamp(OOS_WINDOW[1]))
        ]
        oos_drops = oos.loc[oos["delta"] <= -args.threshold]
        entry = {
            "provenance": provenance,
            "active_min": int(series["active"].min()),
            "active_max": int(series["active"].max()),
            "active_first_day": int(series["active"].iloc[0]),
            "active_last_day": int(series["active"].iloc[-1]),
            "star_first_nonzero_day": (
                str(series.loc[series["active_star"] > 0, "day"].min().date())
                if (series["active_star"] > 0).any()
                else None
            ),
            "drop_days": [
                {
                    "day": str(row.day.date()),
                    "delta": int(row.delta),
                    "active": int(row.active),
                    "expirations": int(row.expirations),
                    "registrations": int(row.registrations),
                }
                for row in drops.itertuples(index=False)
            ],
            "gain_days": [
                {
                    "day": str(row.day.date()),
                    "delta": int(row.delta),
                    "active": int(row.active),
                    "registrations": int(row.registrations),
                }
                for row in gains.itertuples(index=False)
            ],
            "drops_inside_oos_window": [
                {
                    "day": str(row.day.date()),
                    "delta": int(row.delta),
                    "active": int(row.active),
                    "expirations": int(row.expirations),
                    "registrations": int(row.registrations),
                }
                for row in oos_drops.itertuples(index=False)
            ],
        }
        result["sources"][label] = entry
        print(f"\n=== {label} ({provenance['files']} files, {provenance['rows']} rows) ===")
        print(
            f"  active members: first day {entry['active_first_day']}, last day "
            f"{entry['active_last_day']}, min {entry['active_min']}, max {entry['active_max']}"
        )
        print(f"  first day with any STAR member: {entry['star_first_nonzero_day']}")
        print(f"  days with a drop >= {args.threshold}: {len(entry['drop_days'])}")
        for row in entry["drop_days"][:12]:
            print(
                f"    {row['day']}  delta={row['delta']:>6}  active={row['active']:>5} "
                f"(expirations={row['expirations']}, registrations={row['registrations']})"
            )
        print(f"  days with a gain >= {args.threshold}: {len(entry['gain_days'])}")
        for row in entry["gain_days"][:6]:
            print(f"    {row['day']}  delta={row['delta']:>6}  active={row['active']:>5}")
        print(f"  drops INSIDE the OOS window {OOS_WINDOW}: {len(entry['drops_inside_oos_window'])}")
        for row in entry["drops_inside_oos_window"][:12]:
            print(
                f"    {row['day']}  delta={row['delta']:>6}  active={row['active']:>5} "
                f"(expirations={row['expirations']}, registrations={row['registrations']})"
            )

    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
