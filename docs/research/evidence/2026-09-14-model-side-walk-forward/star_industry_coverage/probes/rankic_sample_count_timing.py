"""Date the gap: how wide was yeslab_357's cross-section on 2023-07-05 when RankIC was computed?

The repair tables are empty (factor_repair_audit and factor_repair_quarantine both hold 0 rows), so
no repair overlay rewrote these factors, and the known-formula recompute path covers only
yeslab_391 / yeslab_433. What is left is an ordering question that only a dated record can answer:
did the STAR-Market rows for yeslab_357 / yeslab_471 disappear in the 2026-09-03 factor_daily
rebuild, or were they already absent upstream before that rebuild and backfilled since?

`factor_rank_ic_daily` carries a per-factor per-day sample_count and its own created_at, so it is a
dated measurement of how wide each factor's cross-section was at the moment RankIC was computed.
Read on 2023-07-05 against the same day's controls, the count is directly comparable with the
panel's 3627 valid-label rows: a count near 3627 means the factor covered the day, a count near
3273 means it was already missing the 354 rows the panel reports. created_at says which vintage of
factor_daily that measurement describes.

SAFETY: SELECT-only against dfs://QuantResearchV2Test. No writes, no weights. Run from research_v2/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

import pandas as pd  # noqa: E402
from walk_forward_common import TARGET_DATABASE, new_holder, query  # noqa: E402

OUT = Path("/data/zhangyudong/qrv2_research_artifacts/rankic_sample_count_timing_probe.json")
FACTORS = ("yeslab_357", "yeslab_471", "yeslab_381", "yeslab_464", "yeslab_484", "yeslab_206", "yeslab_0")
DAYS = ("2022.07.29", "2023.07.05", "2023.09.15", "2023.10.09")


def main() -> None:
    holder = new_holder()
    columns = list(
        _run(holder, f"select top 1 * from loadTable('{TARGET_DATABASE}','factor_rank_ic_daily')").columns
    )
    print(f"factor_rank_ic_daily columns: {columns}", flush=True)
    keep = [c for c in ("signal_date", "factor_id", "sample_count", "rank_ic", "created_at", "data_version") if c in columns]
    ids = ", ".join(f"'{f}'" for f in FACTORS)
    result: dict = {"columns": columns, "days": {}}
    for day in DAYS:
        frame = _run(
            holder,
            f"select {', '.join(keep)} from loadTable('{TARGET_DATABASE}','factor_rank_ic_daily') "
            f"where factor_id in [{ids}] and signal_date between date('{day}') and date('{day}')",
        )
        print(f"\n=== {day} ({len(frame)} rows) ===", flush=True)
        entries = []
        if not frame.empty:
            for row in frame.sort_values(["factor_id", "created_at"]).itertuples(index=False):
                entry = {field: str(getattr(row, field)) for field in keep}
                entries.append(entry)
                print(
                    f"  {entry['factor_id']:<11} sample_count={entry.get('sample_count'):>6} "
                    f"created_at={entry.get('created_at', '')[:19]} "
                    f"version={entry.get('data_version', '')[:12]}",
                    flush=True,
                )
        result["days"][day] = entries

    # One monthly sweep so the September step is visible in the same vintage terms.
    frame = _run(
        holder,
        f"select {', '.join(keep)} from loadTable('{TARGET_DATABASE}','factor_rank_ic_daily') "
        "where factor_id in ['yeslab_357', 'yeslab_471', 'yeslab_381'] "
        "and signal_date between date('2023.08.01') and date('2023.10.31')",
    )
    if not frame.empty:
        frame["month"] = pd.to_datetime(frame["signal_date"]).dt.strftime("%Y-%m")
        frame["vintage"] = pd.to_datetime(frame["created_at"]).dt.strftime("%Y-%m-%d")
        pivot = frame.groupby(["month", "factor_id", "vintage"])["sample_count"].agg(["mean", "min", "max", "size"])
        result["monthly"] = [
            {"month": str(month), "factor_id": str(fid), "vintage": str(vintage),
             "mean_sample_count": float(stats["mean"]), "min": int(stats["min"]),
             "max": int(stats["max"]), "days": int(stats["size"])}
            for (month, fid, vintage), stats in pivot.iterrows()
        ]
        print("\n=== monthly sample_count by RankIC vintage ===", flush=True)
        for entry in result["monthly"]:
            print(
                f"  {entry['month']} {entry['factor_id']:<11} vintage={entry['vintage']} "
                f"mean={entry['mean_sample_count']:>8.1f} min={entry['min']:>5} "
                f"max={entry['max']:>5} days={entry['days']:>3}",
                flush=True,
            )

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {OUT}", flush=True)


def _run(holder: dict, sql: str) -> pd.DataFrame:
    ok, frame = query(holder, sql)
    if not ok or not isinstance(frame, pd.DataFrame):
        raise RuntimeError(f"query failed: {sql}")
    return frame


if __name__ == "__main__":
    main()
