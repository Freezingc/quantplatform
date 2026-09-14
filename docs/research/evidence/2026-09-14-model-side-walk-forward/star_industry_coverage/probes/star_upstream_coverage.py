"""Read-only fork test: is the STAR-Market gap in yeslab_357 / yeslab_471 upstream, or ours?

The published-panel analysis (docs/reports/2026-09-14-dense-factor-coverage-gap-and-the-24-days.md)
established *what* is missing: `行业内收益标准化绝对值` (yeslab_357, factor_data8, owner suyanning)
and `AttentionSpillover` (yeslab_471, factor_data10, owner zhangyudong) are jointly absent for
almost every 688/689 security until 2023-09, and their missing sets nest (357 subset of 471).
It could not say *where* the rows are lost, because it only read the published parquet.

This probe settles that with one decisive comparison per factor: read the SAME day from the SAME
source table the materialization reads, and ask whether the STAR codes are there at all.

  absent upstream                 -> producer-side gap; hand to the factor owners, nothing to fix here
  present upstream, value NULL    -> producer emitted the row but no number; still producer-side
  present upstream with a value   -> our materialization dropped a usable row; that would be a bug here

Controls matter more than the targets here. 357 and 471 live on different tables with different
owners, so a shared cause has to show up as "these two differ from their neighbours on the same
table": GP_012 (data8) and GP_095 / 羊群效应_CSAD波动率21 (data10) are nearly complete in the
published panel, and 484 shares an owner with 471. If a whole table had an outage the controls
would be empty too.

SAFETY: SELECT-only, `dfs://YesLab` only, no writes to any database, no artifact mutation, no
weight loading, and it does not import or edit anything the M2 driver uses at runtime beyond the
shared read-only connection helper. Run from research_v2/ so `scripts` and `src` resolve.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

import pandas as pd  # noqa: E402
from walk_forward_common import new_holder, query  # noqa: E402

OUT = Path("/data/zhangyudong/qrv2_research_artifacts/star_upstream_coverage_probe.json")

# (table, source factor_name, factor_id, role)
FACTORS = (
    ("factor_data8", "行业内收益标准化绝对值", "yeslab_357", "target-A"),
    ("factor_data8", "GP_012", "yeslab_381", "control-same-table-as-A"),
    ("factor_data10", "AttentionSpillover", "yeslab_471", "target-B"),
    ("factor_data10", "GP_095", "yeslab_464", "control-same-table-as-B"),
    ("factor_data10", "羊群效应_CSAD波动率21", "yeslab_484", "control-same-owner-as-B"),
)

# 2022-07-29 = the single dropped day of 2022; 2023-07-05 = worst day of 2023 (share 0.1097);
# 2023-10-09 = first month after the breakpoint, the day the gap is supposed to be gone.
SINGLE_DAYS = ("2022.07.29", "2023.07.05", "2023.10.09")

# Monthly cross-check that the October breakpoint is upstream too, not an artifact of materialization.
MONTHS = (
    ("2022.01.01", "2022.01.31"),
    ("2022.07.01", "2022.07.31"),
    ("2023.03.01", "2023.03.31"),
    ("2023.07.01", "2023.07.31"),
    ("2023.09.01", "2023.09.30"),
    ("2023.10.01", "2023.10.31"),
    ("2023.11.01", "2023.11.30"),
)

STAR_PREFIXES = ("688", "689")


def _select_range(holder: dict, table: str, name: str, start: str, end: str) -> pd.DataFrame:
    """Same shape as source_evaluation._query_source_range, minus the definition plumbing."""
    if "'" in name:
        raise ValueError(f"factor name would need escaping: {name!r}")
    sql = (
        "select ts_code, trade_date, value from "
        f"loadTable('dfs://YesLab', '{table}') "
        f"where factor_name = '{name}' "
        f"and trade_date between date('{start}') and date('{end}')"
    )
    ok, frame = query(holder, sql)
    if not ok or not isinstance(frame, pd.DataFrame):
        raise RuntimeError(f"query did not return a DataFrame for {table}/{name} {start}..{end}")
    return frame


def _summarize(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "rows": 0,
            "distinct_codes": 0,
            "null_values": 0,
            "star_rows": 0,
            "star_codes": 0,
            "star_rows_with_value": 0,
            "star_codes_with_value": 0,
            "non_star_codes": 0,
            "non_star_codes_with_value": 0,
            "trade_dates": 0,
        }
    code = frame["ts_code"].astype(str)
    value = pd.to_numeric(frame["value"], errors="coerce")
    is_star = code.str.startswith(STAR_PREFIXES)
    has_value = value.notna()
    star_with_value = is_star & has_value
    non_star_with_value = (~is_star) & has_value
    return {
        "rows": int(len(frame)),
        "distinct_codes": int(code.nunique()),
        "null_values": int((~has_value).sum()),
        "star_rows": int(is_star.sum()),
        "star_codes": int(code[is_star].nunique()),
        "star_rows_with_value": int(star_with_value.sum()),
        "star_codes_with_value": int(code[star_with_value].nunique()),
        "non_star_codes": int(code[~is_star].nunique()),
        "non_star_codes_with_value": int(code[non_star_with_value].nunique()),
        "trade_dates": int(pd.to_datetime(frame["trade_date"]).dt.date.nunique()),
    }


def _verdict(summary: dict) -> str:
    if summary["rows"] == 0:
        return "NO_ROWS_AT_ALL"
    if summary["star_codes"] == 0:
        return "STAR_ABSENT_UPSTREAM"
    if summary["star_codes_with_value"] == 0:
        return "STAR_PRESENT_BUT_NULL_UPSTREAM"
    return "STAR_PRESENT_WITH_VALUE_UPSTREAM"


def main() -> None:
    holder = new_holder()
    result: dict = {"safety": "select-only on dfs://YesLab", "single_day": {}, "monthly": {}}

    print("=== part 1: single-day fork test ===", flush=True)
    for day in SINGLE_DAYS:
        for table, name, factor_id, role in FACTORS:
            frame = _select_range(holder, table, name, day, day)
            summary = _summarize(frame)
            summary["verdict"] = _verdict(summary)
            result["single_day"].setdefault(day, {})[factor_id] = summary
            print(
                f"  {day} {factor_id:<11} {role:<26} rows={summary['rows']:>6} "
                f"codes={summary['distinct_codes']:>5} star={summary['star_codes']:>4} "
                f"(with value {summary['star_codes_with_value']:>4}) "
                f"nonstar_with_value={summary['non_star_codes_with_value']:>5} "
                f"-> {summary['verdict']}",
                flush=True,
            )
            del frame

    print("=== part 2: monthly cross-check on the two targets ===", flush=True)
    for start, end in MONTHS:
        for table, name, factor_id, role in FACTORS:
            if not role.startswith("target"):
                continue
            frame = _select_range(holder, table, name, start, end)
            summary = _summarize(frame)
            summary["verdict"] = _verdict(summary)
            result["monthly"].setdefault(f"{start}..{end}", {})[factor_id] = summary
            print(
                f"  {start[:7]} {factor_id:<11} dates={summary['trade_dates']:>3} "
                f"codes={summary['distinct_codes']:>5} star={summary['star_codes']:>4} "
                f"(with value {summary['star_codes_with_value']:>4}) "
                f"nonstar_with_value={summary['non_star_codes_with_value']:>5} "
                f"-> {summary['verdict']}",
                flush=True,
            )
            del frame

    result["generated_at"] = date.today().isoformat()
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
