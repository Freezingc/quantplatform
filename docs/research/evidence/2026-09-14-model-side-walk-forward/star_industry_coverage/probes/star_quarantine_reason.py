"""The platform already recorded WHY those rows were dropped: read factor_quarantine.

probe_star_missing_chain.py localized the loss to our materialization: upstream YesLab carries the
688/689 rows with values, `factor_daily` has zero STAR rows for yeslab_357 / yeslab_471 on the same
day, and a control factor on the same table (yeslab_381, GP_012) got all of them through. That
leaves the dedup/clean stage as the only place left, and that stage does not drop silently --
`deduplicate_factor_observations` writes every dropped key to `factor_quarantine` with a reason:

  NO_FINITE_VALUE     the value is NaN or +/-inf, so there is nothing finite to publish
  CONFLICT_QUARANTINE the YesLab duplicate rows for one key disagree, and the policy refuses to pick

So this reads the reason rather than inferring it, for the two dropped days and across the October
2023 breakpoint, and keeps the raw `distinct_values_json` so the number behind the reason is
visible. Which of the two it is decides who owns the fix: NO_FINITE_VALUE points upstream at the
factor definition (an industry-relative standardization has no defined value for a security that is
alone in its industry), CONFLICT_QUARANTINE points at the producer's duplicate rows.

SAFETY: SELECT-only against dfs://QuantResearchV2Test. No writes, no weights, no artifact mutation.
Run from research_v2/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

import pandas as pd  # noqa: E402
from walk_forward_common import TARGET_DATABASE, new_holder, query  # noqa: E402

OUT = Path("/data/zhangyudong/qrv2_research_artifacts/star_quarantine_reason_probe.json")
FACTORS = ("yeslab_357", "yeslab_471", "yeslab_381")
DAYS = ("2022.07.29", "2023.07.05")
WINDOW_START, WINDOW_END = "2023.06.01", "2023.11.30"
STAR_PREFIXES = ("688", "689")


def _q(holder: dict, where: str) -> pd.DataFrame:
    sql = (
        "select observation_date, ts_code, factor_id, reason, distinct_values_json, data_version "
        f"from loadTable('{TARGET_DATABASE}','factor_quarantine') where {where}"
    )
    ok, frame = query(holder, sql)
    if not ok or not isinstance(frame, pd.DataFrame):
        raise RuntimeError(f"quarantine query failed: {where}")
    return frame


def _split(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"rows": 0}
    code = frame["ts_code"].astype(str)
    is_star = code.str.startswith(STAR_PREFIXES)
    out: dict = {"rows": int(len(frame)), "by_reason": {}}
    for reason, group in frame.assign(_star=is_star).groupby(frame["reason"].astype(str)):
        out["by_reason"][reason] = {
            "rows": int(len(group)),
            "star_rows": int(group["_star"].sum()),
            "non_star_rows": int((~group["_star"]).sum()),
            "distinct_codes": int(group["ts_code"].nunique()),
        }
    return out


def main() -> None:
    holder = new_holder()
    result: dict = {"days": {}, "window": {}, "samples": {}}

    print("=== part 1: the two dropped days ===", flush=True)
    for day in DAYS:
        for factor_id in FACTORS:
            frame = _q(
                holder,
                f"observation_date between date('{day}') and date('{day}') "
                f"and factor_id='{factor_id}'",
            )
            summary = _split(frame)
            result["days"].setdefault(day, {})[factor_id] = summary
            reasons = summary.get("by_reason", {})
            text = ", ".join(
                f"{reason}: {stats['rows']} rows ({stats['star_rows']} STAR)"
                for reason, stats in sorted(reasons.items())
            ) or "no quarantine rows"
            print(f"  {day} {factor_id:<11} {text}", flush=True)
            if not frame.empty:
                star_rows = frame.loc[frame["ts_code"].astype(str).str.startswith(STAR_PREFIXES)]
                sample_frame = star_rows if not star_rows.empty else frame
                result["samples"].setdefault(day, {})[factor_id] = [
                    {
                        "ts_code": str(row.ts_code),
                        "reason": str(row.reason),
                        "distinct_values": str(row.distinct_values_json)[:120],
                    }
                    for row in sample_frame.head(5).itertuples(index=False)
                ]
                for entry in result["samples"][day][factor_id][:3]:
                    print(
                        f"          e.g. {entry['ts_code']} {entry['reason']} "
                        f"values={entry['distinct_values']}",
                        flush=True,
                    )
            del frame

    print("=== part 2: across the October 2023 breakpoint ===", flush=True)
    for factor_id in ("yeslab_357", "yeslab_471", "yeslab_381"):
        frame = _q(
            holder,
            f"observation_date between date('{WINDOW_START}') and date('{WINDOW_END}') "
            f"and factor_id='{factor_id}'",
        )
        if frame.empty:
            result["window"][factor_id] = {"rows": 0}
            print(f"  {factor_id}: no quarantine rows in {WINDOW_START}..{WINDOW_END}", flush=True)
            continue
        frame["month"] = pd.to_datetime(frame["observation_date"]).dt.strftime("%Y-%m")
        code = frame["ts_code"].astype(str)
        frame["_star"] = code.str.startswith(STAR_PREFIXES)
        monthly = {
            str(month): {
                "rows": int(len(group)),
                "star_rows": int(group["_star"].sum()),
                "reasons": {
                    str(reason): int(count)
                    for reason, count in group["reason"].astype(str).value_counts().items()
                },
            }
            for month, group in frame.groupby("month")
        }
        result["window"][factor_id] = {"rows": int(len(frame)), "monthly": monthly}
        for month, stats in sorted(monthly.items()):
            print(
                f"  {factor_id} {month}: {stats['rows']:>5} quarantined "
                f"({stats['star_rows']:>4} STAR) reasons={stats['reasons']}",
                flush=True,
            )
        del frame

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
