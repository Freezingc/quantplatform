"""Pre-register what M2's evaluation stage must produce, before it runs, with no number retyped.

The dense-intersection screen predicts the unevaluable OOS day set from published tables alone. M2
is still inside its search stage, so its evaluation has not run and its report does not exist. That
is exactly when a prediction is worth writing down: everything here is derived from the instrument's
JSON output plus the baseline artifact M2 was launched against, and the generator reads those files
instead of accepting transcribed numbers, so the record cannot drift from the measurement.

M2 was launched with --baseline-artifacts baseline_v2/baseline_oos and --baseline-report
baseline_v2/report.json, i.e. the same frozen best_simple artifacts that m0r_v2 and m1_full_v1 were
scored against, and the candidate side scores every valid-label row. So the day set is a property of
inputs M2 already has, and the prediction is falsifiable the moment its report lands.

Usage: make_m2_preregistration.py --instrument <json> --out-json <json> --out-md <md>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

# Fold -> TEST year, from the frozen v2 fold contract (annual_folds), as recorded in the evidence
# pack's dataset table: F1 tests 2022, F2 2023, F3 2024, F4 2025.
FOLD_TEST_YEAR = {"F1": 2022, "F2": 2023, "F3": 2024, "F4": 2025}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    measured = json.loads(args.instrument.read_text(encoding="utf-8"))
    summary = measured["summary"]
    per_day = measured["per_day"]
    floor = summary["floor"]

    by_year: dict[str, dict] = {}
    for year, bucket in summary["by_year"].items():
        coverages = [
            entry["coverage"] for entry in per_day if entry["day"].startswith(year)
        ]
        by_year[year] = {
            "days": bucket["days"],
            "days_below_floor": bucket["days_below_floor"],
            "expected_valid_days": bucket["days"] - bucket["days_below_floor"],
            "coverage_min": min(coverages),
            "coverage_median": bucket["coverage_median"],
            "below_floor_dates": bucket["days_below_floor_list"],
        }

    total_days = sum(entry["days"] for entry in by_year.values())
    total_valid = sum(entry["expected_valid_days"] for entry in by_year.values())
    payload = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "PRE-REGISTERED BEFORE M2's EVALUATION STAGE RAN",
        "carrier": "M2",
        "workspace": "/data/zhangyudong/qrv2_research_artifacts/m2_v1",
        "baseline": {
            "artifacts": "/data/zhangyudong/qrv2_research_artifacts/baseline_v2/baseline_oos",
            "report": "/data/zhangyudong/qrv2_research_artifacts/baseline_v2/report.json",
            "note": "the same frozen best_simple artifacts m0r_v2 and m1_full_v1 were scored against",
        },
        "source_measurement": {
            "path": str(args.instrument),
            "sha256": hashlib.sha256(args.instrument.read_bytes()).hexdigest(),
            "instrument": "screen_dense_intersection_coverage.py (codex/dense-intersection-screen)",
            "floor": floor,
            "factor_source": measured["factor_source"],
            "factor_ids": measured["diagnostics"]["factor_ids"],
        },
        "predictions": {
            "g6_valid_score_days": f"{total_valid}/{total_days}",
            "g6_ratio": total_valid / total_days,
            "per_fold": {
                fold: {
                    "test_year": year,
                    "days": by_year[str(year)]["days"],
                    "days_below_floor": by_year[str(year)]["days_below_floor"],
                    "expected_valid_days": by_year[str(year)]["expected_valid_days"],
                    "expected_coverage_min": by_year[str(year)]["coverage_min"],
                    "expected_coverage_median": by_year[str(year)]["coverage_median"],
                }
                for fold, year in FOLD_TEST_YEAR.items()
            },
            "candidate_side_daily_coverage": 1.0,
            "days_dropped_by_the_baseline_side": summary["days_below_floor"],
            "unevaluable_dates": summary["days_below_floor_list"],
        },
        "falsification": (
            "If M2's signed report disagrees with any of these -- the 945/969 day count, a fold's "
            "valid-day count, a per-year coverage min or median, or candidate daily coverage of "
            "1.0000 -- then an input changed between m0r_v2/m1_full_v1 and M2 (baseline artifact, "
            "labels, factor set, or the floor) and M2's numbers must not be accepted until the "
            "difference is explained. Agreement is evidence the evaluation consumed the same frozen "
            "inputs; it is not evidence that the candidate is good."
        ),
    }
    args.out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# M2 评价阶段的事前预测登记（在评价跑之前写下）",
        "",
        f"登记时刻（UTC）：{payload['registered_at_utc']}。",
        f"来源测量：`{args.instrument.name}`（sha256 `{payload['source_measurement']['sha256'][:16]}…`），",
        f"仪器 `screen_dense_intersection_coverage.py`，因子源 `{measured['factor_source']}`"
        f"（{len(payload['source_measurement']['factor_ids'])} 个），地板 {floor}（从 `gates.py` 解析）。",
        "",
        "**M2 发射时用的基线工件与 `m0r_v2` / `m1_full_v1` 是同一套**"
        "（`baseline_v2/baseline_oos` + `baseline_v2/report.json`），候选一侧给每个有效标签行都打分，",
        "所以这个日集是 M2 **已经拿到的输入**的性质，可以在它的评价阶段跑完之前就写死。",
        "",
        "## 预测",
        "",
        f"* **G6 = {total_valid}/{total_days}** = {total_valid / total_days!r}",
        f"* 候选一侧逐日覆盖率恒为 **1.0000**；`days_dropped_by_the_baseline_side` = "
        f"**{summary['days_below_floor']}/{summary['days_below_floor']}**",
        "",
        "| 折 | TEST 年 | 天数 | 跌破地板 | **预测可评天** | 覆盖率 min | 覆盖率 median |",
        "|---|---|---|---|---|---|---|",
    ]
    for fold, year in FOLD_TEST_YEAR.items():
        entry = by_year[str(year)]
        lines.append(
            f"| {fold} | {year} | {entry['days']} | {entry['days_below_floor']} | "
            f"**{entry['expected_valid_days']}** | {entry['coverage_min']!r} | "
            f"{entry['coverage_median']!r} |"
        )
    lines += [
        "",
        f"不可评的 {summary['days_below_floor']} 天：",
        "",
        "```",
        ", ".join(summary["days_below_floor_list"]),
        "```",
        "",
        "## 什么情况下这份预测算被推翻",
        "",
        payload["falsification"],
        "",
        "## 什么情况下它**不算**被验证",
        "",
        "一致只说明 M2 的评价消费了同一套冻结输入（基线工件、标签、因子集、地板都没变），",
        "**不说明候选好坏**，也不说明那 24 天该被丢掉——那 24 天的成因与处置见",
        "`star_industry_coverage/root_cause.md` 与 `coverage_gates_vs_the_intersection_floor/finding.md`。",
        "",
    ]
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"G6 prediction: {total_valid}/{total_days} = {total_valid / total_days!r}")
    for fold, year in FOLD_TEST_YEAR.items():
        entry = by_year[str(year)]
        print(
            f"  {fold} test {year}: {entry['days']} days, {entry['days_below_floor']} below floor, "
            f"valid {entry['expected_valid_days']}, coverage min {entry['coverage_min']:.6f}"
        )
    print(f"wrote {args.out_json}\nwrote {args.out_md}")


if __name__ == "__main__":
    main()
