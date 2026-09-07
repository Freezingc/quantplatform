"""Summarize the M1-M3 real-data candidate comparison (master plan Task 5.1).

Reads whichever of the three family reports exist (``m1_report.json``,
``m2_report.json``, ``m3_report.json``) plus the baseline OOS report, and
renders a deterministic comparison document: per-family gate results, deltas
against the best_simple baseline and against each other, yearly RankIC
distribution, and per-fold worst seeds. The verdict section records either
the shadow pre-approval checklist for passed candidates (Task 5.3) or the
fail-closed statement that no complex candidate passed and the simple
baseline is retained.

Usage:
  summarize_real_candidate_comparison.py \
      --m1-report /data/.../m1_report.json \
      --baseline-report /data/.../baseline_oos_report.json \
      --out docs/reports/2026-09-06-real-candidate-models-comparison.md [--print]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

FAMILY_ORDER: tuple[str, ...] = ("M1", "M2", "M3")
REQUIRED_YEARS: tuple[str, ...] = ("2020", "2021", "2022", "2023", "2024", "2025")
REQUIRED_FOLDS: tuple[str, ...] = ("F1", "F2", "F3", "F4")
GATE_ORDER: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5", "G6")


@dataclass(frozen=True)
class FamilyRow:
    family: str
    status: str  # PASSED / FAILED / NOT_RUN
    mode: str
    mean_rank_ic: float | None
    delta_rank_ic: float | None
    hac_t_statistic: float | None
    valid_day_count: int | None
    all_passed: bool


def load_optional_report(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def family_rows(reports: Mapping[str, dict[str, Any] | None]) -> tuple[FamilyRow, ...]:
    rows: list[FamilyRow] = []
    for family in FAMILY_ORDER:
        report = reports.get(family)
        if not isinstance(report, dict):
            rows.append(
                FamilyRow(family, "NOT_RUN", "-", None, None, None, None, False)
            )
            continue
        mode = str(report.get("mode", "-"))
        oos = cast(dict[str, Any], report.get("oos", {}))
        gates = cast(dict[str, Any], oos.get("gates", {}))
        gates_ok = bool(gates) and all(
            str(gates.get(gate, {}).get("status")) in {"PASS", "NOT_APPLICABLE"}
            for gate in GATE_ORDER
        )
        all_passed = oos.get("all_passed") is True
        if mode != "full":
            status = "NOT_RUN"
        elif all_passed and gates_ok:
            status = "PASSED"
        else:
            status = "FAILED"
        rows.append(
            FamilyRow(
                family=family,
                status=status,
                mode=mode,
                mean_rank_ic=oos.get("mean_rank_ic"),
                delta_rank_ic=oos.get("delta_rank_ic"),
                hac_t_statistic=oos.get("hac_t_statistic"),
                valid_day_count=oos.get("valid_day_count"),
                all_passed=all_passed,
            )
        )
    return tuple(rows)


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _yearly(report: dict[str, Any] | None) -> dict[str, float | None]:
    if not isinstance(report, dict):
        return {year: None for year in REQUIRED_YEARS}
    yearly = cast(dict[str, Any], report.get("oos", {}).get("yearly_mean_rank_ic", {}))
    return {year: yearly.get(year) for year in REQUIRED_YEARS}


def _worst_seeds(report: dict[str, Any] | None) -> dict[str, str]:
    """Per fold: worst seed of the best-ensemble config (selected entry)."""
    seeds: dict[str, str] = {fold: "-" for fold in REQUIRED_FOLDS}
    if not isinstance(report, dict):
        return seeds
    for fold_raw in cast(list[Any], report.get("folds", [])):
        fold = cast(dict[str, Any], fold_raw)
        fold_id = str(fold.get("fold_id"))
        if fold_id not in REQUIRED_FOLDS:
            continue
        results = cast(list[Any], fold.get("search_results", []))
        best = max(
            (cast(dict[str, Any], item) for item in results),
            key=lambda item: float(item.get("ensemble_validation_rank_ic", float("nan"))),
            default=None,
        )
        if best is not None and best.get("worst_seed") is not None:
            seeds[fold_id] = str(best["worst_seed"])
    return seeds


def render_markdown(
    reports: Mapping[str, dict[str, Any] | None],
    baseline_report: dict[str, Any] | None = None,
) -> str:
    rows = family_rows(reports)
    passed = [row for row in rows if row.status == "PASSED"]
    ran = [row for row in rows if row.status != "NOT_RUN"]
    lines: list[str] = []
    lines.append("# Real-Data Candidate Model Comparison (M1-M3)")
    lines.append("")
    lines.append(
        "Master plan Task 5.1 summary of the E-plan candidate sequence "
        "(M1 ResidualMLP -> M2 temporal GRU/TCN -> M3 SetAttention) against the "
        "frozen best_simple baseline evidence chain."
    )
    lines.append("")
    lines.append("## Family summary")
    lines.append("")
    lines.append(
        "| Family | Status | Mean RankIC | Delta vs baseline | HAC t | Valid days |"
    )
    lines.append("|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row.family} | {row.status} | {_fmt(row.mean_rank_ic)} | "
            f"{_fmt(row.delta_rank_ic)} | {_fmt(row.hac_t_statistic, 3)} | "
            f"{row.valid_day_count if row.valid_day_count is not None else '-'} |"
        )
    lines.append("")
    if baseline_report is not None:
        baseline_oos = cast(dict[str, Any], baseline_report.get("oos", {}))
        lines.append(
            f"Baseline (best_simple) OOS mean RankIC: "
            f"{_fmt(baseline_oos.get('mean_rank_ic'))}."
        )
        lines.append("")
    lines.append("## Gate results per family")
    lines.append("")
    for row in ran:
        report = reports.get(row.family)
        assert isinstance(report, dict)
        gates = cast(dict[str, Any], report.get("oos", {}).get("gates", {}))
        lines.append(f"### {row.family}")
        lines.append("")
        lines.append("| Gate | Status | Observed | Threshold |")
        lines.append("|---|---|---|---|")
        for gate in GATE_ORDER:
            entry = cast(dict[str, Any], gates.get(gate, {}))
            observed = entry.get("observed_value")
            threshold = entry.get("threshold")
            lines.append(
                f"| {gate} | {entry.get('status', 'MISSING')} | {_fmt(observed)} | "
                f"{_fmt(threshold, 3) if threshold is not None else '-'} |"
            )
        lines.append("")
    lines.append("## 相对彼此增量 (Deltas against each other, OOS mean RankIC)")
    lines.append("")
    lines.append(
        "Pairwise differences between candidate families that produced full runs "
        "(positive = first family stronger)."
    )
    lines.append("")
    lines.append("| A | B | A - B |")
    lines.append("|---|---|---|")
    for index, left in enumerate(ran):
        for right in ran[index + 1 :]:
            if left.mean_rank_ic is None or right.mean_rank_ic is None:
                continue
            lines.append(
                f"| {left.family} | {right.family} | "
                f"{_fmt(left.mean_rank_ic - right.mean_rank_ic)} |"
            )
    lines.append("")
    lines.append("## Yearly OOS mean RankIC distribution")
    lines.append("")
    header = "| Year | Baseline | " + " | ".join(row.family for row in ran) + " |"
    lines.append(header)
    lines.append("|---" * (len(ran) + 2) + "|")
    baseline_yearly = _yearly(baseline_report)
    for year in REQUIRED_YEARS:
        cells = [year, _fmt(baseline_yearly[year])]
        for row in ran:
            cells.append(_fmt(_yearly(reports.get(row.family))[year]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Worst seeds (per-fold selected config)")
    lines.append("")
    worst_header = "| Fold | " + " | ".join(row.family for row in ran) + " |"
    lines.append(worst_header)
    lines.append("|---" * (len(ran) + 1) + "|")
    worst_by_family = {row.family: _worst_seeds(reports.get(row.family)) for row in ran}
    for fold in REQUIRED_FOLDS:
        cells = [fold]
        for row in ran:
            cells.append(worst_by_family[row.family][fold])
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    if not passed:
        lines.append(
            "- **无复杂候选通过，简单基线保留**：没有任何复杂候选满足全部 G1-G6 "
            "门控（含 G4 增量 >= 0.005）。按主计划风险登记「基线过强」条款记录本结果。"
        )
        for row in rows:
            if row.status == "FAILED":
                lines.append(f"- {row.family} 已运行且判定 WALK_FORWARD_FAILED，失败报告已归档。")
            elif row.status == "NOT_RUN":
                lines.append(f"- {row.family} 未运行（顺序守卫或链路停止）。")
        lines.append("- 不得声称任何复杂候选有效，不得启动 SHADOW。")
    else:
        names = ", ".join(row.family for row in passed)
        lines.append(f"- 通过全部 G1-G6 门控的候选：**{names}**（WALK_FORWARD_PASSED）。")
        lines.append("")
        lines.append("### 影子审批前置清单（移交人工审批）")
        lines.append("")
        lines.append("1. F6.1 部署审批：在首个正式影子日前完成部署审批记录；")
        lines.append("2. 运行至少 60 个有效交易日的 SHADOW，持续记录覆盖、漂移、分数分布与延迟；")
        lines.append("3. 数据/分数监控与回滚演练通过（active pointer 切换与一键回滚）；")
        lines.append("4. 移交人工审批；不擅自启动 SHADOW。")
    lines.append("")
    if baseline_report is not None:
        raw = baseline_report.get("data_versions")
        if isinstance(raw, dict) and raw:
            lines.append("## Data versions (baseline run snapshot)")
            lines.append("")
            lines.append("| Table | Partitions |")
            lines.append("|---|---|")
            for table in sorted(raw):
                partitions = cast(dict[str, Any], raw[table])
                keys = sorted(partitions)
                description = (
                    f"{len(keys)} partitions ({keys[0]}..{keys[-1]})" if keys else "0 partitions"
                )
                lines.append(f"| {table} | {description} |")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-report", type=Path, default=None)
    parser.add_argument("--m2-report", type=Path, default=None)
    parser.add_argument("--m3-report", type=Path, default=None)
    parser.add_argument("--baseline-report", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--print", action="store_true", dest="echo")
    args = parser.parse_args()
    reports = {
        "M1": load_optional_report(args.m1_report),
        "M2": load_optional_report(args.m2_report),
        "M3": load_optional_report(args.m3_report),
    }
    baseline_report = load_optional_report(args.baseline_report)
    markdown = render_markdown(reports, baseline_report=baseline_report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")
    if args.echo:
        print(markdown)
    rows = family_rows(reports)
    passed = [row.family for row in rows if row.status == "PASSED"]
    if passed:
        print(f"passed candidates: {', '.join(passed)}")
    else:
        print("无复杂候选通过，简单基线保留")


if __name__ == "__main__":
    main()
