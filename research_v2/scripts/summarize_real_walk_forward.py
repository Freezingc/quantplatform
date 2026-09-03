"""Summarize the real-data B0-B3 walk-forward report for completion criterion 303.

Reads the JSON report produced by ``run_real_walk_forward.py`` and renders a
gate-by-gate verdict table, the per-fold B0-B3 selection summary, and the
yearly OOS RankIC table. The output is deterministic so it can be archived as
the criterion-303 evidence artifact.

Usage:
  summarize_real_walk_forward.py --report /tmp/real_walk_forward_report.json \
      --out docs/reports/2026-09-03-real-walk-forward-criterion-303.md [--print]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

GATE_ORDER: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5", "G6")
REQUIRED_FOLDS: tuple[str, ...] = ("F01", "F02", "F03", "F04", "F05", "F06")
REQUIRED_YEARS: tuple[str, ...] = ("2020", "2021", "2022", "2023", "2024", "2025")
BASELINE_IDS: tuple[str, ...] = ("B0", "B1", "B2", "B3")

GATE_DESCRIPTIONS: dict[str, str] = {
    "G1": "OOS mean RankIC >= 0.030",
    "G2": "HAC t-statistic >= 2.000",
    "G3": "positive-year share >= 0.70 (2020-2025)",
    "G4": "delta RankIC vs baseline >= 0.005 (N/A for baseline path)",
    "G5": "max single-year positive contribution share <= 0.50",
    "G6": "valid score-day ratio >= 0.95",
}


@dataclass(frozen=True)
class GateVerdict:
    gate: str
    status: str
    observed_value: float | None
    threshold: float | None

    @property
    def satisfied(self) -> bool:
        return self.status in {"PASS", "NOT_APPLICABLE"}


@dataclass(frozen=True)
class FoldSelection:
    fold_id: str
    winner: str
    validation_rank_ic: dict[str, float | None]


@dataclass(frozen=True)
class Criterion303Verdict:
    all_gates_passed: bool
    folds_complete: bool
    baselines_complete: bool
    years_complete: bool

    @property
    def satisfied(self) -> bool:
        return (
            self.all_gates_passed
            and self.folds_complete
            and self.baselines_complete
            and self.years_complete
        )


def load_report(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _oos(report: dict[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], report["oos"])


def gate_verdicts(report: dict[str, Any]) -> tuple[GateVerdict, ...]:
    gates = cast(dict[str, Any], _oos(report)["gates"])
    verdicts: list[GateVerdict] = []
    for gate in GATE_ORDER:
        entry = cast(dict[str, Any], gates[gate])
        verdicts.append(
            GateVerdict(
                gate=gate,
                status=str(entry["status"]),
                observed_value=entry.get("observed_value"),
                threshold=entry.get("threshold"),
            )
        )
    return tuple(verdicts)


def fold_selections(report: dict[str, Any]) -> dict[str, FoldSelection]:
    """Per fold: validation RankIC of every B0-B3 candidate plus the winner."""
    summary: dict[str, FoldSelection] = {}
    for record_raw in cast(list[Any], report["selections"]):
        record = cast(dict[str, Any], record_raw)
        fold_id = str(record["fold_id"])
        rank_ic: dict[str, float | None] = {baseline: None for baseline in BASELINE_IDS}
        for score_raw in cast(list[Any], record["validation_scores"]):
            score = cast(dict[str, Any], score_raw)
            baseline_id = str(score["baseline_id"])
            if baseline_id in BASELINE_IDS:
                rank_ic[baseline_id] = score["mean_rank_ic"]
        summary[fold_id] = FoldSelection(
            fold_id=fold_id,
            winner=str(record["selected_baseline_id"]),
            validation_rank_ic=rank_ic,
        )
    return summary


def yearly_mean_rank_ic(report: dict[str, Any]) -> dict[str, float | None]:
    yearly = cast(dict[str, Any], _oos(report)["yearly_mean_rank_ic"])
    return {year: yearly.get(year) for year in REQUIRED_YEARS}


def data_version_summary(report: dict[str, Any]) -> dict[str, str] | None:
    """Compact per-table version summary, or None when the report lacks versions."""
    raw = report.get("data_versions")
    if not isinstance(raw, dict) or not raw:
        return None
    summary: dict[str, str] = {}
    for table in sorted(raw):
        partitions = cast(dict[str, Any], raw[table])
        if not partitions:
            summary[str(table)] = "0 partitions"
            continue
        keys = sorted(partitions)
        summary[str(table)] = (
            f"{len(keys)} partitions ({keys[0]}..{keys[-1]})"
        )
    return summary


def criterion_303_verdict(report: dict[str, Any]) -> Criterion303Verdict:
    oos = _oos(report)
    selections = fold_selections(report)
    folds_complete = sorted(selections) == list(REQUIRED_FOLDS)
    baselines_complete = folds_complete and all(
        all(
            selections[fold].validation_rank_ic[baseline] is not None
            for baseline in BASELINE_IDS
        )
        for fold in REQUIRED_FOLDS
    )
    yearly = yearly_mean_rank_ic(report)
    years_complete = all(yearly[year] is not None for year in REQUIRED_YEARS)
    return Criterion303Verdict(
        all_gates_passed=bool(oos["all_passed"]),
        folds_complete=folds_complete,
        baselines_complete=baselines_complete,
        years_complete=years_complete,
    )


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    oos = _oos(report)
    verdicts = gate_verdicts(report)
    selections = fold_selections(report)
    yearly = yearly_mean_rank_ic(report)
    verdict = criterion_303_verdict(report)
    lines: list[str] = []
    lines.append("# Real-Data B0-B3 Walk-Forward Report (Criterion 303)")
    lines.append("")
    lines.append(
        "OOS evaluation over F01-F06 (test years 2020-2025) built from the full-range "
        "PIT tables with the admitted factor set."
    )
    lines.append("")
    lines.append("## Overall OOS statistics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Valid day count | {oos['valid_day_count']} |")
    lines.append(f"| Mean RankIC | {_fmt(oos['mean_rank_ic'])} |")
    lines.append(f"| ICIR | {_fmt(oos['icir'])} |")
    lines.append(f"| HAC t-statistic | {_fmt(oos['hac_t_statistic'], 3)} |")
    lines.append(f"| Baseline mean RankIC | {_fmt(oos['baseline_mean_rank_ic'])} |")
    lines.append(f"| Delta RankIC | {_fmt(oos['delta_rank_ic'])} |")
    lines.append("")
    lines.append("## Gate verdicts (G1-G6)")
    lines.append("")
    lines.append("| Gate | Definition | Observed | Threshold | Status |")
    lines.append("|---|---|---|---|---|")
    for item in verdicts:
        threshold = _fmt(item.threshold, 3) if item.threshold is not None else "-"
        lines.append(
            f"| {item.gate} | {GATE_DESCRIPTIONS[item.gate]} | "
            f"{_fmt(item.observed_value)} | {threshold} | {item.status} |"
        )
    lines.append("")
    lines.append("## Per-fold validation selection (B0-B3)")
    lines.append("")
    lines.append("| Fold | B0 | B1 | B2 | B3 | Winner |")
    lines.append("|---|---|---|---|---|---|")
    for fold in REQUIRED_FOLDS:
        selection = selections.get(fold)
        if selection is None:
            lines.append(f"| {fold} | - | - | - | - | MISSING |")
            continue
        cells = [fold]
        for baseline in BASELINE_IDS:
            value = selection.validation_rank_ic[baseline]
            marker = "**" if selection.winner == baseline else ""
            cells.append(f"{marker}{_fmt(value)}{marker}")
        cells.append(selection.winner)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Yearly OOS mean RankIC")
    lines.append("")
    lines.append("| Year | Mean RankIC |")
    lines.append("|---|---|")
    for year in REQUIRED_YEARS:
        lines.append(f"| {year} | {_fmt(yearly[year])} |")
    lines.append("")
    versions = data_version_summary(report)
    if versions is not None:
        lines.append("## Data versions (published_version snapshot)")
        lines.append("")
        lines.append("| Table | Partitions |")
        lines.append("|---|---|")
        for table, description in versions.items():
            lines.append(f"| {table} | {description} |")
        lines.append("")
        lines.append(
            "Full per-partition version mapping is stored in the report JSON "
            "under `data_versions`."
        )
        lines.append("")
    lines.append("## Criterion 303 verdict")
    lines.append("")
    lines.append(f"- All G1-G6 gates satisfied: {'yes' if verdict.all_gates_passed else 'no'}")
    lines.append(f"- F01-F06 all present: {'yes' if verdict.folds_complete else 'no'}")
    lines.append(
        f"- B0-B3 all fitted and scored per fold: "
        f"{'yes' if verdict.baselines_complete else 'no'}"
    )
    lines.append(
        f"- Yearly RankIC complete for 2020-2025: "
        f"{'yes' if verdict.years_complete else 'no'}"
    )
    lines.append("")
    lines.append(
        f"**Criterion 303: {'SATISFIED' if verdict.satisfied else 'NOT SATISFIED'}**"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--print", action="store_true", dest="echo")
    args = parser.parse_args()
    report = load_report(args.report)
    markdown = render_markdown(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")
    if args.echo:
        print(markdown)
    verdict = criterion_303_verdict(report)
    print(f"criterion-303: {'SATISFIED' if verdict.satisfied else 'NOT SATISFIED'}")


if __name__ == "__main__":
    main()
