"""Summarize a real-data candidate-model walk-forward report (M1/M2/M3).

Reads the JSON report produced by ``run_real_candidate_training.py`` (full
mode) and renders a deterministic verdict document: G1-G6 gate table,
per-fold constrained-search summary (best config, ensemble validation
RankIC, worst seed), yearly OOS RankIC, and the overall
``WALK_FORWARD_PASSED`` / ``WALK_FORWARD_FAILED`` verdict. With an optional
``--baseline-report`` the renderer adds the validation-stage comparison
against B0-B3 and the data-version snapshot from the baseline run.

Usage:
  summarize_real_candidate_walk_forward.py --report /data/.../m1_report.json \
      --baseline-report /data/.../baseline_oos_report.json \
      --out docs/reports/2026-09-05-m1-real-candidate-walk-forward.md [--print]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

GATE_ORDER: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5", "G6")
REQUIRED_FOLDS: tuple[str, ...] = ("F1", "F2", "F3", "F4")
REQUIRED_YEARS: tuple[str, ...] = ("2022", "2023", "2024", "2025")
BASELINE_IDS: tuple[str, ...] = ("B0", "B1", "B2", "B3")

GATE_DESCRIPTIONS: dict[str, str] = {
    "G1": "OOS mean RankIC >= 0.030",
    "G2": "HAC t-statistic >= 2.000",
    "G3": "positive-year share >= 0.70 (2022-2025)",
    "G4": "delta RankIC vs best_simple baseline >= 0.005",
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
class FoldSummary:
    fold_id: str
    model_id: str
    dataset_id: str
    best_config: dict[str, Any]
    ensemble_validation_rank_ic: float
    worst_seed: int
    searched_config_count: int
    model_selection_id: str
    baseline_selection_id: str


@dataclass(frozen=True)
class CandidateVerdict:
    all_gates_passed: bool
    folds_complete: bool
    years_complete: bool
    mode_full: bool

    @property
    def satisfied(self) -> bool:
        return (
            self.all_gates_passed
            and self.folds_complete
            and self.years_complete
            and self.mode_full
        )

    @property
    def label(self) -> str:
        return "WALK_FORWARD_PASSED" if self.satisfied else "WALK_FORWARD_FAILED"


def load_report(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _oos(report: dict[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], report["oos"])


def gate_verdicts(report: dict[str, Any]) -> tuple[GateVerdict, ...]:
    gates = cast(dict[str, Any], _oos(report).get("gates", {}))
    verdicts: list[GateVerdict] = []
    for gate in GATE_ORDER:
        entry = gates.get(gate)
        if not isinstance(entry, dict):
            # Fail closed on any missing or malformed gate entry.
            verdicts.append(GateVerdict(gate=gate, status="FAIL", observed_value=None, threshold=None))
            continue
        verdicts.append(
            GateVerdict(
                gate=gate,
                status=str(entry.get("status", "FAIL")),
                observed_value=entry.get("observed_value"),
                threshold=entry.get("threshold"),
            )
        )
    return tuple(verdicts)


def fold_summaries(report: dict[str, Any]) -> dict[str, FoldSummary]:
    """Per fold: best config, ensemble validation RankIC, worst seed, evidence ids."""
    summary: dict[str, FoldSummary] = {}
    for record_raw in cast(list[Any], report.get("folds", [])):
        if not isinstance(record_raw, dict):
            continue
        record = cast(dict[str, Any], record_raw)
        fold_id = str(record.get("fold_id"))
        search_results = cast(list[Any], record.get("search_results", []))
        best = max(
            (cast(dict[str, Any], item) for item in search_results),
            key=lambda item: float(item.get("ensemble_validation_rank_ic", float("nan"))),
            default=None,
        )
        ensemble_rank_ic = (
            float(best["ensemble_validation_rank_ic"]) if best is not None else float("nan")
        )
        worst_seed_raw = best.get("worst_seed") if best is not None else None
        summary[fold_id] = FoldSummary(
            fold_id=fold_id,
            model_id=str(record.get("model_id", "-")),
            dataset_id=str(record.get("dataset_id", "-")),
            best_config=dict(cast(dict[str, Any], record.get("best_config", {}))),
            ensemble_validation_rank_ic=ensemble_rank_ic,
            worst_seed=int(worst_seed_raw) if worst_seed_raw is not None else -1,
            searched_config_count=len(search_results),
            model_selection_id=str(record.get("model_selection_id", "-")),
            baseline_selection_id=str(record.get("baseline_selection_id", "-")),
        )
    return summary


def yearly_mean_rank_ic(report: dict[str, Any]) -> dict[str, float | None]:
    yearly = cast(dict[str, Any], _oos(report).get("yearly_mean_rank_ic", {}))
    return {year: yearly.get(year) for year in REQUIRED_YEARS}


def candidate_verdict(report: dict[str, Any]) -> CandidateVerdict:
    oos = _oos(report)
    folds = fold_summaries(report)
    yearly = yearly_mean_rank_ic(report)
    return CandidateVerdict(
        all_gates_passed=bool(oos.get("all_passed", False))
        and all(item.satisfied for item in gate_verdicts(report)),
        folds_complete=sorted(folds) == list(REQUIRED_FOLDS),
        years_complete=all(yearly[year] is not None for year in REQUIRED_YEARS),
        mode_full=str(report.get("mode")) == "full",
    )


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _config_text(config: dict[str, Any]) -> str:
    if not config:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in sorted(config.items()))


def _baseline_validation_scores(
    baseline_report: dict[str, Any],
) -> dict[str, dict[str, float | None]]:
    scores: dict[str, dict[str, float | None]] = {}
    for record_raw in cast(list[Any], baseline_report.get("selections", [])):
        record = cast(dict[str, Any], record_raw)
        fold_id = str(record.get("fold_id"))
        per_baseline: dict[str, float | None] = {baseline: None for baseline in BASELINE_IDS}
        for score_raw in cast(list[Any], record.get("validation_scores", [])):
            score = cast(dict[str, Any], score_raw)
            baseline_id = str(score.get("baseline_id"))
            if baseline_id in BASELINE_IDS:
                per_baseline[baseline_id] = score.get("mean_rank_ic")
        scores[fold_id] = per_baseline
    return scores


def _data_version_summary(baseline_report: dict[str, Any]) -> dict[str, str]:
    raw = baseline_report.get("data_versions")
    if not isinstance(raw, dict) or not raw:
        return {}
    summary: dict[str, str] = {}
    for table in sorted(raw):
        partitions = cast(dict[str, Any], raw[table])
        if not partitions:
            summary[str(table)] = "0 partitions"
            continue
        keys = sorted(partitions)
        summary[str(table)] = f"{len(keys)} partitions ({keys[0]}..{keys[-1]})"
    return summary


def render_markdown(
    report: dict[str, Any], baseline_report: dict[str, Any] | None = None
) -> str:
    oos = _oos(report)
    family = str(report.get("family", "?"))
    verdict = candidate_verdict(report)
    verdicts = gate_verdicts(report)
    folds = fold_summaries(report)
    yearly = yearly_mean_rank_ic(report)
    lines: list[str] = []
    lines.append(f"# Real-Data {family} Candidate Walk-Forward Report")
    lines.append("")
    lines.append(
        "Constrained-search candidate evaluation over F1-F4 (test years 2022-2025) "
        "against the frozen best_simple baseline evidence chain."
    )
    lines.append("")
    lines.append("## Overall OOS statistics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Valid day count | {oos.get('valid_day_count', '-')} |")
    lines.append(f"| Mean RankIC | {_fmt(oos.get('mean_rank_ic'))} |")
    lines.append(f"| ICIR | {_fmt(oos.get('icir'))} |")
    lines.append(f"| HAC t-statistic | {_fmt(oos.get('hac_t_statistic'), 3)} |")
    lines.append(f"| Baseline mean RankIC | {_fmt(oos.get('baseline_mean_rank_ic'))} |")
    lines.append(f"| Delta RankIC | {_fmt(oos.get('delta_rank_ic'))} |")
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
    lines.append("## Per-fold constrained search")
    lines.append("")
    lines.append(
        "| Fold | Model | Best config | Searched | Ens. val RankIC | Worst seed |"
    )
    lines.append("|---|---|---|---|---|---|")
    for fold_id in REQUIRED_FOLDS:
        entry = folds.get(fold_id)
        if entry is None:
            lines.append(f"| {fold_id} | MISSING | - | - | - | - |")
            continue
        lines.append(
            f"| {fold_id} | {entry.model_id} | {_config_text(entry.best_config)} | "
            f"{entry.searched_config_count} | {_fmt(entry.ensemble_validation_rank_ic)} | "
            f"{entry.worst_seed} |"
        )
    lines.append("")
    lines.append("## Yearly OOS mean RankIC")
    lines.append("")
    lines.append("| Year | Mean RankIC |")
    lines.append("|---|---|")
    for year in REQUIRED_YEARS:
        lines.append(f"| {year} | {_fmt(yearly[year])} |")
    lines.append("")
    if baseline_report is not None:
        baseline_scores = _baseline_validation_scores(baseline_report)
        lines.append("## Validation-stage comparison against B0-B3")
        lines.append("")
        lines.append(
            "Candidate ensemble validation RankIC versus each baseline family's "
            "validation mean RankIC (TEST-stage deltas are only defined against "
            "the selected best_simple baseline)."
        )
        lines.append("")
        header = "| Fold | Candidate | " + " | ".join(BASELINE_IDS) + " | Selected |"
        separator = "|---" * (len(BASELINE_IDS) + 3) + "|"
        lines.append(header)
        lines.append(separator)
        selections = {
            str(record.get("fold_id")): str(record.get("selected_baseline_id"))
            for record in cast(list[Any], baseline_report.get("selections", []))
        }
        for fold_id in REQUIRED_FOLDS:
            entry = folds.get(fold_id)
            per_baseline = baseline_scores.get(fold_id, {})
            candidate_value = (
                _fmt(entry.ensemble_validation_rank_ic) if entry is not None else "-"
            )
            cells = [fold_id, candidate_value]
            for baseline in BASELINE_IDS:
                cells.append(_fmt(per_baseline.get(baseline)))
            cells.append(selections.get(fold_id, "-"))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        versions = _data_version_summary(baseline_report)
        if versions:
            lines.append("## Data versions (baseline run snapshot)")
            lines.append("")
            lines.append("| Table | Partitions |")
            lines.append("|---|---|")
            for table, description in versions.items():
                lines.append(f"| {table} | {description} |")
            lines.append("")
    lines.append("## Candidate verdict")
    lines.append("")
    lines.append(f"- Full-mode run: {'yes' if verdict.mode_full else 'no'}")
    lines.append(f"- All G1-G6 gates satisfied: {'yes' if verdict.all_gates_passed else 'no'}")
    lines.append(f"- F1-F4 all present: {'yes' if verdict.folds_complete else 'no'}")
    lines.append(
        f"- Yearly RankIC complete for 2022-2025: {'yes' if verdict.years_complete else 'no'}"
    )
    lines.append("")
    lines.append(f"**{family} candidate: {verdict.label}**")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--print", action="store_true", dest="echo")
    args = parser.parse_args()
    report = load_report(args.report)
    baseline_report = (
        load_report(args.baseline_report) if args.baseline_report is not None else None
    )
    markdown = render_markdown(report, baseline_report=baseline_report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")
    if args.echo:
        print(markdown)
    verdict = candidate_verdict(report)
    family = str(report.get("family", "?"))
    print(f"{family} candidate: {verdict.label}")


if __name__ == "__main__":
    main()
