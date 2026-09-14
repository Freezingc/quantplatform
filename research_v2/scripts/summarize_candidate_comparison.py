"""Render the cross-candidate comparison that Task 5.1 of the real-candidate plan asks for.

Reads any number of signed full-mode walk-forward reports (``run_real_candidate_training.py``
output) plus, optionally, the frozen baseline report, and renders ONE deterministic markdown
document containing exactly what plan Task 5.1 lists:

* each candidate's G1-G6 gate results;
* each candidate's increment over the baseline (G4) and over every other candidate;
* the leading candidate's yearly distribution and worst seed;
* and, when nothing passed, the explicit "no complex candidate passed, the simple baseline is
  retained" record the master plan's risk register requires.

Two properties are deliberate.

1. **Nothing is recomputed and nothing is estimated.** Every number is copied from a report and
   the source file's sha256 is printed next to it, so the rendered document can be checked
   against the artifacts it claims to come from.
2. **A subtraction is only printed when it is legal.** Candidates are compared only inside a
   comparability group: same ``valid_day_count``, same ``baseline_mean_rank_ic``, same per-fold
   frozen baseline selections and same per-fold dataset ids. That is the condition under which
   two G4 deltas share a baseline term and a day set, so their difference is meaningful.
   Candidates outside the group are listed with the reason and get ``n/a`` cells instead of
   numbers. Within a group the renderer also *verifies* the identity
   ``delta(a) - delta(b) == mean(a) - mean(b)`` (which holds only because the baseline term is
   shared) and prints ``IDENTITY_VIOLATION`` rather than a number if it ever fails.

Rendering is a pure function of the inputs: no timestamp, no hostname, no dict-iteration order.
Rendering the same reports twice yields byte-identical output, which is what lets an archived
comparison be diffed against a re-render.

Usage:
  summarize_candidate_comparison.py \
      --report M0R=/data/.../m0r_v2/report.json \
      --report M1-full=/data/.../m1_full_v1/report.json \
      --baseline-report /data/.../baseline_v2/report.json \
      --out /data/.../comparison.md [--print]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

GATE_ORDER: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5", "G6")
REQUIRED_FOLDS: tuple[str, ...] = ("F1", "F2", "F3", "F4")
REQUIRED_YEARS: tuple[str, ...] = ("2022", "2023", "2024", "2025")
BASELINE_IDS: tuple[str, ...] = ("B0", "B1", "B2", "B3")

# Tolerance for the shared-baseline identity check. The two sides are computed from the same
# floats in a different order, so this is round-off, not disagreement.
IDENTITY_TOLERANCE = 1e-12

NO_CANDIDATE_PASSED_RECORD = (
    "No complex candidate passed the walk-forward gates; the simple baseline is retained. "
    "This is the outcome the master plan's risk register files under 'baseline too strong'."
)

GATE_DESCRIPTIONS: dict[str, str] = {
    "G1": "OOS mean RankIC >= 0.030",
    "G2": "HAC t-statistic >= 2.000",
    "G3": "positive-year share >= 0.70 (2022-2025)",
    "G4": "delta RankIC vs fold-matched best_simple >= 0.005",
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
class FoldRow:
    fold_id: str
    model_id: str
    dataset_id: str
    baseline_selection_id: str
    best_config: dict[str, Any]
    configs_searched: int
    ensemble_validation_rank_ic: float | None
    worst_seed: int | None
    worst_seed_validation_rank_ic: float | None
    seed_validation_rank_ic: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class Candidate:
    label: str
    path: Path
    sha256: str
    family: str
    mode: str
    all_passed: bool
    gates: tuple[GateVerdict, ...]
    valid_day_count: int | None
    mean_rank_ic: float | None
    icir: float | None
    hac_t_statistic: float | None
    baseline_mean_rank_ic: float | None
    delta_rank_ic: float | None
    yearly: tuple[tuple[str, float | None], ...]
    folds: tuple[FoldRow, ...]

    @property
    def gates_passed(self) -> bool:
        """True only if the report says so AND every one of the six gates is present and satisfied.

        ``all()`` over an empty tuple is True, so checking only the verdicts this candidate
        happens to carry would let a report with no gate entries at all render as PASSED. The
        gate list is therefore required to cover GATE_ORDER, not merely to contain no failure.
        """
        if not self.all_passed:
            return False
        by_gate = {item.gate: item for item in self.gates}
        return all(gate in by_gate and by_gate[gate].satisfied for gate in GATE_ORDER)

    @property
    def folds_complete(self) -> bool:
        return tuple(fold.fold_id for fold in self.folds) == REQUIRED_FOLDS

    @property
    def years_complete(self) -> bool:
        return all(value is not None for _, value in self.yearly)

    @property
    def mode_full(self) -> bool:
        return self.mode == "full"

    @property
    def passed(self) -> bool:
        return (
            self.gates_passed and self.folds_complete and self.years_complete and self.mode_full
        )

    @property
    def verdict(self) -> str:
        return "WALK_FORWARD_PASSED" if self.passed else "WALK_FORWARD_FAILED"

    @property
    def configs_searched_max(self) -> int:
        return max((fold.configs_searched for fold in self.folds), default=0)

    @property
    def selection_free(self) -> bool:
        """True when no fold had a choice to make, so validation selection added no optimism."""
        return bool(self.folds) and all(fold.configs_searched <= 1 for fold in self.folds)

    def fold(self, fold_id: str) -> FoldRow | None:
        for row in self.folds:
            if row.fold_id == fold_id:
                return row
        return None


@dataclass(frozen=True)
class ComparabilityKey:
    valid_day_count: int | None
    baseline_mean_rank_ic: float | None
    baseline_selection_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]

    def describe(self) -> str:
        return (
            f"valid_day_count={self.valid_day_count}, "
            f"baseline_mean_rank_ic={self.baseline_mean_rank_ic!r}, "
            f"baseline_selections={','.join(self.baseline_selection_ids) or '-'}, "
            f"datasets={','.join(value[:12] for value in self.dataset_ids) or '-'}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _fold_row(record: dict[str, Any]) -> FoldRow:
    search_results = [
        cast(dict[str, Any], item)
        for item in cast(list[Any], record.get("search_results", []))
        if isinstance(item, dict)
    ]
    best = max(
        search_results,
        key=lambda item: _as_float(item.get("ensemble_validation_rank_ic"))
        if _as_float(item.get("ensemble_validation_rank_ic")) is not None
        else float("-inf"),
        default=None,
    )
    ensemble_ic = _as_float(best.get("ensemble_validation_rank_ic")) if best else None
    seeds = cast(dict[str, Any], (best or {}).get("seed_validation_rank_ic", {}))
    seed_pairs = tuple(
        (int(seed), float(value))
        for seed, value in sorted(seeds.items(), key=lambda item: int(item[0]))
        if _as_float(value) is not None
    )
    worst_seed_raw = (best or {}).get("worst_seed")
    worst_seed = _as_int(worst_seed_raw)
    worst_seed_ic = dict(seed_pairs).get(worst_seed) if worst_seed is not None else None
    return FoldRow(
        fold_id=str(record.get("fold_id", "-")),
        model_id=str(record.get("model_id", "-")),
        dataset_id=str(record.get("dataset_id", "-")),
        baseline_selection_id=str(record.get("baseline_selection_id", "-")),
        best_config=dict(cast(dict[str, Any], record.get("best_config", {}))),
        configs_searched=len(search_results),
        ensemble_validation_rank_ic=ensemble_ic,
        worst_seed=worst_seed,
        worst_seed_validation_rank_ic=worst_seed_ic,
        seed_validation_rank_ic=seed_pairs,
    )


def load_candidate(label: str, path: Path) -> Candidate:
    report = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    oos = cast(dict[str, Any], report.get("oos", {}))
    gates_raw = cast(dict[str, Any], oos.get("gates", {}))
    gates: list[GateVerdict] = []
    for gate in GATE_ORDER:
        entry = gates_raw.get(gate)
        if not isinstance(entry, dict):
            # Fail closed: a missing or malformed gate is a FAIL, never a silent pass.
            gates.append(GateVerdict(gate=gate, status="FAIL", observed_value=None, threshold=None))
            continue
        gates.append(
            GateVerdict(
                gate=gate,
                status=str(entry.get("status", "FAIL")),
                observed_value=_as_float(entry.get("observed_value")),
                threshold=_as_float(entry.get("threshold")),
            )
        )
    folds = tuple(
        _fold_row(cast(dict[str, Any], record))
        for record in cast(list[Any], report.get("folds", []))
        if isinstance(record, dict)
    )
    yearly_raw = cast(dict[str, Any], oos.get("yearly_mean_rank_ic", {}))
    return Candidate(
        label=label,
        path=path,
        sha256=sha256_file(path),
        family=str(report.get("family", "?")),
        mode=str(report.get("mode", "?")),
        all_passed=bool(oos.get("all_passed", False)),
        gates=tuple(gates),
        valid_day_count=_as_int(oos.get("valid_day_count")),
        mean_rank_ic=_as_float(oos.get("mean_rank_ic")),
        icir=_as_float(oos.get("icir")),
        hac_t_statistic=_as_float(oos.get("hac_t_statistic")),
        baseline_mean_rank_ic=_as_float(oos.get("baseline_mean_rank_ic")),
        delta_rank_ic=_as_float(oos.get("delta_rank_ic")),
        yearly=tuple((year, _as_float(yearly_raw.get(year))) for year in REQUIRED_YEARS),
        folds=folds,
    )


def comparability_key(candidate: Candidate) -> ComparabilityKey:
    return ComparabilityKey(
        valid_day_count=candidate.valid_day_count,
        baseline_mean_rank_ic=candidate.baseline_mean_rank_ic,
        baseline_selection_ids=tuple(fold.baseline_selection_id for fold in candidate.folds),
        dataset_ids=tuple(fold.dataset_id for fold in candidate.folds),
    )


def split_comparable(
    candidates: tuple[Candidate, ...],
) -> tuple[ComparabilityKey | None, list[Candidate], list[tuple[Candidate, str]]]:
    """Partition candidates into the reference group and the excluded ones, with reasons.

    The reference group is the largest set of candidates sharing one comparability key; ties are
    broken by CLI order so the result is a pure function of the inputs. A candidate that cannot
    even name four folds is excluded with its own reason rather than being silently grouped.
    """
    excluded: list[tuple[Candidate, str]] = []
    groupable: list[Candidate] = []
    for candidate in candidates:
        if not candidate.folds_complete:
            excluded.append(
                (
                    candidate,
                    "folds incomplete: found "
                    f"{','.join(fold.fold_id for fold in candidate.folds) or 'none'} "
                    f"instead of {','.join(REQUIRED_FOLDS)}",
                )
            )
            continue
        groupable.append(candidate)
    if not groupable:
        return None, [], excluded
    keys: dict[ComparabilityKey, list[Candidate]] = {}
    for candidate in groupable:
        keys.setdefault(comparability_key(candidate), []).append(candidate)
    best_key = max(keys, key=lambda key: (len(keys[key]), -_first_index(groupable, keys[key][0])))
    group = keys[best_key]
    for candidate in groupable:
        # Identity, not equality: two runs of one carrier can be value-identical (M0R and its
        # byte-identical replicate differ only in label and path), and `in` would then test
        # dataclass equality rather than "is this the candidate I put in the group".
        if any(candidate is item for item in group):
            continue
        excluded.append(
            (
                candidate,
                "different comparison basis: "
                f"{comparability_key(candidate).describe()} vs the reference group's "
                f"{best_key.describe()}",
            )
        )
    return best_key, group, excluded


def _first_index(candidates: list[Candidate], target: Candidate) -> int:
    for index, candidate in enumerate(candidates):
        if candidate is target:
            return index
    return len(candidates)


def leading_candidate(group: list[Candidate]) -> Candidate | None:
    """The passed candidate with the largest G4 delta; CLI order breaks exact ties."""
    passed = [
        candidate
        for candidate in group
        if candidate.passed and candidate.delta_rank_ic is not None
    ]
    if not passed:
        return None
    best = max(passed, key=lambda candidate: cast(float, candidate.delta_rank_ic))
    tied = [
        candidate
        for candidate in passed
        if cast(float, candidate.delta_rank_ic) == cast(float, best.delta_rank_ic)
    ]
    return tied[0]


def _fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _fmt_signed(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "-"
    return f"{value:+.{digits}f}"


def _config_text(config: dict[str, Any]) -> str:
    if not config:
        return "-"
    return ", ".join(f"{key}={config[key]}" for key in sorted(config))


def _identity_holds(left: Candidate, right: Candidate) -> bool:
    """delta(a) - delta(b) must equal mean(a) - mean(b) when the baseline term is shared."""
    if None in (
        left.delta_rank_ic,
        right.delta_rank_ic,
        left.mean_rank_ic,
        right.mean_rank_ic,
    ):
        return False
    delta_gap = cast(float, left.delta_rank_ic) - cast(float, right.delta_rank_ic)
    mean_gap = cast(float, left.mean_rank_ic) - cast(float, right.mean_rank_ic)
    return abs(delta_gap - mean_gap) <= IDENTITY_TOLERANCE


def render_markdown(
    candidates: tuple[Candidate, ...],
    baseline_report: dict[str, Any] | None = None,
) -> str:
    lines: list[str] = []
    key, group, excluded = split_comparable(candidates)
    leader = leading_candidate(group)

    lines.append("# Real-Data Candidate Comparison (plan Task 5.1)")
    lines.append("")
    lines.append(
        "Every number below is copied from a signed walk-forward report; nothing is recomputed "
        "and nothing is estimated. Pairwise increments are printed only inside the comparability "
        "group defined in the next section."
    )
    lines.append("")

    lines.append("## Inputs")
    lines.append("")
    lines.append("| Label | Family | Mode | Report sha256 | Path |")
    lines.append("|---|---|---|---|---|")
    for candidate in candidates:
        lines.append(
            f"| {candidate.label} | {candidate.family} | {candidate.mode} | "
            f"`{candidate.sha256[:16]}` | `{candidate.path}` |"
        )
    lines.append("")

    lines.append("## Comparability")
    lines.append("")
    if key is None:
        lines.append(
            "**No comparability group could be formed**: no candidate reports all four folds."
        )
    else:
        lines.append(
            f"Reference group ({len(group)} of {len(candidates)}): "
            f"{', '.join(candidate.label for candidate in group)}. Shared basis: {key.describe()}."
        )
        lines.append("")
        lines.append(
            "Same valid-day count, same baseline mean RankIC, same per-fold frozen baseline "
            "selections and same per-fold dataset ids is exactly the condition under which two "
            "G4 deltas share a baseline term and a day set, so their difference is a legal "
            "subtraction."
        )
    for candidate, reason in excluded:
        lines.append("")
        lines.append(f"* **{candidate.label}** excluded from all delta arithmetic: {reason}.")
    lines.append("")

    lines.append("## Gate matrix (G1-G6)")
    lines.append("")
    lines.append("| Label | " + " | ".join(GATE_ORDER) + " | Verdict |")
    lines.append("|---" * (len(GATE_ORDER) + 2) + "|")
    for candidate in candidates:
        cells = [candidate.label]
        by_gate = {item.gate: item for item in candidate.gates}
        for gate in GATE_ORDER:
            item = by_gate.get(gate)
            cells.append(f"{_fmt(item.observed_value)} {item.status}" if item else "- MISSING")
        cells.append(candidate.verdict)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    definitions = "; ".join(f"{gate} {GATE_DESCRIPTIONS[gate]}" for gate in GATE_ORDER)
    lines.append(f"Gate definitions: {definitions}")
    lines.append("")

    lines.append("## Standings")
    lines.append("")
    lines.append(
        "| Label | Grid (max configs/fold) | Validation-selection free | Mean RankIC | ICIR | "
        "HAC t | G4 delta vs baseline | Verdict |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.delta_rank_ic is None,
            -(candidate.delta_rank_ic or 0.0),
            _first_index(list(candidates), candidate),
        ),
    )
    for candidate in ordered:
        lines.append(
            f"| {candidate.label} | {candidate.configs_searched_max} | "
            f"{'yes' if candidate.selection_free else 'no'} | "
            f"{_fmt(candidate.mean_rank_ic)} | {_fmt(candidate.icir)} | "
            f"{_fmt(candidate.hac_t_statistic, 3)} | "
            f"{_fmt_signed(candidate.delta_rank_ic)} | {candidate.verdict} |"
        )
    lines.append("")
    lines.append(
        "A candidate whose grid is larger than one config chose its winner on the validation "
        "split, so its OOS delta carries validation-selection optimism; a one-config candidate "
        "carries none. Read a gap between the two as 'the wider grid is still behind by that "
        "much, with bias in its favour'."
    )
    lines.append("")

    lines.append("## Pairwise increments (delta RankIC, row minus column)")
    lines.append("")
    if key is None or len(group) < 2:
        lines.append("_Not rendered: fewer than two candidates in the comparability group._")
    else:
        header = "| row \\ col | " + " | ".join(candidate.label for candidate in group) + " |"
        lines.append(header)
        lines.append("|---" * (len(group) + 1) + "|")
        for left in group:
            cells = [left.label]
            for right in group:
                if left is right:
                    cells.append("0")
                    continue
                if None in (left.delta_rank_ic, right.delta_rank_ic):
                    cells.append("n/a")
                    continue
                gap = cast(float, left.delta_rank_ic) - cast(float, right.delta_rank_ic)
                cells.append(
                    _fmt_signed(gap) if _identity_holds(left, right) else "IDENTITY_VIOLATION"
                )
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(
            "Because every candidate in the group shares one baseline term, `delta(a) - delta(b)` "
            "must equal `mean(a) - mean(b)`; each cell is printed only after that identity is "
            f"checked to {IDENTITY_TOLERANCE:g}. A cell reading `IDENTITY_VIOLATION` means the two "
            "reports disagree about their own arithmetic and neither should be quoted."
        )
    lines.append("")

    lines.append("## Leading candidate")
    lines.append("")
    if leader is None:
        lines.append(f"**{NO_CANDIDATE_PASSED_RECORD}**")
    else:
        lines.append(
            f"**{leader.label}** ({leader.family}, G4 {_fmt_signed(leader.delta_rank_ic)}, "
            f"grid {leader.configs_searched_max} config(s) per fold, "
            f"validation-selection free: {'yes' if leader.selection_free else 'no'})."
        )
        lines.append("")
        lines.append("### Yearly OOS distribution")
        lines.append("")
        lines.append("| Year | Mean RankIC |")
        lines.append("|---|---|")
        for year, value in leader.yearly:
            lines.append(f"| {year} | {_fmt(value)} |")
        lines.append("")
        lines.append("### Per-fold winner and worst seed (validation stage)")
        lines.append("")
        lines.append(
            "| Fold | Model | Winning config | Searched | Ens. val RankIC | Worst seed | "
            "Its val RankIC | All seeds |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        worst_overall: tuple[str, int, float] | None = None
        for fold_id in REQUIRED_FOLDS:
            fold = leader.fold(fold_id)
            if fold is None:
                lines.append(f"| {fold_id} | MISSING | - | - | - | - | - | - |")
                continue
            seeds_text = ", ".join(
                f"{seed}:{value:.6f}" for seed, value in fold.seed_validation_rank_ic
            )
            lines.append(
                f"| {fold_id} | {fold.model_id} | {_config_text(fold.best_config)} | "
                f"{fold.configs_searched} | {_fmt(fold.ensemble_validation_rank_ic)} | "
                f"{fold.worst_seed if fold.worst_seed is not None else '-'} | "
                f"{_fmt(fold.worst_seed_validation_rank_ic)} | {seeds_text or '-'} |"
            )
            seed_ic = fold.worst_seed_validation_rank_ic
            if (
                fold.worst_seed is not None
                and seed_ic is not None
                and (worst_overall is None or seed_ic < worst_overall[2])
            ):
                worst_overall = (fold_id, fold.worst_seed, seed_ic)
        lines.append("")
        if worst_overall is None:
            lines.append(
                "Worst seed: **not derivable** (no per-seed validation RankIC in the report)."
            )
        else:
            lines.append(
                f"Worst single seed across folds: **{worst_overall[1]} on {worst_overall[0]}** at "
                f"{worst_overall[2]:.6f} validation RankIC. These are validation-stage numbers: "
                "the signed reports do not carry a per-seed OOS breakdown, so an OOS worst seed is "
                "**not derivable** from them and is not printed here."
            )
    lines.append("")

    if baseline_report is not None:
        lines.append("## Baseline run (the term every G4 subtracts)")
        lines.append("")
        baseline_oos = cast(dict[str, Any], baseline_report.get("oos", {}))
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Valid day count | {baseline_oos.get('valid_day_count', '-')} |")
        lines.append(f"| Mean RankIC | {_fmt(_as_float(baseline_oos.get('mean_rank_ic')))} |")
        lines.append(f"| ICIR | {_fmt(_as_float(baseline_oos.get('icir')))} |")
        lines.append(
            f"| HAC t-statistic | {_fmt(_as_float(baseline_oos.get('hac_t_statistic')), 3)} |"
        )
        yearly_raw = cast(dict[str, Any], baseline_oos.get("yearly_mean_rank_ic", {}))
        for year in REQUIRED_YEARS:
            lines.append(f"| Mean RankIC {year} | {_fmt(_as_float(yearly_raw.get(year)))} |")
        selections = {
            str(record.get("fold_id")): str(record.get("selected_baseline_id"))
            for record in cast(list[Any], baseline_report.get("selections", []))
            if isinstance(record, dict)
        }
        lines.append(
            "| Fold-matched best_simple | "
            + ", ".join(f"{fold}={selections.get(fold, '-')}" for fold in REQUIRED_FOLDS)
            + " |"
        )
        lines.append("")
        lines.append(
            "**Do not subtract these yearly values from a candidate's yearly values.** The "
            "baseline run scores its own day set (it sees only the dense baseline-eligible "
            "factors), while each candidate's report carries the baseline *recomputed on the "
            "candidate's day set* as `baseline_mean_rank_ic`. The pooled figures differ for "
            "exactly that reason, and the signed reports contain no per-year baseline on the "
            "candidate day set, so per-year deltas are not derivable from them."
        )
        lines.append("")

    lines.append("## Limits of this document")
    lines.append("")
    lines.append(
        "* RankIC only: no transaction cost, impact cost, capacity or turnover constraint is "
        "applied anywhere in these numbers."
    )
    lines.append(
        "* The six gates are a joint threshold, not a multiple-testing correction; a wider grid "
        "raises the chance of clearing G4 on validation-selected luck."
    )
    lines.append(
        "* A candidate whose grid exceeds one config is a per-fold ensemble of up to four "
        "different winning architectures, not one model specification."
    )
    lines.append("* The locked 2026 segment is not part of any of these runs.")
    lines.append("")
    return "\n".join(lines)


def _parse_spec(spec: str) -> tuple[str, Path]:
    label, separator, path = spec.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError(
            f"--report expects LABEL=PATH (got {spec!r}); the label is what the rendered "
            "tables call this run, so it must be explicit"
        )
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=_parse_spec,
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="a signed full-mode walk-forward report; repeatable, order is preserved",
    )
    parser.add_argument("--baseline-report", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--print", action="store_true", dest="echo")
    args = parser.parse_args()

    labels = [label for label, _ in args.report]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"--report labels must be unique, got {labels}")
    candidates = tuple(load_candidate(label, path) for label, path in args.report)
    baseline_report = (
        cast(dict[str, Any], json.loads(args.baseline_report.read_text(encoding="utf-8")))
        if args.baseline_report is not None
        else None
    )
    markdown = render_markdown(candidates, baseline_report=baseline_report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")
    if args.echo:
        print(markdown)
    leader = leading_candidate(split_comparable(candidates)[1])
    if leader is None:
        print(NO_CANDIDATE_PASSED_RECORD)
    else:
        print(f"leading candidate: {leader.label} (G4 {_fmt_signed(leader.delta_rank_ic)})")


if __name__ == "__main__":
    main()
