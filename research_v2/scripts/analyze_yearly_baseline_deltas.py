#!/usr/bin/env python3
"""Per-year baseline RankIC on the candidate's own day set: the number the reports do not carry.

The gap this closes. A signed candidate report carries its own ``yearly_mean_rank_ic`` (on the
candidate's scoreable days, 945 of them for the frozen-v2 runs) and a *pooled*
``baseline_mean_rank_ic`` recomputed on those same days. The baseline run's own report carries
per-year values, but on the baseline's day set (969 days, because the dense-only baseline can
score days where the six sparse candidate-only factors are NaN). So no artifact holds the
baseline's per-year RankIC on the candidate's day set, and a per-year increment over the
baseline -- the most informative table in any write-up of these runs -- was not derivable.

It is derivable from what is already published, without recomputing any model: the evaluator
itself builds the baseline's daily RankIC on the matched coverage in order to produce the pooled
``baseline_mean_rank_ic`` (``model_evaluator.evaluate_frozen_oos``, the baseline branch). This
script replays exactly that branch and adds the one call the evaluator does not make,
``summarize_rank_ic_by_year`` on the baseline's daily frame.

Nothing here is trusted on argument. Two acceptance checks must pass, both bit-for-bit, before
a single new number is printed:

* **A** the replayed evaluation must reproduce the signed report's pooled mean RankIC, baseline
  mean RankIC, delta, valid-day count and all four per-year candidate values;
* **B** the replayed baseline branch must reproduce the evaluator's own pooled
  ``baseline_mean_rank_ic``.

B is what makes the per-year split credible: the per-year numbers come from the same daily frame
whose pooled mean is already published, so if the pooled mean matches, the split of it is a
partition of a number the report already vouches for. On any mismatch the script prints the
mismatch and exits non-zero without printing the new numbers.

Read-only: it opens signed artifacts to verify and read them, and writes only to ``--out-*``.
No database access, no model loading, no weight reading -- the scores are the published ones.

Usage:
    QRV2_SELECTION_EVIDENCE_HMAC_KEY=... (or --key PATH)
    analyze_yearly_baseline_deltas.py --workspace .../m0r_v2 --report .../m0r_v2/report.json \
        [--key .../secrets/selection_evidence_hmac.key] [--out-json X.json] [--out-md X.md]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402

from research_v2.datasets.storage import DatasetArtifactStore  # noqa: E402
from research_v2.evaluation.model_evaluator import (  # noqa: E402
    _match_prediction_coverage,
    _valid_label_day_count,
)
from research_v2.evaluation.model_walk_forward import (  # noqa: E402
    _labels_with_lineage,
    _prediction_with_lineage,
    _require_best_simple_evidence,
    _require_candidate_evidence,
    _require_verified_model_bundle,
)
from research_v2.evaluation.rank_ic import (  # noqa: E402
    compute_daily_rank_ic,
    summarize_rank_ic,
    summarize_rank_ic_by_year,
)
from research_v2.evaluation.selection_evidence import (  # noqa: E402
    FrozenPredictionStore,
    SelectionEvidenceVerifier,
)
from research_v2.evaluation.walk_forward import (  # noqa: E402
    OosFold,
    _assemble_oos_frames,
    evaluate_walk_forward_oos,
)
from research_v2.training.model_bundle import ModelArtifactBundleStore  # noqa: E402

FOLDS: tuple[str, ...] = ("F1", "F2", "F3", "F4")
MINIMUM_CROSS_SECTION = 30  # what every real run passes; see run_real_candidate_training.py
SELECTION_KEY_ENV = "QRV2_SELECTION_EVIDENCE_HMAC_KEY"
PREDICTION_KEY_ENV = "QRV2_PREDICTION_EVIDENCE_HMAC_KEY"


def prediction_ids_by_fold(workspace: Path) -> dict[str, dict[str, str]]:
    """{fold: {"candidate": id, "best_simple": id}} read from the signed prediction manifests."""
    found: dict[str, dict[str, str]] = {}
    root = workspace / "predictions"
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        metadata = manifest.get("metadata", {})
        fold = str(metadata.get("fold_id"))
        kind = str(metadata.get("kind"))
        if kind not in {"candidate", "best_simple"}:
            continue
        found.setdefault(fold, {})[kind] = str(manifest["prediction_id"])
    return found


def assemble_oos_folds(
    workspace: Path, report: dict[str, Any], environ: dict[str, str]
) -> dict[str, OosFold]:
    """Rebuild the evaluator's four OosFold triples, running every evidence check it runs."""
    selections = SelectionEvidenceVerifier.from_environment(
        workspace / "selections", environ=environ
    )
    predictions = FrozenPredictionStore.from_environment(workspace / "predictions", environ=environ)
    bundles = ModelArtifactBundleStore(workspace / "models")
    datasets = DatasetArtifactStore(workspace / "datasets")
    ids = prediction_ids_by_fold(workspace)
    selection_by_fold = {
        str(fold.get("fold_id")): str(fold.get("model_selection_id"))
        for fold in report.get("folds", [])
    }
    oos_folds: dict[str, OosFold] = {}
    for fold in FOLDS:
        if fold not in selection_by_fold:
            raise ValueError(f"{fold}: the report names no model_selection_id")
        kinds = ids.get(fold, {})
        if "candidate" not in kinds or "best_simple" not in kinds:
            raise ValueError(f"{fold}: prediction artifacts found for kinds {sorted(kinds)}")
        selection = selections.load_model_selection(selection_by_fold[fold])
        if selection.fold_id != fold:
            raise ValueError(f"{fold}: selection evidence is bound to {selection.fold_id}")
        _require_verified_model_bundle(bundles, selection)
        candidate = predictions.load(kinds["candidate"], selections=selections)
        baseline = predictions.load(kinds["best_simple"], selections=selections)
        _require_candidate_evidence(candidate, selection, fold)
        baseline_selection = selections.load_baseline_selection(selection.baseline_selection_id)
        _require_best_simple_evidence(
            baseline,
            baseline_selection.evidence_id,
            baseline_selection.selected_baseline_id,
            fold,
            selection.test_dataset_id,
        )
        test_frame = datasets.load_split(selection.test_dataset_id, "test", expected_fold_id=fold)
        labels = _labels_with_lineage(test_frame, fold, selection.test_dataset_id)
        oos_folds[fold] = OosFold(
            candidate_predictions=_prediction_with_lineage(candidate.scores, candidate, fold),
            labels=labels,
            baseline_predictions=_prediction_with_lineage(baseline.scores, baseline, fold),
        )
        print(
            f"  {fold}: candidate {len(candidate.scores)} rows, "
            f"best_simple {len(baseline.scores)} rows ({baseline_selection.selected_baseline_id}), "
            f"labels {len(labels)} rows -- evidence chain verified",
            flush=True,
        )
    return oos_folds


def _same(left: Any, right: Any) -> bool:
    """Bit-for-bit equality for floats, so 'reproduces the report' cannot mean 'close enough'."""
    if isinstance(left, float) and isinstance(right, float):
        return repr(left) == repr(right)
    return left == right


def rebuilt_pooled_delta(rows: list[dict[str, Any]]) -> tuple[float, int]:
    """Day-weighted mean of the per-year deltas, and the day count it was weighted over.

    The pooled delta a report publishes IS this quantity: the pooled mean is the mean over all
    valid days and the yearly means partition those same days. Recomputing it from the partition
    is acceptance check C, and C is what ties the new per-year baseline column to a number the
    signed report already vouches for.
    """
    weighted = 0.0
    total_days = 0
    for row in rows:
        if row.get("status") != "OK":
            continue
        days = int(row["candidate_valid_days"])
        weighted += float(row["delta_rank_ic"]) * days
        total_days += days
    return (weighted / total_days if total_days else float("nan")), total_days


def sensitivity_without_score_health(
    candidates: Any, baseline_matched: Any, labels: Any
) -> dict[str, Any]:
    """Re-measure everything with the score-health floor switched off, and say what moved.

    Why this exists: the 24 OOS days a frozen-v2 candidate cannot be evaluated on are not days
    without scores -- they carry about 3,000 scored rows each -- they are days whose coverage
    (scored rows over valid-label rows) measures 0.895-0.899 against the production floor of
    exactly 0.90 in ``ScoreHealthEvidence.passes``. A margin that thin is a legitimate question
    from a reader: does the result depend on the floor? This answers it by re-running the same
    metric with ``require_score_health=False`` and reporting the pooled and per-year deltas side
    by side. It is a sensitivity check, NOT an alternative official number: the gates a report
    publishes are computed with the floor on, and nothing here rewrites them.
    """
    candidate_daily = compute_daily_rank_ic(
        candidates,
        labels,
        minimum_cross_section=MINIMUM_CROSS_SECTION,
        include_invalid=True,
        require_score_health=False,
    )
    baseline_daily = compute_daily_rank_ic(
        baseline_matched,
        labels,
        minimum_cross_section=MINIMUM_CROSS_SECTION,
        include_invalid=True,
        require_score_health=False,
    )
    candidate_summary = summarize_rank_ic(candidate_daily)
    baseline_summary = summarize_rank_ic(baseline_daily)
    candidate_yearly = summarize_rank_ic_by_year(candidate_daily)
    baseline_yearly = summarize_rank_ic_by_year(baseline_daily)
    yearly = []
    for year in range(2022, 2026):
        crow = candidate_yearly.loc[candidate_yearly["year"] == year]
        brow = baseline_yearly.loc[baseline_yearly["year"] == year]
        if crow.empty or brow.empty:
            yearly.append({"year": year, "status": "MISSING"})
            continue
        candidate_mean = float(crow["mean_rank_ic"].iloc[0])
        baseline_mean = float(brow["mean_rank_ic"].iloc[0])
        yearly.append(
            {
                "year": year,
                "candidate_mean_rank_ic": candidate_mean,
                "candidate_valid_days": int(crow["valid_day_count"].iloc[0]),
                "baseline_mean_rank_ic": baseline_mean,
                "baseline_valid_days": int(brow["valid_day_count"].iloc[0]),
                "delta_rank_ic": candidate_mean - baseline_mean,
                "status": "OK",
            }
        )
    return {
        "require_score_health": False,
        "pooled": {
            "candidate_mean_rank_ic": candidate_summary.mean_rank_ic,
            "candidate_valid_days": candidate_summary.valid_day_count,
            "baseline_mean_rank_ic": baseline_summary.mean_rank_ic,
            "baseline_valid_days": baseline_summary.valid_day_count,
            "delta_rank_ic": candidate_summary.mean_rank_ic - baseline_summary.mean_rank_ic,
        },
        "yearly": yearly,
    }


def describe_unevaluable_days(
    candidate_daily: pd.DataFrame, baseline_daily: pd.DataFrame, label_days: int
) -> dict[str, Any]:
    """Characterise the OOS days the candidate cannot be evaluated on, per year.

    `compute_daily_rank_ic(include_invalid=True)` keeps every label date with the diagnostics that
    decided it, so this reads the reason off the frame instead of guessing at it: a day is
    unevaluable when fewer than `minimum_cross_section` rows carry both a valid label and a finite
    score, or when the score-health thresholds (coverage, unique scores, spread) reject it. The
    median coverage and sample count on those days say which of the two it was.
    """
    def per_year(frame: pd.DataFrame) -> dict[int, dict[str, Any]]:
        working = frame.copy()
        working["year"] = pd.to_datetime(working["signal_date"]).dt.year
        out: dict[int, dict[str, Any]] = {}
        for year, group in working.groupby("year", sort=True):
            invalid = group.loc[~group["rank_ic_valid"].astype(bool)]
            out[int(year)] = {
                "label_dates": int(len(group)),
                "valid_days": int(len(group) - len(invalid)),
                "invalid_days": int(len(invalid)),
                "median_coverage_on_invalid_days": (
                    float(invalid["coverage"].median()) if len(invalid) else None
                ),
                "median_sample_count_on_invalid_days": (
                    float(invalid["sample_count"].median()) if len(invalid) else None
                ),
                "median_valid_label_count_on_invalid_days": (
                    float(invalid["valid_label_count"].median()) if len(invalid) else None
                ),
                "invalid_dates": [
                    str(day) for day in sorted(invalid["signal_date"].astype(str).tolist())
                ],
            }
        return out

    candidate_by_year = per_year(candidate_daily)
    baseline_by_year = per_year(baseline_daily)
    for year, entry in candidate_by_year.items():
        baseline_entry = baseline_by_year.get(year, {})
        entry["baseline_invalid_days_same_frame"] = baseline_entry.get("invalid_days")
    candidate_invalid = sum(entry["invalid_days"] for entry in candidate_by_year.values())
    baseline_invalid = sum(entry["invalid_days"] for entry in baseline_by_year.values())
    return {
        "label_day_count": label_days,
        "candidate_invalid_days": candidate_invalid,
        "baseline_invalid_days_on_the_same_frame": baseline_invalid,
        "by_year": candidate_by_year,
    }


def analyse(
    workspace: Path,
    report_path: Path,
    environ: dict[str, str],
    *,
    without_score_health: bool = False,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    oos = report.get("oos", {})
    print(f"assembling the OOS frames for {workspace.name} from signed artifacts")
    oos_folds = assemble_oos_folds(workspace, report, environ)

    evaluation = evaluate_walk_forward_oos(
        oos_folds, minimum_cross_section=MINIMUM_CROSS_SECTION, is_baseline=False
    )
    mismatches: list[str] = []
    expected = {
        "mean_rank_ic": (evaluation.summary.mean_rank_ic, oos.get("mean_rank_ic")),
        "baseline_mean_rank_ic": (
            evaluation.baseline_mean_rank_ic,
            oos.get("baseline_mean_rank_ic"),
        ),
        "delta_rank_ic": (evaluation.delta_rank_ic, oos.get("delta_rank_ic")),
        "valid_day_count": (evaluation.summary.valid_day_count, oos.get("valid_day_count")),
    }
    for name, (got, want) in expected.items():
        if not _same(got, want):
            mismatches.append(f"A/{name}: replayed {got!r} vs report {want!r}")
    for year, value in evaluation.yearly_mean_rank_ic.items():
        want = (oos.get("yearly_mean_rank_ic") or {}).get(str(year))
        if not _same(value, want):
            mismatches.append(f"A/yearly {year}: replayed {value!r} vs report {want!r}")

    candidates, labels, baselines = _assemble_oos_frames(oos_folds)
    _candidate_matched, baseline_matched = _match_prediction_coverage(candidates, baselines)
    baseline_daily = compute_daily_rank_ic(
        baseline_matched,
        labels,
        minimum_cross_section=MINIMUM_CROSS_SECTION,
        include_invalid=True,
        require_score_health=True,
    )
    baseline_summary = summarize_rank_ic(baseline_daily)
    if not _same(baseline_summary.mean_rank_ic, evaluation.baseline_mean_rank_ic):
        mismatches.append(
            "B/baseline pooled mean: replayed baseline branch "
            f"{baseline_summary.mean_rank_ic!r} vs the evaluator's "
            f"{evaluation.baseline_mean_rank_ic!r}"
        )

    # The same baseline scores on the coverage the baseline artifact itself carries, i.e. before
    # _match_prediction_coverage restricts them to the rows the candidate can score. The
    # difference between this and the matched frame is exactly the set of OOS days/rows the
    # candidate cannot be compared on, which is the caveat any per-year table has to carry.
    baseline_full_daily = compute_daily_rank_ic(
        baselines,
        labels,
        minimum_cross_section=MINIMUM_CROSS_SECTION,
        include_invalid=True,
        require_score_health=True,
    )
    baseline_full_summary = summarize_rank_ic(baseline_full_daily)

    # G6's denominator, measured here rather than read from the report: the days on which the
    # label frame itself offers a scoreable cross-section. The gap between it and the candidate's
    # valid days is the set of OOS days the candidate cannot be evaluated on, which is the number
    # a per-year table has to disclose and which no report states outright.
    label_days = _valid_label_day_count(labels, MINIMUM_CROSS_SECTION)

    if mismatches:
        return {"accepted": False, "mismatches": mismatches}

    candidate_yearly = summarize_rank_ic_by_year(evaluation.daily_rank_ic)
    baseline_yearly = summarize_rank_ic_by_year(baseline_daily)
    baseline_full_yearly = summarize_rank_ic_by_year(baseline_full_daily)
    rows: list[dict[str, Any]] = []
    for year in range(2022, 2026):
        crow = candidate_yearly.loc[candidate_yearly["year"] == year]
        brow = baseline_yearly.loc[baseline_yearly["year"] == year]
        frow = baseline_full_yearly.loc[baseline_full_yearly["year"] == year]
        if crow.empty or brow.empty:
            rows.append({"year": year, "status": "MISSING"})
            continue
        candidate_mean = float(crow["mean_rank_ic"].iloc[0])
        baseline_mean = float(brow["mean_rank_ic"].iloc[0])
        days = int(crow["valid_day_count"].iloc[0])
        delta = candidate_mean - baseline_mean
        full_days = int(frow["valid_day_count"].iloc[0]) if not frow.empty else 0
        full_mean = float(frow["mean_rank_ic"].iloc[0]) if not frow.empty else float("nan")
        matched_days = int(brow["valid_day_count"].iloc[0])
        lost_days = full_days - matched_days
        lost_mean = (
            (full_mean * full_days - baseline_mean * matched_days) / lost_days
            if lost_days > 0
            else None
        )
        rows.append(
            {
                "year": year,
                "candidate_mean_rank_ic": candidate_mean,
                "candidate_valid_days": days,
                "baseline_mean_rank_ic_on_candidate_days": baseline_mean,
                "baseline_valid_days": matched_days,
                "delta_rank_ic": delta,
                "baseline_valid_days_before_coverage_matching": full_days,
                "baseline_mean_rank_ic_before_coverage_matching": full_mean,
                "baseline_days_lost_to_coverage_matching": lost_days,
                "baseline_mean_rank_ic_on_days_lost_to_matching": lost_mean,
                "status": "OK",
            }
        )
    # Acceptance C: the yearly deltas must rebuild the pooled delta the report publishes. This is
    # arithmetic, not a tolerance for sloppiness -- the pooled mean IS the day-weighted mean of the
    # yearly means over the same valid days, so a partition that does not add back up means one of
    # the two frames is not the frame the pooled number came from.
    pooled_delta = evaluation.delta_rank_ic
    rebuilt, total_days = rebuilt_pooled_delta(rows)
    if pooled_delta is None or abs(rebuilt - pooled_delta) > 1e-12:
        mismatches.append(f"C/day-weighted rebuild: {rebuilt!r} vs pooled {pooled_delta!r}")
    gate_g6 = ((oos.get("gates") or {}).get("G6") or {}).get("observed_value")
    rebuilt_g6 = (
        evaluation.summary.valid_day_count / label_days if label_days else float("nan")
    )
    if gate_g6 is None or abs(rebuilt_g6 - float(gate_g6)) > 1e-12:
        mismatches.append(f"D/G6 rebuild: {rebuilt_g6!r} vs report {gate_g6!r}")
    unevaluable = describe_unevaluable_days(
        evaluation.daily_rank_ic, baseline_daily, label_days
    )
    if mismatches:
        return {"accepted": False, "mismatches": mismatches}
    return {
        "accepted": True,
        "workspace": str(workspace),
        "report": str(report_path),
        "report_sha256": _sha256(report_path),
        "family": report.get("family"),
        "minimum_cross_section": MINIMUM_CROSS_SECTION,
        "acceptance": {
            "A_reproduces_the_signed_report": {
                name: {"replayed": got, "report": want} for name, (got, want) in expected.items()
            },
            "B_baseline_branch_reproduces_the_pooled_baseline": {
                "replayed": baseline_summary.mean_rank_ic,
                "evaluator": evaluation.baseline_mean_rank_ic,
            },
            "C_day_weighted_yearly_deltas_rebuild_the_pooled_delta": {
                "rebuilt": rebuilt,
                "pooled": pooled_delta,
                "total_days": total_days,
            },
            "D_valid_day_ratio_rebuilds_g6": {
                "rebuilt": rebuilt_g6,
                "report": gate_g6,
                "label_day_count": label_days,
            },
        },
        "unevaluable_days": unevaluable,
        "pooled": {
            "candidate_mean_rank_ic": evaluation.summary.mean_rank_ic,
            "baseline_mean_rank_ic_on_candidate_days": baseline_summary.mean_rank_ic,
            "baseline_mean_rank_ic_before_coverage_matching": baseline_full_summary.mean_rank_ic,
            "baseline_valid_days_before_coverage_matching": baseline_full_summary.valid_day_count,
            "delta_rank_ic": evaluation.delta_rank_ic,
            "candidate_valid_days": evaluation.summary.valid_day_count,
            "baseline_valid_days": baseline_summary.valid_day_count,
        },
        "yearly": rows,
        "sensitivity_without_score_health": (
            sensitivity_without_score_health(candidates, baseline_matched, labels)
            if without_score_health
            else None
        ),
    }


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell(value: Any) -> str:
    """One markdown table cell: None stays a dash, floats keep four decimals, ints stay ints."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# Per-year baseline RankIC on the candidate's day set: {result['family']}",
        "",
        f"Source report `{result['report']}` (sha256 `{result['report_sha256'][:16]}`).",
        "Both acceptance checks passed bit-for-bit, so the per-year baseline column is a",
        "partition of the pooled baseline mean the signed report already publishes.",
        "",
        "| Year | Candidate mean RankIC | Days | Baseline on the same days | Delta | "
        "Baseline before coverage matching | Days | Days lost to matching | Baseline mean there |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in result["yearly"]:
        if row.get("status") != "OK":
            lines.append(f"| {row['year']} | MISSING | - | - | - | - | - | - | - |")
            continue
        lost_mean = row["baseline_mean_rank_ic_on_days_lost_to_matching"]
        lines.append(
            f"| {row['year']} | {row['candidate_mean_rank_ic']:.6f} | "
            f"{row['candidate_valid_days']} | "
            f"{row['baseline_mean_rank_ic_on_candidate_days']:.6f} | "
            f"{row['delta_rank_ic']:+.6f} | "
            f"{row['baseline_mean_rank_ic_before_coverage_matching']:.6f} | "
            f"{row['baseline_valid_days_before_coverage_matching']} | "
            f"{row['baseline_days_lost_to_coverage_matching']} | "
            f"{'-' if lost_mean is None else f'{lost_mean:.6f}'} |"
        )
    pooled = result["pooled"]
    unevaluable = result.get("unevaluable_days", {})
    lines += [
        "",
        f"Pooled: candidate {pooled['candidate_mean_rank_ic']:.6f} over "
        f"{pooled['candidate_valid_days']} days, baseline "
        f"{pooled['baseline_mean_rank_ic_on_candidate_days']:.6f} over "
        f"{pooled['baseline_valid_days']} days, delta {pooled['delta_rank_ic']:+.6f}. "
        f"Before coverage matching the same baseline artifact averages "
        f"{pooled['baseline_mean_rank_ic_before_coverage_matching']:.6f} over "
        f"{pooled['baseline_valid_days_before_coverage_matching']} days.",
        "",
        "These columns are all computed inside this candidate's workspace, on this candidate's "
        "label frame. The baseline run's OWN report is a separate evaluation on a different "
        "(dense-only) dataset and a different valid-day count; its numbers are not reproduced "
        "here and must not be mixed into this table.",
        "",
        "Acceptance: A the replay reproduces the signed report bit for bit; B the replayed "
        "baseline branch reproduces the evaluator's pooled baseline mean; C the day-weighted "
        "yearly deltas rebuild the published pooled delta to 1e-12; D the valid-day ratio "
        "rebuilds the report's G6 to 1e-12.",
        "",
        "## Days the candidate cannot be evaluated on",
        "",
        f"The label frame offers a scoreable cross-section on "
        f"{unevaluable.get('label_day_count', '-')} OOS days (that count is G6's denominator). "
        f"The candidate is evaluated on {pooled['candidate_valid_days']} of them and is "
        f"unevaluable on {unevaluable.get('candidate_invalid_days', '-')}; the baseline scores "
        f"carried in this same workspace are unevaluable on "
        f"{unevaluable.get('baseline_invalid_days_on_the_same_frame', '-')} of them.",
        "",
        "| Year | Label dates | Candidate valid | Candidate unevaluable | "
        "Baseline unevaluable (same frame) | Median coverage on those days | "
        "Median scored rows | Median valid-label rows |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for year, entry in sorted((unevaluable.get("by_year") or {}).items()):
        lines.append(
            f"| {year} | {entry['label_dates']} | {entry['valid_days']} | "
            f"{entry['invalid_days']} | "
            f"{_cell(entry.get('baseline_invalid_days_same_frame'))} | "
            f"{_cell(entry.get('median_coverage_on_invalid_days'))} | "
            f"{_cell(entry.get('median_sample_count_on_invalid_days'))} | "
            f"{_cell(entry.get('median_valid_label_count_on_invalid_days'))} |"
        )
    lines += [
        "",
        "A day is unevaluable when fewer than 30 rows carry both a valid label and a finite score, "
        "or when the production score-health thresholds reject it -- "
        "`ScoreHealthEvidence.passes` requires `0.90 <= coverage <= 1.0`, at least 10 unique "
        "scores and a score standard deviation above 1e-6. The median coverage column says which "
        "of the two it was, and the exact dates are in the JSON output.",
        "",
        "Method: the frames are rebuilt from the signed selection, prediction and dataset",
        "artifacts with the evaluator's own checks; the baseline branch is a replay of",
        "`model_evaluator.evaluate_frozen_oos`'s baseline path plus `summarize_rank_ic_by_year`.",
        "",
    ]
    sensitivity = result.get("sensitivity_without_score_health")
    if sensitivity:
        pooled_off = sensitivity["pooled"]
        lines += [
            "## Sensitivity: the same measurement with the score-health floor off",
            "",
            "| Year | Candidate mean RankIC | Days | Baseline | Days | Delta |",
            "|---|---|---|---|---|---|",
        ]
        for row in sensitivity["yearly"]:
            if row.get("status") != "OK":
                lines.append(f"| {row['year']} | MISSING | - | - | - | - |")
                continue
            lines.append(
                f"| {row['year']} | {row['candidate_mean_rank_ic']:.6f} | "
                f"{row['candidate_valid_days']} | {row['baseline_mean_rank_ic']:.6f} | "
                f"{row['baseline_valid_days']} | {row['delta_rank_ic']:+.6f} |"
            )
        lines += [
            "",
            f"Pooled with the floor off: candidate "
            f"{pooled_off['candidate_mean_rank_ic']:.6f} over "
            f"{pooled_off['candidate_valid_days']} days, baseline "
            f"{pooled_off['baseline_mean_rank_ic']:.6f} over "
            f"{pooled_off['baseline_valid_days']} days, delta "
            f"{pooled_off['delta_rank_ic']:+.6f} -- against the published "
            f"{pooled['delta_rank_ic']:+.6f} over {pooled['candidate_valid_days']} days.",
            "",
            "This is a sensitivity check, not an alternative official number: the gates in the "
            "signed report are computed with the floor on, and nothing here rewrites them.",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--key",
        type=Path,
        default=None,
        help="path to the HMAC key file; defaults to the two QRV2_*_HMAC_KEY env vars",
    )
    parser.add_argument(
        "--sensitivity-no-score-health",
        action="store_true",
        help="also re-measure with the 0.90 coverage floor off; a sensitivity check, "
        "not an alternative official number",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    environ = dict(os.environ)
    if args.key is not None:
        secret = args.key.read_text().strip()
        environ[SELECTION_KEY_ENV] = secret
        environ[PREDICTION_KEY_ENV] = secret
    if not environ.get(SELECTION_KEY_ENV) or not environ.get(PREDICTION_KEY_ENV):
        raise SystemExit(
            f"need the HMAC key: pass --key PATH or set {SELECTION_KEY_ENV} and "
            f"{PREDICTION_KEY_ENV}"
        )

    result = analyse(
        args.workspace,
        args.report,
        environ,
        without_score_health=args.sensitivity_no_score_health,
    )
    if not result["accepted"]:
        print("REFUSED: the replay does not reproduce the signed report, so no new number is "
              "printed. Mismatches:")
        for mismatch in result["mismatches"]:
            print(f"  {mismatch}")
        return 1
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    markdown = render_markdown(result)
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
