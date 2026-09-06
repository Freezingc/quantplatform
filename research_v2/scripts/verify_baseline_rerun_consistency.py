"""Verify a baseline walk-forward re-run reproduces the criterion-303 evidence.

Deterministic comparison of a new walk-forward report against an archived
baseline report: folds, per-fold selections (winner, validation RankIC, day
counts, unselected reasons), OOS statistics, gates, and the published
data-version mapping must match; every fold-matched baseline prediction
artifact must exist with the full lineage columns. Exit code 0 only when fully
consistent.

The comparison is driven by what the *reference* carries. The archived
criterion-303 report predates ``invalid_day_count``, ``unselected_reasons`` and
a recorded ``data_versions`` mapping, and Task 2.4's CONSISTENT verdict was
produced against it by a re-run that did carry them; requiring fields the
reference lacks would overturn a verdict that was already issued and relied on.
So a field absent from the reference is not required of the re-run, and a field
present in the reference is verified. That is what makes the verifier a stronger
trust root rather than a different one when Task 4.1 repoints ``--reference`` at
the frozen-set baseline report, which does carry all of them.

Usage:
  verify_baseline_rerun_consistency.py --report <new.json> \
      --reference docs/reports/2026-09-04-real-walk-forward-criterion-303.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

EXPECTED_FOLDS: tuple[str, ...] = ("F01", "F02", "F03", "F04", "F05", "F06")
EXPECTED_BASELINES: tuple[str, ...] = ("B0", "B1", "B2", "B3")
ARTIFACT_COLUMNS: tuple[str, ...] = (
    "signal_date",
    "ts_code",
    "score",
    "model_id",
    "fold_id",
    "dataset_id",
)


def _close(left: object, right: object) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
    return left == right


def compare_reports(new: dict[str, Any], reference: dict[str, Any]) -> list[str]:
    """Return human-readable inconsistencies (empty list == fully consistent)."""
    problems: list[str] = []
    if tuple(new.get("folds", [])) != tuple(reference.get("folds", [])):
        problems.append(f"folds differ: {new.get('folds')} vs {reference.get('folds')}")
    new_selections = {
        entry["fold_id"]: entry for entry in cast(list[Any], new.get("selections", []))
    }
    ref_selections = {
        entry["fold_id"]: entry
        for entry in cast(list[Any], reference.get("selections", []))
    }
    for fold_id in EXPECTED_FOLDS:
        if fold_id not in new_selections:
            problems.append(f"missing selection for {fold_id}")
            continue
        if fold_id not in ref_selections:
            problems.append(f"reference missing selection for {fold_id}")
            continue
        entered = new_selections[fold_id]
        archived = ref_selections[fold_id]
        if entered["selected_baseline_id"] != archived["selected_baseline_id"]:
            problems.append(
                f"{fold_id} winner differs: {entered['selected_baseline_id']} "
                f"vs {archived['selected_baseline_id']}"
            )
        entered_scores = {
            score["baseline_id"]: score for score in entered["validation_scores"]
        }
        archived_scores = {
            score["baseline_id"]: score for score in archived["validation_scores"]
        }
        if set(entered_scores) != set(EXPECTED_BASELINES):
            problems.append(f"{fold_id} lacks all four baseline scores")
            continue
        for baseline_id in EXPECTED_BASELINES:
            if baseline_id not in archived_scores:
                continue
            entered_score = cast(dict[str, Any], entered_scores[baseline_id])
            archived_score = cast(dict[str, Any], archived_scores[baseline_id])
            if not _close(
                entered_score["mean_rank_ic"],
                archived_score["mean_rank_ic"],
            ):
                problems.append(
                    f"{fold_id}/{baseline_id} validation RankIC differs: "
                    f"{entered_score['mean_rank_ic']} "
                    f"vs {archived_score['mean_rank_ic']}"
                )
            # Day counts are compared only when the reference records them. A mean RankIC
            # that matches to 1e-9 over 228 days is not the same evidence as one over 240,
            # and invalid_day_count is where a coverage regression would first show.
            for day_field in ("valid_day_count", "invalid_day_count"):
                if day_field not in archived_score:
                    continue
                if not _close(
                    entered_score.get(day_field), archived_score.get(day_field)
                ):
                    problems.append(
                        f"{fold_id}/{baseline_id} {day_field} differs: "
                        f"{entered_score.get(day_field)} vs {archived_score.get(day_field)}"
                    )
        if "unselected_reasons" in archived and entered.get(
            "unselected_reasons"
        ) != archived.get("unselected_reasons"):
            problems.append(
                f"{fold_id} unselected_reasons differ: "
                f"{entered.get('unselected_reasons')} vs {archived.get('unselected_reasons')}"
            )
    new_oos = cast(dict[str, Any], new.get("oos", {}))
    ref_oos = cast(dict[str, Any], reference.get("oos", {}))
    for key in (
        "valid_day_count",
        "mean_rank_ic",
        "icir",
        "hac_t_statistic",
        "baseline_mean_rank_ic",
    ):
        if not _close(new_oos.get(key), ref_oos.get(key)):
            problems.append(f"oos.{key} differs: {new_oos.get(key)} vs {ref_oos.get(key)}")
    new_gates = {
        gate: result["status"] for gate, result in new_oos.get("gates", {}).items()
    }
    ref_gates = {
        gate: result["status"] for gate, result in ref_oos.get("gates", {}).items()
    }
    if new_gates != ref_gates:
        problems.append(f"gate statuses differ: {new_gates} vs {ref_gates}")
    new_yearly = new_oos.get("yearly_mean_rank_ic", {})
    ref_yearly = ref_oos.get("yearly_mean_rank_ic", {})
    for year, value in ref_yearly.items():
        if not _close(new_yearly.get(year), value):
            problems.append(
                f"yearly {year} RankIC differs: {new_yearly.get(year)} vs {value}"
            )
    if "data_versions" in reference:
        new_versions = cast(dict[str, Any], new.get("data_versions") or {})
        ref_versions = cast(dict[str, Any], reference.get("data_versions") or {})
        if new_versions != ref_versions:
            tables = sorted(set(new_versions) | set(ref_versions))
            differing = [
                table for table in tables if new_versions.get(table) != ref_versions.get(table)
            ]
            problems.append(
                "data_versions differ for: "
                + (", ".join(differing) if differing else "<whole mapping>")
            )
    return problems


def check_artifacts(report: dict[str, Any]) -> list[str]:
    """Every declared fold-matched baseline prediction artifact must be readable."""
    problems: list[str] = []
    artifacts = report.get("baseline_oos_artifacts")
    if not isinstance(artifacts, dict):
        return ["report has no baseline_oos_artifacts mapping"]
    for fold_id in EXPECTED_FOLDS:
        path_value = artifacts.get(fold_id)
        if not isinstance(path_value, str):
            problems.append(f"missing artifact path for {fold_id}")
            continue
        target = Path(path_value)
        if not target.exists():
            problems.append(f"artifact missing on disk for {fold_id}: {target}")
            continue
        frame = pd.read_parquet(target)
        if tuple(frame.columns) != ARTIFACT_COLUMNS:
            problems.append(
                f"artifact columns for {fold_id}: {tuple(frame.columns)} "
                f"(expected {ARTIFACT_COLUMNS})"
            )
        if frame.empty:
            problems.append(f"artifact empty for {fold_id}")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    new = cast(dict[str, Any], json.loads(args.report.read_text(encoding="utf-8")))
    reference = cast(
        dict[str, Any], json.loads(args.reference.read_text(encoding="utf-8"))
    )
    problems = compare_reports(new, reference) + check_artifacts(new)
    if problems:
        for problem in problems:
            print(f"INCONSISTENT: {problem}")
        return 1
    print("CONSISTENT: re-run reproduces the criterion-303 evidence exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
