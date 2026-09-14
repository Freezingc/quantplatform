"""Tests for the cross-candidate comparison renderer (plan Task 5.1).

The point of most of these is not that the renderer prints tables -- it is that the two guards
which make its subtractions legal can be seen to fail. A comparison document that quietly
subtracted two G4 deltas measured on different day sets, or that named a gate-failing run as
the leading candidate, would be worse than no document at all, because it would be quoted.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import summarize_candidate_comparison as module  # noqa: E402

SEEDS: tuple[int, ...] = (20260801, 20260802, 20260803)
BASELINE_IC = 0.0852268662737014


def _gate(status: str, observed: float | None, threshold: float | None) -> dict[str, Any]:
    return {"status": status, "observed_value": observed, "threshold": threshold}


def _seed_values(ensemble: float, worst_seed: int) -> dict[str, float]:
    offsets = {SEEDS[0]: 0.0001, SEEDS[1]: 0.0002, SEEDS[2]: 0.0003}
    values = {str(seed): ensemble + offsets[seed] for seed in SEEDS}
    values[str(worst_seed)] = ensemble - 0.0005  # strictly the minimum of the three
    return values


def _search_results(configs: int, ensemble: float, worst_seed: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index in range(configs):
        value = ensemble if index == configs - 1 else ensemble - 0.01 * (configs - index)
        results.append(
            {
                "config": {"hidden_dim": 64 + index, "dropout": 0.1},
                "ensemble_validation_rank_ic": value,
                "seed_validation_rank_ic": _seed_values(value, worst_seed),
                "worst_seed": worst_seed,
            }
        )
    return results


def _fold(fold_id: str, *, ensemble: float, worst_seed: int, configs: int) -> dict[str, Any]:
    return {
        "fold_id": fold_id,
        # Per-fold ids depend on the fold only, so two candidates built from this fixture share
        # a comparison basis by construction and differ only where a test says so.
        "dataset_id": f"dataset-{fold_id.lower()}",
        "model_id": f"carrier-{fold_id.lower()}",
        "best_config": {"num_leaves": 15},
        "search_results": _search_results(configs, ensemble, worst_seed),
        "baseline_selection_id": f"baseline-{fold_id.lower()}",
        "model_selection_id": f"model-{fold_id.lower()}",
    }


def _report(
    *,
    family: str = "M0R",
    mode: str = "full",
    delta: float = 0.009,
    days: int = 945,
    baseline_ic: float = BASELINE_IC,
    all_passed: bool | None = None,
    folds: str = "all",
    configs: int = 1,
    worst_seed: int = 20260803,
) -> dict[str, Any]:
    passed = delta >= 0.005 if all_passed is None else all_passed
    fold_ids = list(module.REQUIRED_FOLDS) if folds == "all" else ["F1", "F2", "F3"]
    return {
        "family": family,
        "mode": mode,
        "folds": [
            _fold(
                fold_id,
                ensemble=0.09 + 0.001 * index,
                worst_seed=worst_seed,
                configs=configs,
            )
            for index, fold_id in enumerate(fold_ids)
        ],
        "oos": {
            "valid_day_count": days,
            "mean_rank_ic": baseline_ic + delta,
            "icir": 0.75,
            "hac_t_statistic": 12.5,
            "baseline_mean_rank_ic": baseline_ic,
            "delta_rank_ic": delta,
            "yearly_mean_rank_ic": {
                "2022": 0.0931,
                "2023": 0.0730,
                "2024": 0.1045,
                "2025": 0.1048,
            },
            "gates": {
                "G1": _gate("PASS", baseline_ic + delta, 0.03),
                "G2": _gate("PASS", 12.5, 2.0),
                "G3": _gate("PASS", 1.0, 0.7),
                "G4": _gate("PASS" if passed else "FAIL", delta, 0.005),
                "G5": _gate("PASS", 0.28, 0.5),
                "G6": _gate("PASS", 0.975, 0.95),
            },
            "all_passed": passed,
        },
    }


def _candidate(tmp_path: Path, label: str, report: dict[str, Any]) -> module.Candidate:
    path = tmp_path / f"{label}_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return module.load_candidate(label, path)


def _baseline_report() -> dict[str, Any]:
    return {
        "oos": {
            "valid_day_count": 969,
            "mean_rank_ic": 0.08705554536301544,
            "icir": 0.7507,
            "hac_t_statistic": 12.7498,
            "yearly_mean_rank_ic": {
                "2022": 0.0812,
                "2023": 0.0781,
                "2024": 0.0906,
                "2025": 0.0983,
            },
        },
        "selections": [
            {"fold_id": fold_id, "selected_baseline_id": "B2"}
            for fold_id in module.REQUIRED_FOLDS
        ],
    }


class TestComparability:
    def test_same_basis_forms_one_group(self, tmp_path: Path) -> None:
        group = (
            _candidate(tmp_path, "A", _report(delta=0.009)),
            _candidate(tmp_path, "B", _report(family="M1", delta=0.007, configs=27)),
        )
        key, comparable, excluded = module.split_comparable(group)
        assert key is not None
        assert [item.label for item in comparable] == ["A", "B"]
        assert excluded == []

    def test_different_day_count_is_excluded_with_the_reason(self, tmp_path: Path) -> None:
        group = (
            _candidate(tmp_path, "A", _report(delta=0.009)),
            _candidate(tmp_path, "B", _report(delta=0.012, days=969)),
        )
        _key, comparable, excluded = module.split_comparable(group)
        assert [item.label for item in comparable] == ["A"]
        assert [item.label for item, _ in excluded] == ["B"]
        reason = excluded[0][1]
        assert "different comparison basis" in reason
        assert "valid_day_count=969" in reason

    def test_different_baseline_term_is_excluded(self, tmp_path: Path) -> None:
        group = (
            _candidate(tmp_path, "A", _report(delta=0.009)),
            _candidate(tmp_path, "B", _report(delta=0.009, baseline_ic=0.0800)),
        )
        _key, comparable, excluded = module.split_comparable(group)
        assert [item.label for item in comparable] == ["A"]
        assert "baseline_mean_rank_ic" in excluded[0][1]

    def test_incomplete_folds_excluded_before_grouping(self, tmp_path: Path) -> None:
        group = (
            _candidate(tmp_path, "A", _report(delta=0.009)),
            _candidate(tmp_path, "B", _report(delta=0.020, folds="partial")),
        )
        _key, comparable, excluded = module.split_comparable(group)
        assert [item.label for item in comparable] == ["A"]
        assert "folds incomplete" in excluded[0][1]
        assert "F1,F2,F3" in excluded[0][1]

    def test_largest_group_is_the_reference_and_ties_follow_input_order(
        self, tmp_path: Path
    ) -> None:
        majority = tuple(
            _candidate(tmp_path, label, _report(delta=0.009)) for label in ("A", "B", "C")
        )
        outlier = (_candidate(tmp_path, "D", _report(delta=0.030, days=900)),)
        _key, comparable, excluded = module.split_comparable(majority + outlier)
        assert [item.label for item in comparable] == ["A", "B", "C"]
        assert [item.label for item, _ in excluded] == ["D"]

        tied = (
            _candidate(tmp_path, "X", _report(delta=0.009, days=900)),
            _candidate(tmp_path, "Y", _report(delta=0.009)),
        )
        key, comparable, _excluded = module.split_comparable(tied)
        assert [item.label for item in comparable] == ["X"]
        assert key is not None and key.valid_day_count == 900

    def test_no_group_when_no_candidate_has_four_folds(self, tmp_path: Path) -> None:
        group = (_candidate(tmp_path, "A", _report(folds="partial")),)
        key, comparable, excluded = module.split_comparable(group)
        assert key is None
        assert comparable == []
        assert len(excluded) == 1


class TestLeadingCandidate:
    def test_leading_is_the_largest_delta_among_passed(self, tmp_path: Path) -> None:
        group = [
            _candidate(tmp_path, "A", _report(delta=0.006)),
            _candidate(tmp_path, "B", _report(delta=0.009)),
            _candidate(tmp_path, "C", _report(delta=0.007)),
        ]
        leader = module.leading_candidate(group)
        assert leader is not None and leader.label == "B"

    def test_a_failing_run_cannot_lead_however_large_its_delta(self, tmp_path: Path) -> None:
        group = [
            _candidate(tmp_path, "A", _report(delta=0.006)),
            _candidate(tmp_path, "BIG", _report(delta=0.050, all_passed=False)),
        ]
        leader = module.leading_candidate(group)
        assert leader is not None and leader.label == "A"

    def test_smoke_mode_never_leads(self, tmp_path: Path) -> None:
        group = [
            _candidate(tmp_path, "A", _report(delta=0.006)),
            _candidate(tmp_path, "SMOKE", _report(delta=0.050, mode="smoke")),
        ]
        leader = module.leading_candidate(group)
        assert leader is not None and leader.label == "A"

    def test_none_when_nothing_passed(self, tmp_path: Path) -> None:
        group = [_candidate(tmp_path, "A", _report(delta=0.001))]
        assert module.leading_candidate(group) is None

    def test_exact_tie_is_broken_by_input_order(self, tmp_path: Path) -> None:
        group = [
            _candidate(tmp_path, "FIRST", _report(delta=0.009)),
            _candidate(tmp_path, "SECOND", _report(family="M1", delta=0.009)),
        ]
        leader = module.leading_candidate(group)
        assert leader is not None and leader.label == "FIRST"


class TestIdentityGuard:
    def test_shared_baseline_identity_holds_for_consistent_reports(self, tmp_path: Path) -> None:
        left = _candidate(tmp_path, "A", _report(delta=0.009))
        right = _candidate(tmp_path, "B", _report(delta=0.006))
        assert module._identity_holds(left, right)

    def test_inconsistent_report_is_refused_a_number(self, tmp_path: Path) -> None:
        report = _report(delta=0.006)
        # Break the arithmetic without touching the gate: delta no longer equals mean - baseline.
        report["oos"]["delta_rank_ic"] = 0.02
        left = _candidate(tmp_path, "A", _report(delta=0.009))
        right = _candidate(tmp_path, "B", report)
        assert not module._identity_holds(left, right)
        markdown = module.render_markdown((left, right))
        # Assert on the matrix section only: the explanatory paragraph below it names the
        # sentinel string too, so a whole-document search passes whether or not the guard fired.
        matrix = markdown.split("## Pairwise increments")[1].split("## Leading candidate")[0]
        assert "IDENTITY_VIOLATION" in matrix
        row = next(line for line in matrix.splitlines() if line.startswith("| A |"))
        assert row == "| A | 0 | IDENTITY_VIOLATION |"
        assert "-0.011000" not in matrix  # the bogus gap 0.009 - 0.02 must not be printed


class TestRender:
    def test_render_is_deterministic(self, tmp_path: Path) -> None:
        group = (
            _candidate(tmp_path, "A", _report(delta=0.009)),
            _candidate(tmp_path, "B", _report(family="M1", delta=0.007, configs=27)),
        )
        assert module.render_markdown(group) == module.render_markdown(group)

    def test_pairwise_cell_is_the_difference_of_the_two_deltas(self, tmp_path: Path) -> None:
        group = (
            _candidate(tmp_path, "A", _report(delta=0.009149)),
            _candidate(tmp_path, "B", _report(family="M1", delta=0.007827)),
        )
        markdown = module.render_markdown(group)
        assert "| A | 0 | +0.001322 |" in markdown
        assert "| B | -0.001322 | 0 |" in markdown

    def test_excluded_candidate_gets_no_pairwise_row(self, tmp_path: Path) -> None:
        group = (
            _candidate(tmp_path, "A", _report(delta=0.009)),
            _candidate(tmp_path, "B", _report(delta=0.008)),
            _candidate(tmp_path, "ODD", _report(delta=0.030, days=969)),
        )
        markdown = module.render_markdown(group)
        assert "**ODD** excluded from all delta arithmetic" in markdown
        matrix = markdown.split("## Pairwise increments")[1].split("## Leading candidate")[0]
        assert "ODD" not in matrix

    def test_no_passed_candidate_records_the_required_sentence(self, tmp_path: Path) -> None:
        group = (
            _candidate(tmp_path, "A", _report(delta=0.001)),
            _candidate(tmp_path, "B", _report(delta=0.004)),
        )
        markdown = module.render_markdown(group)
        assert module.NO_CANDIDATE_PASSED_RECORD in markdown
        assert "simple baseline is retained" in markdown

    def test_worst_seed_names_the_minimum_across_folds(self, tmp_path: Path) -> None:
        report = _report(delta=0.009, worst_seed=20260802)
        # Make F3's worst seed strictly the lowest of the four folds.
        report["folds"][2]["search_results"][-1]["seed_validation_rank_ic"]["20260802"] = 0.01
        candidate = _candidate(tmp_path, "A", report)
        markdown = module.render_markdown((candidate,))
        assert "**20260802 on F3**" in markdown
        assert "0.010000 validation RankIC" in markdown
        assert "not derivable" in markdown  # the OOS worst seed is not in a signed report

    def test_grid_size_drives_the_selection_free_flag(self, tmp_path: Path) -> None:
        single = _candidate(tmp_path, "PINNED", _report(delta=0.009, configs=1))
        searched = _candidate(tmp_path, "GRID", _report(delta=0.008, configs=27))
        assert single.selection_free and not searched.selection_free
        markdown = module.render_markdown((single, searched))
        standings = markdown.split("## Standings")[1].split("## Pairwise")[0]
        pinned_row = next(line for line in standings.splitlines() if line.startswith("| PINNED"))
        grid_row = next(line for line in standings.splitlines() if line.startswith("| GRID"))
        assert "| 1 | yes |" in pinned_row
        assert "| 27 | no |" in grid_row

    def test_missing_gate_fails_closed_at_load(self, tmp_path: Path) -> None:
        report = _report(delta=0.009)
        del report["oos"]["gates"]["G4"]
        candidate = _candidate(tmp_path, "A", report)
        gate = next(item for item in candidate.gates if item.gate == "G4")
        assert gate.status == "FAIL" and gate.observed_value is None
        assert candidate.verdict == "WALK_FORWARD_FAILED"
        markdown = module.render_markdown((candidate,))
        row = next(line for line in markdown.splitlines() if line.startswith("| A | 0."))
        assert "- FAIL" in row  # no observed value, and no silent pass
        assert module.NO_CANDIDATE_PASSED_RECORD in markdown

    def test_a_candidate_with_no_gates_at_all_renders_missing(self, tmp_path: Path) -> None:
        # The load path always emits six verdicts, so this is the renderer's own guard: a
        # Candidate built elsewhere with an empty gate tuple must not render as a pass.
        candidate = _candidate(tmp_path, "A", _report(delta=0.009))
        stripped = dataclasses.replace(candidate, gates=())
        markdown = module.render_markdown((stripped,))
        assert "| - MISSING |" in markdown
        assert stripped.verdict == "WALK_FORWARD_FAILED"

    def test_baseline_section_refuses_per_year_subtraction(self, tmp_path: Path) -> None:
        candidate = _candidate(tmp_path, "A", _report(delta=0.009))
        markdown = module.render_markdown((candidate,), baseline_report=_baseline_report())
        assert "## Baseline run (the term every G4 subtracts)" in markdown
        assert "**Do not subtract these yearly values" in markdown
        assert "| Valid day count | 969 |" in markdown
        assert "F1=B2" in markdown

    def test_every_candidate_appears_in_the_gate_matrix(self, tmp_path: Path) -> None:
        group = (
            _candidate(tmp_path, "A", _report(delta=0.009)),
            _candidate(tmp_path, "B", _report(family="M1", delta=0.001)),
        )
        markdown = module.render_markdown(group)
        matrix = markdown.split("## Gate matrix")[1].split("## Standings")[0]
        assert "| A |" in matrix and "| B |" in matrix
        assert matrix.count("WALK_FORWARD_FAILED") == 1


class TestCli:
    def test_writes_the_document_and_names_the_leader(self, tmp_path: Path, capsys: Any) -> None:
        first = tmp_path / "a.json"
        second = tmp_path / "b.json"
        first.write_text(json.dumps(_report(delta=0.009)), encoding="utf-8")
        second.write_text(json.dumps(_report(family="M1", delta=0.007)), encoding="utf-8")
        out = tmp_path / "comparison.md"
        sys.argv = [
            "summarize_candidate_comparison.py",
            "--report",
            f"A={first}",
            "--report",
            f"B={second}",
            "--out",
            str(out),
        ]
        module.main()
        captured = capsys.readouterr()
        assert "leading candidate: A" in captured.out
        text = out.read_text(encoding="utf-8")
        assert text.startswith("# Real-Data Candidate Comparison (plan Task 5.1)")
        assert module.render_markdown(
            (module.load_candidate("A", first), module.load_candidate("B", second))
        ) == text

    def test_duplicate_labels_are_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "a.json"
        path.write_text(json.dumps(_report()), encoding="utf-8")
        sys.argv = [
            "summarize_candidate_comparison.py",
            "--report",
            f"A={path}",
            "--report",
            f"A={path}",
            "--out",
            str(tmp_path / "out.md"),
        ]
        with pytest.raises(SystemExit, match="labels must be unique"):
            module.main()

    def test_a_spec_without_a_label_is_refused(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="LABEL=PATH"):
            module._parse_spec("/data/some/report.json")
