"""Tests for the criterion-303 walk-forward report summarizer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import summarize_real_walk_forward as summarizer


def _validation_scores() -> list[dict[str, object]]:
    scores = []
    for baseline in ("B0", "B1", "B2", "B3"):
        scores.append(
            {
                "baseline_id": baseline,
                "mean_rank_ic": 0.02 + 0.001 * int(baseline[1]),
                "valid_day_count": 240,
            }
        )
    return scores


def _make_report(
    *,
    all_passed: bool = True,
    folds: tuple[str, ...] = ("F01", "F02", "F03", "F04", "F05", "F06"),
    gate_overrides: dict[str, dict[str, object]] | None = None,
    drop_baselines: bool = False,
    drop_year: str | None = None,
) -> dict[str, object]:
    gates: dict[str, dict[str, object]] = {
        "G1": {"status": "PASS", "observed_value": 0.035, "threshold": 0.03},
        "G2": {"status": "PASS", "observed_value": 2.5, "threshold": 2.0},
        "G3": {"status": "PASS", "observed_value": 0.833, "threshold": 0.7},
        "G4": {"status": "NOT_APPLICABLE", "observed_value": None, "threshold": None},
        "G5": {"status": "PASS", "observed_value": 0.3, "threshold": 0.5},
        "G6": {"status": "PASS", "observed_value": 0.99, "threshold": 0.95},
    }
    if gate_overrides:
        gates.update(gate_overrides)
    selections = []
    for fold in folds:
        scores = _validation_scores()
        if drop_baselines:
            scores = [score for score in scores if score["baseline_id"] != "B3"]
        selections.append(
            {"fold_id": fold, "selected_baseline_id": "B2", "validation_scores": scores}
        )
    yearly: dict[str, float] = {str(year): 0.03 for year in range(2020, 2026)}
    if drop_year is not None:
        yearly.pop(drop_year)
    return {
        "folds": list(folds),
        "selections": selections,
        "oos": {
            "valid_day_count": 1400,
            "mean_rank_ic": 0.035,
            "icir": 0.42,
            "hac_t_statistic": 2.5,
            "baseline_mean_rank_ic": 0.035,
            "delta_rank_ic": None,
            "yearly_mean_rank_ic": yearly,
            "gates": gates,
            "all_passed": all_passed,
        },
    }


def test_gate_verdicts_cover_all_gates() -> None:
    verdicts = summarizer.gate_verdicts(_make_report())
    assert [verdict.gate for verdict in verdicts] == ["G1", "G2", "G3", "G4", "G5", "G6"]
    assert all(verdict.satisfied for verdict in verdicts)


def test_failed_gate_is_not_satisfied() -> None:
    report = _make_report(
        all_passed=False,
        gate_overrides={"G2": {"status": "FAIL", "observed_value": 1.2, "threshold": 2.0}},
    )
    by_gate = {verdict.gate: verdict for verdict in summarizer.gate_verdicts(report)}
    assert by_gate["G2"].status == "FAIL"
    assert not by_gate["G2"].satisfied


def test_criterion_303_satisfied_on_complete_report() -> None:
    assert summarizer.criterion_303_verdict(_make_report()).satisfied


def test_criterion_303_fails_when_fold_missing() -> None:
    verdict = summarizer.criterion_303_verdict(_make_report(folds=("F01", "F02", "F03")))
    assert not verdict.folds_complete
    assert not verdict.satisfied


def test_criterion_303_fails_when_baseline_missing() -> None:
    verdict = summarizer.criterion_303_verdict(_make_report(drop_baselines=True))
    assert not verdict.baselines_complete
    assert not verdict.satisfied


def test_criterion_303_fails_when_year_missing() -> None:
    verdict = summarizer.criterion_303_verdict(_make_report(drop_year="2025"))
    assert not verdict.years_complete
    assert not verdict.satisfied


def test_criterion_303_fails_when_gates_fail() -> None:
    verdict = summarizer.criterion_303_verdict(_make_report(all_passed=False))
    assert not verdict.satisfied


def test_render_markdown_contains_key_sections() -> None:
    markdown = summarizer.render_markdown(_make_report())
    assert "# Real-Data B0-B3 Walk-Forward Report (Criterion 303)" in markdown
    assert "| G1 |" in markdown
    assert "**Criterion 303: SATISFIED**" in markdown
    for fold in ("F01", "F06"):
        assert f"| {fold} |" in markdown


def test_render_markdown_marks_winner() -> None:
    markdown = summarizer.render_markdown(_make_report())
    assert "**0.0220**" in markdown  # B2 validation RankIC bolded as winner


def test_main_writes_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_make_report()), encoding="utf-8")
    out_path = tmp_path / "out" / "summary.md"
    argv = sys.argv
    sys.argv = [
        "summarize_real_walk_forward.py",
        "--report",
        str(report_path),
        "--out",
        str(out_path),
    ]
    try:
        summarizer.main()
    finally:
        sys.argv = argv
    assert out_path.exists()
    captured = capsys.readouterr()
    assert "criterion-303: SATISFIED" in captured.out


def test_data_version_summary_absent_returns_none() -> None:
    assert summarizer.data_version_summary(_make_report()) is None


def test_data_version_summary_compact() -> None:
    report = _make_report()
    report["data_versions"] = {
        "universe_daily": {"2020.01M": "v1", "2020.02M": "v2"},
        "label_daily": {},
    }
    summary = summarizer.data_version_summary(report)
    assert summary == {
        "label_daily": "0 partitions",
        "universe_daily": "2 partitions (2020.01M..2020.02M)",
    }


def test_render_markdown_includes_versions_when_present() -> None:
    report = _make_report()
    report["data_versions"] = {"universe_daily": {"2020.01M": "v1"}}
    markdown = summarizer.render_markdown(report)
    assert "## Data versions (published_version snapshot)" in markdown
    assert "| universe_daily | 1 partitions (2020.01M..2020.01M) |" in markdown


def test_render_markdown_without_versions_still_renders_verdict() -> None:
    markdown = summarizer.render_markdown(_make_report())
    assert "Data versions" not in markdown
    assert "## Criterion 303 verdict" in markdown


# --- Task 3.3: the extended selection fields have to reach the rendered verdict -------------

_REPORTS = Path(__file__).resolve().parents[4] / "docs" / "reports"
_LEGACY_JSON = _REPORTS / "2026-09-04-real-walk-forward-criterion-303.json"
_LEGACY_MD = _REPORTS / "2026-09-04-real-walk-forward-criterion-303.md"


def _extended_report(**overrides: object) -> dict[str, object]:
    """The shape ``run_real_walk_forward.py`` emits since the extended fields landed.

    Taken field for field from the Task 2.4 re-run report: each validation score carries
    ``invalid_day_count`` beside ``valid_day_count``, each selection carries
    ``unselected_reasons`` as ``[baseline_id, reason]`` pairs, and the report carries a
    top-level ``baseline_oos_artifacts`` mapping fold id to a parquet path.
    """
    report = _make_report()
    for selection in report["selections"]:  # type: ignore[union-attr]
        for score in selection["validation_scores"]:
            score["invalid_day_count"] = 0
        selection["unselected_reasons"] = [
            [baseline, "LOWER_VALIDATION_RANKIC"] for baseline in ("B0", "B1", "B3")
        ]
    report["baseline_oos_artifacts"] = {
        fold: f"/tmp/run/baseline_oos/{fold}.parquet"
        for fold in ("F01", "F02", "F03", "F04", "F05", "F06")
    }
    report.update(overrides)
    return report


@pytest.mark.skipif(not _LEGACY_MD.exists(), reason="archived criterion-303 evidence absent")
def test_the_archived_criterion_303_markdown_is_still_reproduced_byte_for_byte() -> None:
    """The archived evidence is a real file, so re-rendering it is a real regression test.

    This is the strongest available pin on the renderer: not a synthetic expectation but the
    committed markdown that criterion-303 was accepted against. Anything the extended-field
    work does to the legacy rendering path shows up here as a byte difference, which is why
    the new sections are appended conditionally rather than woven into the existing tables.
    """
    report = summarizer.load_report(_LEGACY_JSON)

    assert summarizer.render_markdown(report) == _LEGACY_MD.read_text(encoding="utf-8")


def test_invalid_day_counts_are_rendered_beside_the_valid_ones() -> None:
    """``invalid_day_count`` is the field that says how many days a baseline lost.

    It is the one number in the report that connects a validation score to the coverage
    problem the frozen-set plan is blocked on, and before this it existed only in the JSON:
    a reader of the verdict saw 0.0724 and had no way to know whether it came from 238 days
    or from 118.
    """
    markdown = summarizer.render_markdown(_extended_report())

    assert "## Selection detail" in markdown
    assert "Invalid days" in markdown
    assert "| F01 | B0 | 240 | 0 |" in markdown


def test_a_baseline_that_lost_days_shows_it() -> None:
    """The reason the field matters: a score computed on fewer days must not look like one
    computed on all of them."""
    report = _extended_report()
    for selection in report["selections"]:  # type: ignore[union-attr]
        if selection["fold_id"] != "F03":
            continue
        for score in selection["validation_scores"]:
            if score["baseline_id"] == "B1":
                score["valid_day_count"] = 226
                score["invalid_day_count"] = 14

    markdown = summarizer.render_markdown(report)

    assert "| F03 | B1 | 226 | 14 |" in markdown


def test_why_each_baseline_lost_is_rendered_per_fold() -> None:
    report = _extended_report()

    markdown = summarizer.render_markdown(report)

    assert "LOWER_VALIDATION_RANKIC" in markdown
    assert "| F01 | B0 | 240 | 0 | not selected: LOWER_VALIDATION_RANKIC |" in markdown
    assert "| F01 | B2 | 240 | 0 | **selected** |" in markdown


def test_fold_matched_baseline_artifacts_are_listed_with_their_paths() -> None:
    """3.3's other half: the fold-matched baseline predictions have to be findable.

    A path recorded only in JSON is a path nobody opens. Task 5.1 consumes these artifacts
    as ``--baseline-artifacts``, so the verdict that issues them should say where they are.
    """
    markdown = summarizer.render_markdown(_extended_report())

    assert "## Fold-matched baseline OOS artifacts" in markdown
    assert "| F01 | /tmp/run/baseline_oos/F01.parquet |" in markdown
    assert markdown.count("baseline_oos/F0") == 6


def test_a_report_without_the_extended_fields_grows_no_empty_sections() -> None:
    """Backward compatibility, asserted rather than assumed.

    The archived criterion-303 report predates all three fields. Rendering it must not grow
    a heading with nothing under it, which is how a report starts lying about what was
    measured -- an empty "Invalid days" column reads as "zero invalid days", not as "not
    recorded".
    """
    markdown = summarizer.render_markdown(_make_report())

    assert "## Selection detail" not in markdown
    assert "Invalid days" not in markdown
    assert "## Fold-matched baseline OOS artifacts" not in markdown
    assert "## Criterion 303 verdict" in markdown


def test_the_extended_sections_come_before_the_verdict_so_the_verdict_stays_last() -> None:
    markdown = summarizer.render_markdown(_extended_report())

    assert markdown.index("## Selection detail") < markdown.index("## Criterion 303 verdict")
    assert markdown.index("## Fold-matched baseline OOS artifacts") < markdown.index(
        "## Criterion 303 verdict"
    )


def test_a_partially_extended_report_renders_what_it_has_and_omits_what_it_lacks() -> None:
    """The three fields landed together in practice, but the renderer must not assume it.

    A report carrying ``invalid_day_count`` but no ``baseline_oos_artifacts`` is what an
    interrupted run produces, and it should render the detail table and simply not claim
    there are artifacts.
    """
    report = _extended_report()
    del report["baseline_oos_artifacts"]

    markdown = summarizer.render_markdown(report)

    assert "## Selection detail" in markdown
    assert "## Fold-matched baseline OOS artifacts" not in markdown
