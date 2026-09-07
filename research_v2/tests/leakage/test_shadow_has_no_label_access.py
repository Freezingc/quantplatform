"""Canary checks for the label-free SHADOW monitoring boundary."""

from __future__ import annotations

import ast
from datetime import date
from pathlib import Path

import pytest

from research_v2.reporting.shadow import ShadowDailyReport, build_shadow_daily_report

_MODEL_ID = "sha256:" + "a" * 64


def test_shadow_modules_do_not_import_label_or_performance_evaluation_code() -> None:
    modules = (
        Path(__file__).parents[2] / "src/research_v2/monitoring/shadow.py",
        Path(__file__).parents[2] / "src/research_v2/reporting/shadow.py",
    )
    forbidden = ("labels", "rank_ic", "pnl", "performance")

    for module in modules:
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        imported = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ] + [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        assert not any(
            token in name.lower() for name in imported for token in forbidden
        ), module


def test_shadow_report_rejects_any_label_or_return_derived_value() -> None:
    with pytest.raises(ValueError, match="payload"):
        build_shadow_daily_report(
            signal_date=date(2026, 8, 27),
            model_id=_MODEL_ID,
            model_version=1,
            pipeline_status="SUCCESS",
            prediction_published=True,
            inference_manifest_id="sha256:" + "b" * 64,
            artifact_bundle_hash="c" * 64,
            source_watermark_id="d" * 64,
            latency_seconds=1.0,
            universe_size=1,
            feature_coverage={"quality": 1.0},
            score_health={"finite_coverage": 1.0},
            seed_agreement={"minimum_spearman": 1.0},
            drift_alert_keys=(),
            event_ids=("future_return_observed",),
        )


def test_shadow_report_rejects_unapproved_numeric_health_metrics() -> None:
    with pytest.raises(ValueError, match="score_health"):
        build_shadow_daily_report(
            signal_date=date(2026, 8, 27),
            model_id=_MODEL_ID,
            model_version=1,
            pipeline_status="SUCCESS",
            prediction_published=True,
            inference_manifest_id="sha256:" + "b" * 64,
            artifact_bundle_hash="c" * 64,
            source_watermark_id="d" * 64,
            latency_seconds=1.0,
            universe_size=1,
            feature_coverage={"quality": 1.0},
            score_health={"future_close": 12.4},
            seed_agreement={"minimum_spearman": 1.0},
            drift_alert_keys=(),
            event_ids=("e" * 64,),
        )


def test_shadow_report_rejects_out_of_domain_health_metric_values() -> None:
    with pytest.raises(ValueError, match="score_health"):
        build_shadow_daily_report(
            signal_date=date(2026, 8, 27),
            model_id=_MODEL_ID,
            model_version=1,
            pipeline_status="SUCCESS",
            prediction_published=True,
            inference_manifest_id="sha256:" + "b" * 64,
            artifact_bundle_hash="c" * 64,
            source_watermark_id="d" * 64,
            latency_seconds=1.0,
            universe_size=1,
            feature_coverage={"quality": 1.0},
            score_health={
                "finite_coverage": 2.0,
                "unique_score_count": 1.5,
                "score_std": 0.1,
            },
            seed_agreement={"minimum_spearman": 1.0},
            drift_alert_keys=(),
            event_ids=("e" * 64,),
        )


# --- what the report contract refuses ----------------------------------------
#
# Twelve of sixteen mutations of reporting/shadow.py survived this file. The four that
# died are the token walk being applied at all, the token list, sequence inspection, and
# the publication equivalence that 续六十六's SH-H/SH-I equivalence rests on. What was not
# pinned: how the token walk matches, and every numeric domain check.


def _daily_report(**overrides: object) -> ShadowDailyReport:
    """A valid label-free report, with any builder argument overridden."""
    arguments: dict[str, object] = {
        "signal_date": date(2026, 8, 27),
        "model_id": _MODEL_ID,
        "model_version": 1,
        "pipeline_status": "SUCCESS",
        "prediction_published": True,
        "inference_manifest_id": "sha256:" + "b" * 64,
        "artifact_bundle_hash": "c" * 64,
        "source_watermark_id": "d" * 64,
        "latency_seconds": 1.0,
        "universe_size": 1,
        "feature_coverage": {"quality": 1.0},
        "score_health": {"finite_coverage": 1.0, "unique_score_count": 800, "score_std": 0.15},
        "seed_agreement": {"minimum_spearman": 0.42},
        "drift_alert_keys": (),
        "event_ids": ("e" * 64,),
    }
    arguments.update(overrides)
    return build_shadow_daily_report(**arguments)  # type: ignore[arg-type]


def test_shadow_report_rejects_forbidden_tokens_in_any_case_and_in_mapping_keys() -> None:
    """The walk lowercases before matching, and inspects mapping KEYS and not only values.

    Both halves were unpinned. The existing leakage test carries its forbidden token in a
    lower-case sequence value, which a case-sensitive match still catches and which
    dropping the key inspection does not affect -- so neither mutation was observable
    there. ``feature_coverage`` names are free-form, which makes a key a realistic place
    for a label- or return-derived quantity to be recorded, and mixed case is how such a
    name would actually be written.
    """
    assert _daily_report().verify() is None  # control: the untampered report is valid

    for coverage_name in ("Label_Coverage", "label_coverage", "RETURN_5D", "PnL_Share"):
        with pytest.raises(ValueError, match="payload"):
            _daily_report(feature_coverage={coverage_name: 1.0})


def test_a_tampered_report_is_refused_for_what_changed_and_not_only_for_its_hash() -> None:
    """``verify()`` validates the payload before comparing the id, so the reason survives.

    Every tampering below would also be caught by the content-hash comparison, which is
    exactly why the message matters: without these clauses a report that claims a computed
    performance status, or carries an unknown extra field, is refused as "the hash does not
    match" -- true, but it hides that the shadow phase just produced a report saying it
    computed performance. ``performance_status == LOCKED_NOT_COMPUTED`` is the field-level
    half of this file's label-free boundary; the import canary above is the other half.
    """
    report = _daily_report()
    assert report.verify() is None

    cases = (
        (
            ShadowDailyReport(
                report.report_id, {**report.payload, "performance_status": "IC_COMPUTED"}
            ),
            "performance status",
        ),
        (
            ShadowDailyReport(report.report_id, {**report.payload, "surprise": 1}),
            "payload",
        ),
        (
            ShadowDailyReport(report.report_id, {**report.payload, "latency_seconds": 999.0}),
            "does not match report_id",
        ),
        (ShadowDailyReport("0" * 63, report.payload), "report_id is invalid"),
    )
    for tampered, match in cases:
        with pytest.raises(ValueError, match=match):
            tampered.verify()


def test_shadow_report_refuses_out_of_domain_metrics_and_an_unlisted_pipeline_status() -> None:
    """Every numeric domain and the closed status vocabulary, one case each.

    The two existing metric tests both target ``score_health``'s key set and its coverage
    value, which is why five other bounds were unpinned: a zero unique-score count, a
    Spearman outside [-1, 1], a coverage of 5.0, a zero latency SLO and a negative
    latency. Each is a value that cannot occur in a real run, and each is a value that a
    broken writer would produce silently -- the report is what the shadow phase publishes
    instead of performance, so an impossible number in it is the only signal there is.
    ``pipeline_status`` is a closed vocabulary of three, and a fourth value must not be
    issuable even though ``_is_valid_shadow_day`` would not count such a day anyway.
    """
    cases = (
        (
            {"score_health": {"finite_coverage": 1.0, "unique_score_count": 0, "score_std": 0.1}},
            "score_health",
        ),
        ({"seed_agreement": {"minimum_spearman": 4.0}}, "seed_agreement"),
        ({"feature_coverage": {"quality": 5.0}}, "feature_coverage"),
        ({"latency_slo_seconds": 0.0}, "latency_slo_seconds"),
        ({"latency_seconds": -1.0}, "latency_seconds"),
        ({"pipeline_status": "PARTIAL", "prediction_published": False}, "pipeline_status"),
    )
    for overrides, match in cases:
        with pytest.raises(ValueError, match=match):
            _daily_report(**overrides)


def test_a_report_cannot_claim_success_without_publishing_or_publish_without_success() -> None:
    """The publication equivalence, pinned in the module that owns it.

    This is the line 续六十六's SH-H/SH-I equivalence rests on: with it, "the pipeline
    succeeded" and "the prediction was published" are one predicate, so the shadow day
    filter's two conditions are one defence written twice; without it they become two real
    holes. It was already killed at full scope before this test existed, but only by the
    monitoring suites -- the sixth case in the attribution matrix, a property of this module
    pinned by another module's tests, and the cheapest of the six to move home.
    """
    for status, published in (
        ("SUCCESS", False),
        ("FAILED", True),
        ("NOOP_NON_TRADING_DAY", True),
    ):
        with pytest.raises(ValueError, match="conflicts with publication state"):
            _daily_report(pipeline_status=status, prediction_published=published)
