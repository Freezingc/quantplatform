"""Canary checks for the label-free SHADOW monitoring boundary."""

from __future__ import annotations

import ast
from datetime import date
from pathlib import Path

import pytest

from research_v2.reporting.shadow import build_shadow_daily_report

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
