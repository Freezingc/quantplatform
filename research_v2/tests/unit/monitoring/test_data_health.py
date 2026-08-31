from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from research_v2.monitoring.data_health import (
    FeatureDistributionReference,
    FeatureReference,
    MonitoringReferenceArtifact,
    ScoreDistributionReference,
    evaluate_feature_data_health,
)
from research_v2.monitoring.drift import DriftSeverity
from research_v2.monitoring.policy import MonitoringPolicy


def _reference() -> FeatureReference:
    return FeatureReference(
        feature_name="quality",
        bin_edges=(0.0, 0.5, 1.0),
        reference_bin_masses=(0.5, 0.5),
        reference_missing_rate=0.0,
        minimum_finite_coverage=0.95,
    )


def _frame(signal_date: date) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": [signal_date] * 20,
            "ts_code": [f"S{index:03d}" for index in range(20)],
            "quality": [0.1] * 10 + [0.9] * 10,
            "available_date": [signal_date] * 20,
        }
    )


def test_feature_data_health_blocks_missing_keys_future_data_and_low_row_coverage() -> None:
    signal_date = date(2026, 8, 27)
    healthy = _frame(signal_date)

    report = evaluate_feature_data_health(
        healthy,
        signal_date=signal_date,
        references=(_reference(),),
        expected_universe_size=20,
        recent_universe_size_median=20,
    )

    assert not report.blocking_reasons
    assert report.features[0].psi.severity is DriftSeverity.INFO

    malformed = healthy.drop(columns=["quality"]).iloc[:18].copy()
    malformed.loc[0, "available_date"] = date(2026, 8, 28)
    malformed.loc[1, "ts_code"] = malformed.loc[0, "ts_code"]
    blocked = evaluate_feature_data_health(
        malformed,
        signal_date=signal_date,
        references=(_reference(),),
        expected_universe_size=20,
        recent_universe_size_median=20,
    )

    assert set(blocked.blocking_reasons) == {
        "DUPLICATE_BUSINESS_KEY",
        "FUTURE_AVAILABLE_DATE",
        "INSUFFICIENT_ROW_COVERAGE",
        "MISSING_REQUIRED_FEATURE:quality",
    }


def test_feature_data_health_reports_factor_coverage_and_psi_without_blocking() -> None:
    signal_date = date(2026, 8, 27)
    frame = _frame(signal_date)
    frame.loc[:2, "quality"] = None
    frame.loc[3:, "quality"] = 0.99

    report = evaluate_feature_data_health(
        frame,
        signal_date=signal_date,
        references=(_reference(),),
        expected_universe_size=20,
        recent_universe_size_median=20,
    )

    assert not report.blocking_reasons
    assert report.features[0].finite_coverage == 0.85
    assert report.features[0].psi.severity is DriftSeverity.ERROR
    assert report.investigation_alerts == (
        "LOW_FACTOR_COVERAGE:quality",
        "MISSING_RATE_INCREASE:quality",
        "PSI_ERROR:quality",
    )


def test_feature_data_health_counts_values_outside_reference_bins_as_tail_drift() -> None:
    signal_date = date(2026, 8, 27)
    frame = _frame(signal_date).assign(quality=2.0)

    report = evaluate_feature_data_health(
        frame,
        signal_date=signal_date,
        references=(_reference(),),
        expected_universe_size=20,
        recent_universe_size_median=20,
    )

    assert report.features[0].psi.severity is DriftSeverity.ERROR


def test_feature_data_health_blocks_missing_available_dates_and_blank_business_keys() -> None:
    signal_date = date(2026, 8, 27)
    frame = _frame(signal_date)
    frame.loc[0, "available_date"] = None
    frame.loc[1, "ts_code"] = " "

    report = evaluate_feature_data_health(
        frame,
        signal_date=signal_date,
        references=(_reference(),),
        expected_universe_size=20,
        recent_universe_size_median=20,
    )

    assert {"MISSING_AVAILABLE_DATE", "INVALID_BUSINESS_KEY"}.issubset(
        report.blocking_reasons
    )


def test_feature_data_health_uses_loaded_policy_thresholds() -> None:
    signal_date = date(2026, 8, 27)
    report = evaluate_feature_data_health(
        _frame(signal_date).iloc[:18],
        signal_date=signal_date,
        references=(_reference(),),
        expected_universe_size=20,
        recent_universe_size_median=20,
        policy=MonitoringPolicy(minimum_row_coverage=0.90),
    )

    assert not report.blocking_reasons


def _reference_artifact() -> MonitoringReferenceArtifact:
    return MonitoringReferenceArtifact(
        model_id="model-1",
        feature_references=(
            FeatureDistributionReference(
                feature_name="quality",
                bin_edges=tuple(index / 10 for index in range(11)),
                bin_masses=(0.1,) * 10,
                missing_rate=0.02,
                median=0.5,
                iqr=0.2,
            ),
        ),
        validation_score_reference=ScoreDistributionReference(
            bin_edges=tuple(index / 10 for index in range(11)),
            bin_masses=(0.1,) * 10,
            finite_coverage=0.99,
        ),
    )


def test_monitoring_reference_artifact_has_stable_canonical_digest() -> None:
    artifact = _reference_artifact()
    reordered = MonitoringReferenceArtifact(
        model_id=artifact.model_id,
        feature_references=artifact.feature_references,
        validation_score_reference=artifact.validation_score_reference,
        validation_period=artifact.validation_period,
        training_period=artifact.training_period,
    )

    assert artifact.digest == reordered.digest
    document = json.loads(artifact.to_json())
    assert document["content_hash"] == artifact.digest
    assert artifact.to_json().endswith("\n")


def test_monitoring_reference_artifact_write_is_immutable_and_load_is_model_bound(
    tmp_path: Path,
) -> None:
    artifact = _reference_artifact()
    path = tmp_path / "reference.json"

    assert artifact.write(path) == artifact.digest
    loaded = MonitoringReferenceArtifact.load(path, expected_model_id="model-1")

    assert loaded == artifact
    with pytest.raises(FileExistsError, match="immutable"):
        artifact.write(path)
    with pytest.raises(ValueError, match="model_id"):
        MonitoringReferenceArtifact.load(path, expected_model_id="other-model")


@pytest.mark.parametrize(
    "tamper",
    [
        lambda document: document["payload"].update({"model_id": "other-model"}),
        lambda document: document["payload"]["features"][0].update({"median": 0.7}),
        lambda document: document.update({"content_hash": "0" * 64}),
    ],
)
def test_monitoring_reference_artifact_rejects_tampered_content(
    tmp_path: Path,
    tamper: object,
) -> None:
    path = tmp_path / "reference.json"
    path.write_text(_reference_artifact().to_json(), encoding="utf-8")
    document = json.loads(path.read_text(encoding="utf-8"))
    tamper(document)  # type: ignore[operator]
    path.write_text(
        json.dumps(document, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="integrity|canonical"):
        MonitoringReferenceArtifact.load(path)


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"content_hash": "0" * 64, "payload": {}},
        {"content_hash": "0" * 64, "payload": {"model_id": "model-1"}},
    ],
)
def test_monitoring_reference_artifact_rejects_malformed_documents(
    tmp_path: Path,
    document: dict[str, object],
) -> None:
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="reference artifact"):
        MonitoringReferenceArtifact.load(path)


def test_monitoring_reference_artifact_converts_features_without_losing_stats() -> None:
    artifact = _reference_artifact()

    references = artifact.feature_references_for_health(minimum_finite_coverage=0.90)

    assert references == (
        FeatureReference(
            feature_name="quality",
            bin_edges=tuple(index / 10 for index in range(11)),
            reference_bin_masses=(0.1,) * 10,
            reference_missing_rate=0.02,
            minimum_finite_coverage=0.90,
            reference_median=0.5,
            reference_iqr=0.2,
        ),
    )
