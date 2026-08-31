"""Feature-frame health checks kept separate from inference publication policy."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from research_v2.monitoring.drift import DriftSeverity, PsiResult, population_stability_index
from research_v2.monitoring.policy import DEFAULT_MONITORING_POLICY, MonitoringPolicy


@dataclass(frozen=True)
class FeatureReference:
    feature_name: str
    bin_edges: tuple[float, ...]
    reference_bin_masses: tuple[float, ...]
    reference_missing_rate: float
    minimum_finite_coverage: float
    reference_median: float | None = None
    reference_iqr: float | None = None

    def __post_init__(self) -> None:
        if not self.feature_name.strip():
            raise ValueError("feature_name cannot be empty")
        if len(self.bin_edges) < 2 or len(self.reference_bin_masses) != len(self.bin_edges) - 1:
            raise ValueError("feature reference bins are invalid")
        if any(not math.isfinite(value) for value in self.bin_edges) or any(
        right <= left for left, right in zip(self.bin_edges, self.bin_edges[1:], strict=False)
        ):
            raise ValueError("feature reference bin edges must be finite and increasing")
        if any(not math.isfinite(value) or value < 0 for value in self.reference_bin_masses):
            raise ValueError("feature reference bin masses are invalid")
        if not 0 <= self.reference_missing_rate <= 1:
            raise ValueError("reference_missing_rate must be between zero and one")
        if not 0 <= self.minimum_finite_coverage <= 1:
            raise ValueError("minimum_finite_coverage must be between zero and one")
        if self.reference_median is not None and not math.isfinite(self.reference_median):
            raise ValueError("reference_median must be finite when provided")
        if self.reference_iqr is not None and (
            not math.isfinite(self.reference_iqr) or self.reference_iqr < 0
        ):
            raise ValueError("reference_iqr must be finite and non-negative when provided")


@dataclass(frozen=True)
class FeatureDistributionReference:
    """One frozen, winsorized/transformed feature distribution from model training."""

    feature_name: str
    bin_edges: tuple[float, ...]
    bin_masses: tuple[float, ...]
    missing_rate: float
    median: float
    iqr: float
    transform_id: str = "winsorized_transformed"

    def __post_init__(self) -> None:
        if not isinstance(self.feature_name, str) or not self.feature_name.strip():
            raise ValueError("feature distribution feature_name cannot be empty")
        try:
            edges = tuple(
                _finite_number(value, "feature distribution bin edge") for value in self.bin_edges
            )
            masses = tuple(
                _finite_number(value, "feature distribution bin mass") for value in self.bin_masses
            )
        except TypeError as error:
            raise ValueError("feature distribution bins are invalid") from error
        _validate_decile_bins(edges, masses, "feature distribution")
        object.__setattr__(self, "bin_edges", edges)
        object.__setattr__(self, "bin_masses", masses)
        if self.transform_id != "winsorized_transformed":
            raise ValueError("feature distribution transform_id must be winsorized_transformed")
        self_missing_rate = _finite_number(self.missing_rate, "feature distribution missing_rate")
        if not 0 <= self_missing_rate <= 1:
            raise ValueError("feature distribution missing_rate must be between zero and one")
        object.__setattr__(self, "missing_rate", self_missing_rate)
        object.__setattr__(
            self,
            "median",
            _finite_number(self.median, "feature distribution median"),
        )
        self_iqr = _finite_number(self.iqr, "feature distribution iqr")
        if self_iqr < 0:
            raise ValueError("feature distribution iqr must be finite and non-negative")
        object.__setattr__(self, "iqr", self_iqr)


@dataclass(frozen=True)
class ScoreDistributionReference:
    """Frozen validation ensemble-score deciles and finite coverage."""

    bin_edges: tuple[float, ...]
    bin_masses: tuple[float, ...]
    finite_coverage: float

    def __post_init__(self) -> None:
        try:
            edges = tuple(
                _finite_number(value, "score distribution bin edge") for value in self.bin_edges
            )
            masses = tuple(
                _finite_number(value, "score distribution bin mass") for value in self.bin_masses
            )
        except TypeError as error:
            raise ValueError("score distribution bins are invalid") from error
        _validate_decile_bins(edges, masses, "score distribution")
        object.__setattr__(self, "bin_edges", edges)
        object.__setattr__(self, "bin_masses", masses)
        coverage = _finite_number(self.finite_coverage, "score distribution finite_coverage")
        if not 0 <= coverage <= 1:
            raise ValueError("score distribution finite_coverage must be between zero and one")
        object.__setattr__(self, "finite_coverage", coverage)


@dataclass(frozen=True)
class MonitoringReferenceArtifact:
    """Immutable model-bound reference distributions for daily drift monitoring."""

    model_id: str
    feature_references: tuple[FeatureDistributionReference, ...]
    validation_score_reference: ScoreDistributionReference
    training_period: str = "FINAL_TRAIN_2020_2024"
    validation_period: str = "VALIDATION_2025"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("reference artifact model_id must be a non-empty string")
        object.__setattr__(self, "feature_references", tuple(self.feature_references))
        if not self.feature_references:
            raise ValueError("reference artifact requires at least one feature reference")
        if any(
            not isinstance(value, FeatureDistributionReference)
            for value in self.feature_references
        ):
            raise ValueError("reference artifact feature references are invalid")
        names = [value.feature_name for value in self.feature_references]
        if len(set(names)) != len(names):
            raise ValueError("reference artifact feature names must be unique")
        if not isinstance(self.validation_score_reference, ScoreDistributionReference):
            raise ValueError("reference artifact score reference is invalid")
        if not isinstance(self.training_period, str) or not self.training_period.strip():
            raise ValueError("reference artifact training_period must be non-empty")
        if not isinstance(self.validation_period, str) or not self.validation_period.strip():
            raise ValueError("reference artifact validation_period must be non-empty")
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("reference artifact schema_version is unsupported")

    @property
    def digest(self) -> str:
        """Return the SHA-256 of the canonical artifact payload, excluding its envelope hash."""
        return hashlib.sha256(_canonical_json(self._payload()).encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        document = {"content_hash": self.digest, "payload": self._payload()}
        return _canonical_json(document) + "\n"

    def write(self, path: Path) -> str:
        """Create one canonical artifact exactly once and return its content digest."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            _write_exclusive(destination, self.to_json())
        except FileExistsError as error:
            raise FileExistsError(
                f"monitoring reference artifact is immutable: {destination}"
            ) from error
        return self.digest

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_model_id: str | None = None,
    ) -> MonitoringReferenceArtifact:
        """Load only a canonical, hash-valid artifact bound to the requested model."""
        source = Path(path)
        try:
            raw = source.read_text(encoding="utf-8")
            document = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"reference artifact cannot be loaded: {source}") from error
        if raw != _canonical_json(document) + "\n":
            raise ValueError("reference artifact is not canonical JSON")
        if not isinstance(document, Mapping) or set(document) != {"content_hash", "payload"}:
            raise ValueError("reference artifact document schema is invalid")
        content_hash = document.get("content_hash")
        payload = document.get("payload")
        if not isinstance(content_hash, str) or not _is_sha256(content_hash):
            raise ValueError("reference artifact content_hash is invalid")
        if not isinstance(payload, Mapping):
            raise ValueError("reference artifact payload is invalid")
        expected_hash = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        if content_hash != expected_hash:
            raise ValueError("reference artifact integrity verification failed")
        try:
            artifact = cls._from_payload(payload)
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError("reference artifact payload is invalid") from error
        if expected_model_id is not None and artifact.model_id != expected_model_id:
            raise ValueError("reference artifact model_id does not match expected model_id")
        return artifact

    def feature_references_for_health(
        self,
        *,
        minimum_finite_coverage: float = 0.95,
    ) -> tuple[FeatureReference, ...]:
        """Adapt frozen feature statistics to the existing daily health evaluator."""
        return tuple(
            FeatureReference(
                feature_name=reference.feature_name,
                bin_edges=reference.bin_edges,
                reference_bin_masses=reference.bin_masses,
                reference_missing_rate=reference.missing_rate,
                minimum_finite_coverage=minimum_finite_coverage,
                reference_median=reference.median,
                reference_iqr=reference.iqr,
            )
            for reference in self.feature_references
        )

    def _payload(self) -> dict[str, object]:
        return {
            "features": [
                {
                    "bin_edges": list(reference.bin_edges),
                    "bin_masses": list(reference.bin_masses),
                    "feature_name": reference.feature_name,
                    "iqr": reference.iqr,
                    "median": reference.median,
                    "missing_rate": reference.missing_rate,
                    "transform_id": reference.transform_id,
                }
                for reference in self.feature_references
            ],
            "model_id": self.model_id,
            "schema_version": self.schema_version,
            "training_period": self.training_period,
            "validation_period": self.validation_period,
            "validation_score": {
                "bin_edges": list(self.validation_score_reference.bin_edges),
                "bin_masses": list(self.validation_score_reference.bin_masses),
                "finite_coverage": self.validation_score_reference.finite_coverage,
            },
        }

    @classmethod
    def _from_payload(cls, payload: Mapping[str, object]) -> MonitoringReferenceArtifact:
        required = {
            "features",
            "model_id",
            "schema_version",
            "training_period",
            "validation_period",
            "validation_score",
        }
        if set(payload) != required:
            raise ValueError("reference artifact payload schema is invalid")
        features = payload["features"]
        if not isinstance(features, Sequence) or isinstance(features, (str, bytes, bytearray)):
            raise ValueError("reference artifact features are invalid")
        feature_references = tuple(_feature_distribution_from_mapping(value) for value in features)
        score = payload["validation_score"]
        if not isinstance(score, Mapping):
            raise ValueError("reference artifact validation score is invalid")
        return cls(
            model_id=_required_string(payload["model_id"], "model_id"),
            feature_references=feature_references,
            validation_score_reference=ScoreDistributionReference(
                bin_edges=_required_float_tuple(score.get("bin_edges"), "score bin_edges"),
                bin_masses=_required_float_tuple(score.get("bin_masses"), "score bin_masses"),
                finite_coverage=_required_float(
                    score.get("finite_coverage"), "score finite_coverage"
                ),
            ),
            training_period=_required_string(payload["training_period"], "training_period"),
            validation_period=_required_string(payload["validation_period"], "validation_period"),
            schema_version=_required_schema_version(payload["schema_version"]),
        )


def _validate_decile_bins(
    bin_edges: tuple[float, ...],
    bin_masses: tuple[float, ...],
    name: str,
) -> None:
    if len(bin_edges) != 11 or len(bin_masses) != 10:
        raise ValueError(f"{name} must contain ten decile bins")
    edges = tuple(_finite_number(value, f"{name} bin edge") for value in bin_edges)
    masses = tuple(_finite_number(value, f"{name} bin mass") for value in bin_masses)
    if any(right <= left for left, right in zip(edges, edges[1:], strict=False)):
        raise ValueError(f"{name} bin edges must be strictly increasing")
    if any(value < 0 for value in masses) or not math.isclose(
        sum(masses), 1.0, rel_tol=1e-6, abs_tol=1e-9
    ):
        raise ValueError(f"{name} bin masses must be non-negative and sum to one")


def _feature_distribution_from_mapping(value: object) -> FeatureDistributionReference:
    if not isinstance(value, Mapping):
        raise ValueError("reference artifact feature entry is invalid")
    required = {
        "bin_edges",
        "bin_masses",
        "feature_name",
        "iqr",
        "median",
        "missing_rate",
        "transform_id",
    }
    if set(value) != required:
        raise ValueError("reference artifact feature schema is invalid")
    return FeatureDistributionReference(
        feature_name=_required_string(value["feature_name"], "feature_name"),
        bin_edges=_required_float_tuple(value["bin_edges"], "feature bin_edges"),
        bin_masses=_required_float_tuple(value["bin_masses"], "feature bin_masses"),
        missing_rate=_required_float(value["missing_rate"], "feature missing_rate"),
        median=_required_float(value["median"], "feature median"),
        iqr=_required_float(value["iqr"], "feature iqr"),
        transform_id=_required_string(value["transform_id"], "feature transform_id"),
    )


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"reference artifact {name} must be a non-empty string")
    return value


def _required_float(value: object, name: str) -> float:
    return _finite_number(value, f"reference artifact {name}")


def _required_float_tuple(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"reference artifact {name} must be a sequence")
    return tuple(_required_float(item, name) for item in value)


def _required_schema_version(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != 1:
        raise ValueError("reference artifact schema_version is unsupported")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _write_exclusive(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class FeatureHealth:
    feature_name: str
    finite_coverage: float
    missing_rate: float
    psi: PsiResult


@dataclass(frozen=True)
class DataHealthReport:
    row_coverage: float
    universe_size: int
    blocking_reasons: tuple[str, ...]
    investigation_alerts: tuple[str, ...]
    features: tuple[FeatureHealth, ...]


def evaluate_feature_data_health(
    frame: pd.DataFrame,
    *,
    signal_date: date,
    references: Sequence[FeatureReference],
    expected_universe_size: int,
    recent_universe_size_median: float,
    policy: MonitoringPolicy = DEFAULT_MONITORING_POLICY,
) -> DataHealthReport:
    """Evaluate PIT business keys and feature distributions without choosing publish policy."""
    if not isinstance(signal_date, date):
        raise ValueError("signal_date must be a date")
    if expected_universe_size < 1 or recent_universe_size_median <= 0:
        raise ValueError("expected universe size and recent median must be positive")
    if (
        not references
        or len({reference.feature_name for reference in references}) != len(references)
    ):
        raise ValueError("feature references must be non-empty and unique")
    required = {"signal_date", "ts_code", "available_date"}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"feature frame is missing columns: {missing}")
    normalized = frame.copy(deep=True)
    normalized["signal_date"] = pd.to_datetime(normalized["signal_date"], errors="raise").dt.date
    normalized["available_date"] = pd.to_datetime(
        normalized["available_date"], errors="raise"
    ).dt.date
    scoped = normalized.loc[normalized["signal_date"] == signal_date].copy()
    blockers: list[str] = []
    alerts: list[str] = []
    keys = ["signal_date", "ts_code"]
    if scoped.duplicated(keys).any():
        blockers.append("DUPLICATE_BUSINESS_KEY")
    if scoped["ts_code"].isna().any() or scoped["ts_code"].astype(str).str.strip().eq("").any():
        blockers.append("INVALID_BUSINESS_KEY")
    if scoped["available_date"].isna().any():
        blockers.append("MISSING_AVAILABLE_DATE")
    if (scoped["available_date"] > signal_date).any():
        blockers.append("FUTURE_AVAILABLE_DATE")
    universe_size = int(len(scoped))
    row_coverage = universe_size / expected_universe_size
    if row_coverage < policy.minimum_row_coverage:
        blockers.append("INSUFFICIENT_ROW_COVERAGE")
    lower_bound = policy.universe_size_lower_ratio * recent_universe_size_median
    upper_bound = policy.universe_size_upper_ratio * recent_universe_size_median
    if universe_size < lower_bound or universe_size > upper_bound:
        blockers.append("UNIVERSE_SIZE_OUT_OF_RANGE")
    features: list[FeatureHealth] = []
    for reference in references:
        if reference.feature_name not in scoped.columns:
            blockers.append(f"MISSING_REQUIRED_FEATURE:{reference.feature_name}")
            continue
        values = pd.to_numeric(scoped[reference.feature_name], errors="coerce")
        finite = np.isfinite(values)
        finite_values = values.loc[finite].to_numpy(dtype=float)
        finite_coverage = float(finite.sum() / expected_universe_size)
        missing_rate = 1.0 - finite_coverage
        bounded_values = np.clip(finite_values, reference.bin_edges[0], reference.bin_edges[-1])
        bin_masses = tuple(
            float(value)
            for value in np.histogram(bounded_values, bins=np.asarray(reference.bin_edges))[0]
        )
        psi = population_stability_index(
            reference.reference_bin_masses,
            bin_masses,
            policy=policy,
        )
        features.append(
            FeatureHealth(reference.feature_name, finite_coverage, missing_rate, psi)
        )
        if finite_coverage < reference.minimum_finite_coverage:
            alerts.append(f"LOW_FACTOR_COVERAGE:{reference.feature_name}")
        if (
            missing_rate - reference.reference_missing_rate
            > policy.maximum_missing_rate_increase
        ):
            alerts.append(f"MISSING_RATE_INCREASE:{reference.feature_name}")
        if psi.severity is DriftSeverity.ERROR:
            alerts.append(f"PSI_ERROR:{reference.feature_name}")
    return DataHealthReport(
        row_coverage=row_coverage,
        universe_size=universe_size,
        blocking_reasons=tuple(sorted(set(blockers))),
        investigation_alerts=tuple(alerts),
        features=tuple(features),
    )
