"""Immutable selection and prediction evidence for ordinary model walk-forward tests."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd

from research_v2.baselines.orchestration import BaselineSelectionRecord
from research_v2.datasets.manifest import build_dataset_manifest

_EVIDENCE_ID = re.compile(r"^[0-9a-f]{64}$")
_SELECTION_EVIDENCE_KEY_ENV = "QRV2_SELECTION_EVIDENCE_HMAC_KEY"
_PREDICTION_EVIDENCE_KEY_ENV = "QRV2_PREDICTION_EVIDENCE_HMAC_KEY"
PredictionKind = Literal["candidate", "best_simple"]


@dataclass(frozen=True)
class BaselineSelectionEvidence:
    evidence_id: str
    fold_id: str
    test_dataset_id: str
    selected_baseline_id: str
    validation_rank_ic: float


@dataclass(frozen=True)
class ModelSelectionEvidence:
    evidence_id: str
    fold_id: str
    model_family: str
    model_artifact_id: str
    model_bundle_digest: str
    training_dataset_id: str
    validation_dataset_id: str
    test_dataset_id: str
    baseline_selection_id: str
    validation_rank_ic: float


@dataclass(frozen=True)
class FrozenPredictionArtifact:
    prediction_id: str
    selection_id: str
    kind: PredictionKind
    fold_id: str
    dataset_id: str
    model_id: str
    scores: pd.DataFrame


class SelectionEvidenceVerifier:
    """Read immutable selection decisions using the evaluator-owned HMAC key."""

    def __init__(self, root: Path, *, verification_key: bytes) -> None:
        if not isinstance(verification_key, bytes) or not verification_key:
            raise ValueError("selection evidence verification_key must be non-empty bytes")
        self._root = root
        self._verification_key = verification_key

    @classmethod
    def from_environment(
        cls,
        root: Path,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> SelectionEvidenceVerifier:
        """Build the evaluator verifier from its deployment-only environment secret."""
        environment = os.environ if environ is None else environ
        secret = environment.get(_SELECTION_EVIDENCE_KEY_ENV, "")
        if not isinstance(secret, str) or not secret:
            raise ValueError(
                "selection evidence verification requires "
                f"{_SELECTION_EVIDENCE_KEY_ENV}"
            )
        return cls(root, verification_key=secret.encode("utf-8"))

    def load_baseline_selection(self, evidence_id: str) -> BaselineSelectionEvidence:
        payload = self._load("baseline", evidence_id, expected_type="baseline_selection")
        _validate_baseline_payload(payload)
        return BaselineSelectionEvidence(
            evidence_id=evidence_id,
            fold_id=_payload_text(payload, "fold_id"),
            test_dataset_id=_payload_text(payload, "test_dataset_id"),
            selected_baseline_id=_payload_baseline(payload),
            validation_rank_ic=_payload_metric(payload, "validation_rank_ic"),
        )

    def load_model_selection(self, evidence_id: str) -> ModelSelectionEvidence:
        payload = self._load("model", evidence_id, expected_type="model_selection")
        baseline_selection_id = _payload_text(payload, "baseline_selection_id")
        baseline = self.load_baseline_selection(baseline_selection_id)
        fold_id = _payload_text(payload, "fold_id")
        test_dataset_id = _payload_text(payload, "test_dataset_id")
        if baseline.fold_id != fold_id or baseline.test_dataset_id != test_dataset_id:
            raise ValueError(
                "model selection baseline evidence does not match its fold or test dataset"
            )
        return ModelSelectionEvidence(
            evidence_id=evidence_id,
            fold_id=fold_id,
            model_family=_payload_text(payload, "model_family"),
            model_artifact_id=_payload_text(payload, "model_artifact_id"),
            model_bundle_digest=_payload_digest(payload, "model_bundle_digest"),
            training_dataset_id=_payload_text(payload, "training_dataset_id"),
            validation_dataset_id=_payload_text(payload, "validation_dataset_id"),
            test_dataset_id=test_dataset_id,
            baseline_selection_id=baseline_selection_id,
            validation_rank_ic=_payload_metric(payload, "validation_rank_ic"),
        )

    def _load(self, kind: str, evidence_id: str, *, expected_type: str) -> dict[str, object]:
        _validate_evidence_id(evidence_id)
        path = self._root / kind / f"{evidence_id}.json" / "record.json"
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"selection evidence cannot be loaded: {evidence_id}") from error
        if not isinstance(document, dict) or document.get("evidence_id") != evidence_id:
            raise ValueError("selection evidence record is malformed")
        payload = document.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != expected_type:
            raise ValueError("selection evidence type is malformed")
        typed = cast(dict[str, object], payload)
        signature = document.get("signature")
        if not isinstance(signature, str) or not hmac.compare_digest(
            signature,
            _signature(self._verification_key, evidence_id, typed),
        ):
            raise ValueError("selection evidence signature verification failed")
        if _evidence_id(typed) != evidence_id:
            raise ValueError("selection evidence content hash does not match its identity")
        return typed


class SelectionEvidenceStore(SelectionEvidenceVerifier):
    """Issue signed validation decisions in the trusted pre-TEST selection service."""

    def __init__(self, root: Path, *, signing_key: bytes) -> None:
        super().__init__(root, verification_key=signing_key)
        self._signing_key = signing_key

    def write_baseline_selection(
        self,
        *,
        record: BaselineSelectionRecord,
        test_dataset_id: str,
    ) -> BaselineSelectionEvidence:
        selected_metric = _validate_baseline_selection_record(record)
        payload = {
            "fold_id": record.fold_id,
            "selected_baseline_id": record.selected_baseline_id,
            "test_dataset_id": test_dataset_id,
            "type": "baseline_selection",
            "unselected_reasons": [list(reason) for reason in record.unselected_reasons],
            "validation_rank_ic": selected_metric,
            "validation_scores": [
                {
                    "baseline_id": score.baseline_id,
                    "invalid_day_count": score.invalid_day_count,
                    "mean_rank_ic": _optional_metric(score.mean_rank_ic),
                    "valid_day_count": score.valid_day_count,
                }
                for score in record.validation_scores
            ],
        }
        evidence = BaselineSelectionEvidence(
            evidence_id=_evidence_id(payload),
            fold_id=_required_text(record.fold_id, "fold_id"),
            test_dataset_id=_required_text(test_dataset_id, "test_dataset_id"),
            selected_baseline_id=record.selected_baseline_id,
            validation_rank_ic=selected_metric,
        )
        self._write("baseline", evidence.evidence_id, payload)
        return evidence

    def write_model_selection(
        self,
        *,
        fold_id: str,
        model_family: str,
        model_artifact_id: str,
        model_bundle_digest: str,
        training_dataset_id: str,
        validation_dataset_id: str,
        test_dataset_id: str,
        baseline_selection_id: str,
        validation_rank_ic: float,
    ) -> ModelSelectionEvidence:
        baseline = self.load_baseline_selection(baseline_selection_id)
        if baseline.fold_id != fold_id or baseline.test_dataset_id != test_dataset_id:
            raise ValueError(
                "model selection must reference the same fold and test dataset baseline"
            )
        payload = {
            "baseline_selection_id": baseline_selection_id,
            "fold_id": fold_id,
            "model_artifact_id": model_artifact_id,
            "model_bundle_digest": _required_digest(
                model_bundle_digest,
                "model_bundle_digest",
            ),
            "model_family": model_family,
            "test_dataset_id": test_dataset_id,
            "training_dataset_id": training_dataset_id,
            "type": "model_selection",
            "validation_dataset_id": validation_dataset_id,
            "validation_rank_ic": _finite_metric(validation_rank_ic, "validation_rank_ic"),
        }
        evidence = ModelSelectionEvidence(
            evidence_id=_evidence_id(payload),
            fold_id=_required_text(fold_id, "fold_id"),
            model_family=_required_text(model_family, "model_family"),
            model_artifact_id=_required_text(model_artifact_id, "model_artifact_id"),
            model_bundle_digest=_required_digest(
                model_bundle_digest,
                "model_bundle_digest",
            ),
            training_dataset_id=_required_text(training_dataset_id, "training_dataset_id"),
            validation_dataset_id=_required_text(validation_dataset_id, "validation_dataset_id"),
            test_dataset_id=_required_text(test_dataset_id, "test_dataset_id"),
            baseline_selection_id=_required_text(baseline_selection_id, "baseline_selection_id"),
            validation_rank_ic=float(validation_rank_ic),
        )
        self._write("model", evidence.evidence_id, payload)
        return evidence

    def _write(self, kind: str, evidence_id: str, payload: dict[str, object]) -> None:
        destination = self._root / kind / f"{evidence_id}.json"
        if destination.exists():
            raise FileExistsError(f"selection evidence is immutable: {evidence_id}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".selection-", dir=destination.parent))
        try:
            document = {
                "evidence_id": evidence_id,
                "payload": payload,
                "signature": _signature(self._signing_key, evidence_id, payload),
            }
            path = temporary / "record.json"
            path.write_text(_canonical_json(document) + "\n", encoding="utf-8")
            temporary.rename(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

class FrozenPredictionStore:
    """Issue and verify signed score outputs after immutable pre-TEST selection."""

    def __init__(self, root: Path, *, signing_key: bytes) -> None:
        if not isinstance(signing_key, bytes) or not signing_key:
            raise ValueError("prediction evidence signing_key must be non-empty bytes")
        self._root = root
        self._signing_key = signing_key

    @classmethod
    def from_environment(
        cls,
        root: Path,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> FrozenPredictionStore:
        """Build the evaluator prediction verifier from its deployment-only environment key."""
        environment = os.environ if environ is None else environ
        secret = environment.get(_PREDICTION_EVIDENCE_KEY_ENV, "")
        if not isinstance(secret, str) or not secret:
            raise ValueError(
                "prediction evidence verification requires "
                f"{_PREDICTION_EVIDENCE_KEY_ENV}"
            )
        return cls(root, signing_key=secret.encode("utf-8"))

    def write(
        self,
        *,
        selection_id: str,
        kind: PredictionKind,
        fold_id: str,
        dataset_id: str,
        model_id: str,
        scores: pd.DataFrame,
        selections: SelectionEvidenceVerifier,
    ) -> FrozenPredictionArtifact:
        if kind == "candidate":
            model_selection = selections.load_model_selection(selection_id)
            selection_fold_id = model_selection.fold_id
            selection_dataset_id = model_selection.test_dataset_id
            expected_model_id = model_selection.model_artifact_id
        elif kind == "best_simple":
            baseline_selection = selections.load_baseline_selection(selection_id)
            selection_fold_id = baseline_selection.fold_id
            selection_dataset_id = baseline_selection.test_dataset_id
            expected_model_id = baseline_selection.selected_baseline_id
        else:
            raise ValueError("prediction kind must be candidate or best_simple")
        if selection_fold_id != fold_id or selection_dataset_id != dataset_id:
            raise ValueError(
                "prediction evidence must match the frozen selection fold and test dataset"
            )
        if model_id != expected_model_id:
            raise ValueError("prediction model_id does not match its frozen selection")
        normalized = _normalize_scores(scores)
        metadata = {
            "dataset_id": dataset_id,
            "fold_id": fold_id,
            "kind": kind,
            "model_id": model_id,
            "selection_id": selection_id,
        }
        manifest = build_dataset_manifest(
            normalized,
            business_keys=("signal_date", "ts_code"),
            metadata=metadata,
        )
        destination = self._root / manifest.dataset_id
        if destination.exists():
            raise FileExistsError(f"frozen prediction artifact is immutable: {manifest.dataset_id}")
        self._root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".prediction-", dir=self._root))
        try:
            parquet = temporary / "scores.parquet"
            normalized.to_parquet(parquet, engine="pyarrow", compression="zstd", index=False)
            document = {
                "metadata": metadata,
                "prediction_id": manifest.dataset_id,
                "sha256": _sha256_file(parquet),
            }
            document["signature"] = _prediction_signature(self._signing_key, document)
            (temporary / "manifest.json").write_text(
                _canonical_json(document) + "\n",
                encoding="utf-8",
            )
            temporary.rename(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return FrozenPredictionArtifact(
            prediction_id=manifest.dataset_id,
            selection_id=selection_id,
            kind=kind,
            fold_id=fold_id,
            dataset_id=dataset_id,
            model_id=model_id,
            scores=normalized,
        )

    def load(
        self,
        prediction_id: str,
        *,
        selections: SelectionEvidenceVerifier,
    ) -> FrozenPredictionArtifact:
        _validate_evidence_id(prediction_id)
        root = self._root / prediction_id
        try:
            document = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            scores_path = root / "scores.parquet"
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"frozen prediction artifact cannot be loaded: {prediction_id}"
            ) from error
        if not isinstance(document, dict) or document.get("prediction_id") != prediction_id:
            raise ValueError("frozen prediction manifest is malformed")
        expected_hash = document.get("sha256")
        metadata = document.get("metadata")
        if not isinstance(expected_hash, str) or not isinstance(metadata, dict):
            raise ValueError("frozen prediction manifest is incomplete")
        signature = document.get("signature")
        signed_payload = {
            "metadata": metadata,
            "prediction_id": prediction_id,
            "sha256": expected_hash,
        }
        if not isinstance(signature, str) or not hmac.compare_digest(
            signature,
            _prediction_signature(self._signing_key, signed_payload),
        ):
            raise ValueError("frozen prediction signature verification failed")
        scores_bytes = scores_path.read_bytes()
        if hashlib.sha256(scores_bytes).hexdigest() != expected_hash:
            raise ValueError("frozen prediction artifact integrity verification failed")
        normalized = _normalize_scores(pd.read_parquet(BytesIO(scores_bytes), engine="pyarrow"))
        manifest = build_dataset_manifest(
            normalized,
            business_keys=("signal_date", "ts_code"),
            metadata=metadata,
        )
        if manifest.dataset_id != prediction_id:
            raise ValueError("frozen prediction content hash does not match its identity")
        kind = metadata.get("kind")
        if kind not in {"candidate", "best_simple"}:
            raise ValueError("frozen prediction kind is invalid")
        selection_id = _payload_text(metadata, "selection_id")
        fold_id = _payload_text(metadata, "fold_id")
        dataset_id = _payload_text(metadata, "dataset_id")
        model_id = _payload_text(metadata, "model_id")
        if kind == "candidate":
            model_selection = selections.load_model_selection(selection_id)
            selection_fold_id = model_selection.fold_id
            selection_dataset_id = model_selection.test_dataset_id
            expected_model_id = model_selection.model_artifact_id
        else:
            baseline_selection = selections.load_baseline_selection(selection_id)
            selection_fold_id = baseline_selection.fold_id
            selection_dataset_id = baseline_selection.test_dataset_id
            expected_model_id = baseline_selection.selected_baseline_id
        if (
            selection_fold_id != fold_id
            or selection_dataset_id != dataset_id
            or expected_model_id != model_id
        ):
            raise ValueError("frozen prediction does not match its selection evidence")
        return FrozenPredictionArtifact(
            prediction_id=prediction_id,
            selection_id=selection_id,
            kind=cast(PredictionKind, kind),
            fold_id=fold_id,
            dataset_id=dataset_id,
            model_id=model_id,
            scores=normalized,
        )


def _normalize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    expected = ("signal_date", "ts_code", "score")
    if not isinstance(scores, pd.DataFrame) or tuple(scores.columns) != expected:
        raise ValueError(
            "frozen prediction scores must exactly contain signal_date, ts_code, score"
        )
    normalized = scores.copy(deep=True)
    normalized["signal_date"] = pd.to_datetime(normalized["signal_date"], errors="raise").dt.date
    if normalized["signal_date"].isna().any() or normalized["ts_code"].isna().any():
        raise ValueError("frozen prediction business keys cannot be null")
    normalized["ts_code"] = normalized["ts_code"].astype(str)
    if normalized["ts_code"].str.strip().eq("").any():
        raise ValueError("frozen prediction ts_code cannot be empty")
    if normalized.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("frozen prediction scores contain duplicate business keys")
    numeric = pd.to_numeric(normalized["score"], errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("frozen prediction scores must be finite")
    normalized["score"] = numeric.astype(float)
    return normalized.sort_values(["signal_date", "ts_code"], kind="mergesort", ignore_index=True)


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty")
    return value


def _required_digest(value: str, name: str) -> str:
    if not isinstance(value, str) or not _EVIDENCE_ID.fullmatch(value):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return value


def _finite_metric(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _payload_text(payload: dict[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"selection evidence {name} is invalid")
    return value


def _payload_digest(payload: dict[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str):
        raise ValueError(f"selection evidence {name} must be a SHA-256 hex digest")
    return _required_digest(value, f"selection evidence {name}")


def _payload_metric(payload: dict[str, object], name: str) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"selection evidence {name} is invalid")
    return _finite_metric(float(value), name)


def _payload_baseline(payload: dict[str, object]) -> str:
    baseline = _payload_text(payload, "selected_baseline_id")
    if baseline not in {"B0", "B1", "B2", "B3"}:
        raise ValueError("selection evidence selected_baseline_id is invalid")
    return baseline


def _validate_baseline_selection_record(record: BaselineSelectionRecord) -> float:
    if not isinstance(record, BaselineSelectionRecord):
        raise TypeError("baseline selection evidence requires a BaselineSelectionRecord")
    if not record.fold_id.strip() or record.selected_baseline_id not in {"B0", "B1", "B2", "B3"}:
        raise ValueError("baseline selection record identity is invalid")
    scores = {score.baseline_id: score for score in record.validation_scores}
    expected = {"B0", "B1", "B2", "B3"}
    if set(scores) != expected or len(scores) != len(record.validation_scores):
        raise ValueError("baseline selection record must contain exactly B0,B1,B2,B3 scores")
    means = {
        baseline_id: _optional_metric(score.mean_rank_ic) for baseline_id, score in scores.items()
    }
    winner = _baseline_winner(means)
    if winner != record.selected_baseline_id:
        raise ValueError(
            "baseline selection record selected winner does not match validation scores"
        )
    reasons = {baseline_id for baseline_id, _ in record.unselected_reasons}
    if reasons != expected.difference({winner}):
        raise ValueError("baseline selection record must explain every unselected baseline")
    return _finite_metric(scores[winner].mean_rank_ic, "selected validation_rank_ic")


def _validate_baseline_payload(payload: dict[str, object]) -> None:
    selected = _payload_baseline(payload)
    values = payload.get("validation_scores")
    if not isinstance(values, list):
        raise ValueError("selection evidence validation_scores is invalid")
    means: dict[str, float | None] = {}
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("selection evidence validation_scores is invalid")
        baseline_id = value.get("baseline_id")
        metric = value.get("mean_rank_ic")
        if not isinstance(baseline_id, str) or baseline_id in means:
            raise ValueError("selection evidence validation_scores is invalid")
        if metric is None:
            means[baseline_id] = None
        elif isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise ValueError("selection evidence validation_scores is invalid")
        else:
            means[baseline_id] = _finite_metric(float(metric), "validation_scores")
    if set(means) != {"B0", "B1", "B2", "B3"} or _baseline_winner(means) != selected:
        raise ValueError("selection evidence selected winner does not match validation scores")
    if _payload_metric(payload, "validation_rank_ic") != means[selected]:
        raise ValueError("selection evidence selected validation_rank_ic is invalid")


def _baseline_winner(means: dict[str, float | None]) -> str:
    eligible = [(baseline_id, value) for baseline_id, value in means.items() if value is not None]
    if not eligible:
        raise ValueError("baseline selection record has no valid validation winner")
    complexity = {"B0": 0, "B1": 1, "B2": 2, "B3": 3}
    return min(eligible, key=lambda item: (-item[1], complexity[item[0]]))[0]


def _optional_metric(value: float) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _evidence_id(payload: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _signature(signing_key: bytes, evidence_id: str, payload: dict[str, object]) -> str:
    document = {"evidence_id": evidence_id, "payload": payload}
    return hmac.new(
        signing_key,
        _canonical_json(document).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _prediction_signature(signing_key: bytes, payload: Mapping[str, object]) -> str:
    document = {"type": "frozen_prediction", **payload}
    return hmac.new(
        signing_key,
        _canonical_json(document).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _validate_evidence_id(value: str) -> None:
    if not isinstance(value, str) or not _EVIDENCE_ID.fullmatch(value):
        raise ValueError("evidence identity must be a SHA-256 hex digest")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


__all__ = [
    "BaselineSelectionEvidence",
    "FrozenPredictionArtifact",
    "FrozenPredictionStore",
    "ModelSelectionEvidence",
    "SelectionEvidenceStore",
    "SelectionEvidenceVerifier",
]
