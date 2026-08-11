"""Immutable complete model bundles containing exactly three audited seed artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from research_v2.training.ensemble import REQUIRED_SEEDS

_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class ModelArtifactBundle:
    model_id: str
    path: Path


class ModelArtifactBundleStore:
    """Write and verify self-contained model artifacts without serializing Python objects."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def write(
        self,
        model_id: str,
        *,
        model_card: Mapping[str, object],
        config: Mapping[str, object],
        signature: Mapping[str, object],
        feature_order: tuple[str, ...],
        preprocessing: Mapping[str, object],
        seed_checkpoints: Mapping[int, Path],
        validation_predictions: pd.DataFrame,
    ) -> ModelArtifactBundle:
        """Atomically publish a complete three-seed bundle, refusing partial or overwrite writes."""
        self._validate_model_id(model_id)
        if not feature_order or len(set(feature_order)) != len(feature_order):
            raise ValueError("feature_order must be non-empty and unique")
        if set(seed_checkpoints) != set(REQUIRED_SEEDS):
            raise ValueError("model bundle requires all required seed checkpoints")
        required_predictions = {"signal_date", "ts_code", "percentile_rank"}
        if missing := sorted(required_predictions.difference(validation_predictions.columns)):
            raise ValueError(f"validation predictions are missing columns: {missing}")
        destination = self._root / model_id
        if destination.exists():
            raise FileExistsError(f"model artifact is immutable and already exists: {model_id}")
        self._root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".model-", dir=self._root))
        try:
            self._write_json(temporary / "model_card.json", model_card)
            self._write_json(temporary / "config.json", config)
            self._write_json(temporary / "signature.json", signature)
            self._write_json(
                temporary / "feature_order.json",
                {"feature_order": list(feature_order)},
            )
            self._write_json(temporary / "preprocessing.json", preprocessing)
            for seed in REQUIRED_SEEDS:
                source = seed_checkpoints[seed]
                if not source.is_file():
                    raise ValueError(f"seed checkpoint is missing: {seed}")
                shutil.copyfile(source, temporary / f"seed-{seed}.pt")
            predictions = validation_predictions.sort_values(
                ["signal_date", "ts_code"],
                kind="mergesort",
                ignore_index=True,
            )
            predictions.to_parquet(
                temporary / "validation_predictions.parquet",
                engine="pyarrow",
                compression="zstd",
                index=False,
            )
            self._write_checksums(temporary)
            if destination.exists():
                raise FileExistsError(f"model artifact is immutable and already exists: {model_id}")
            temporary.rename(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return ModelArtifactBundle(model_id=model_id, path=destination)

    def verify(self, model_id: str) -> bool:
        try:
            self._validate_model_id(model_id)
            root = self._root / model_id
            checksum_path = root / "checksums.sha256"
            if not checksum_path.is_file():
                return False
            lines = checksum_path.read_text(encoding="utf-8").splitlines()
            expected_files = {
                "config.json",
                "feature_order.json",
                "model_card.json",
                "preprocessing.json",
                "signature.json",
                "validation_predictions.parquet",
                *(f"seed-{seed}.pt" for seed in REQUIRED_SEEDS),
            }
            actual_files: set[str] = set()
            for line in lines:
                digest, separator, relative = line.partition("  ")
                if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    return False
                path = (root / relative).resolve()
                if root.resolve() not in path.parents or not path.is_file():
                    return False
                if _sha256_file(path) != digest:
                    return False
                actual_files.add(relative)
            return actual_files == expected_files
        except (OSError, ValueError):
            return False

    def bundle_path(self, model_id: str) -> Path:
        """Return one validated bundle root for a loader after it has verified integrity."""
        self._validate_model_id(model_id)
        return self._root / model_id

    def load_metadata(
        self,
        model_id: str,
        *,
        expected_dataset_id: str,
        expected_feature_order: tuple[str, ...],
        expected_signature: Mapping[str, object],
        expected_preprocessor_hash: str,
    ) -> dict[str, Mapping[str, object]]:
        """Verify integrity, then reject bundles whose data/model contract differs from request."""
        if not self.verify(model_id):
            raise ValueError(f"model bundle integrity verification failed: {model_id}")
        root = self._root / model_id
        model_card = self._read_json_mapping(root / "model_card.json")
        config = self._read_json_mapping(root / "config.json")
        signature = self._read_json_mapping(root / "signature.json")
        feature_document = self._read_json_mapping(root / "feature_order.json")
        preprocessing = self._read_json_mapping(root / "preprocessing.json")
        if model_card.get("dataset_id") != expected_dataset_id:
            raise ValueError("model bundle dataset_id does not match the requested dataset")
        feature_order = feature_document.get("feature_order")
        if feature_order != list(expected_feature_order):
            raise ValueError(
                "model bundle feature_order does not match the requested feature order"
            )
        if dict(signature) != dict(expected_signature):
            raise ValueError("model bundle signature does not match the requested model contract")
        if preprocessing.get("state_hash") != expected_preprocessor_hash:
            raise ValueError("model bundle preprocessor state does not match the requested state")
        return {
            "config": config,
            "feature_order": feature_document,
            "model_card": model_card,
            "preprocessing": preprocessing,
            "signature": signature,
        }

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, object]) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _read_json_mapping(path: Path) -> Mapping[str, object]:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
            raise ValueError(f"model bundle metadata is not a string-keyed mapping: {path.name}")
        return loaded

    @staticmethod
    def _write_checksums(root: Path) -> None:
        entries = [
            path for path in root.iterdir() if path.is_file() and path.name != "checksums.sha256"
        ]
        lines = [f"{_sha256_file(path)}  {path.name}" for path in sorted(entries)]
        (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _validate_model_id(model_id: str) -> None:
        if not _MODEL_ID.fullmatch(model_id):
            raise ValueError("model_id contains unsupported characters")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
