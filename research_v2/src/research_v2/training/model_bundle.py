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


@dataclass(frozen=True)
class _VerifiedBundleSnapshot:
    digest: str
    files: Mapping[str, bytes]


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
            self._load_verified_snapshot(model_id)
            return True
        except (OSError, ValueError):
            return False

    def bundle_path(self, model_id: str) -> Path:
        """Return one validated bundle root for a loader after it has verified integrity."""
        self._validate_model_id(model_id)
        return self._root / model_id

    def bundle_digest(self, model_id: str) -> str:
        """Return the snapshot digest a trusted selection record must bind to."""
        try:
            return self._load_verified_snapshot(model_id).digest
        except (OSError, ValueError) as error:
            raise ValueError(f"model bundle integrity verification failed: {model_id}") from error

    def load_model_card(
        self,
        model_id: str,
        *,
        expected_bundle_digest: str | None = None,
    ) -> Mapping[str, object]:
        """Read a checksum-verified model card from one immutable bundle snapshot."""
        try:
            snapshot = self._load_verified_snapshot(model_id)
        except (OSError, ValueError) as error:
            raise ValueError(f"model bundle integrity verification failed: {model_id}") from error
        if expected_bundle_digest is not None and snapshot.digest != expected_bundle_digest:
            raise ValueError("model bundle digest does not match frozen model selection")
        return self._read_json_mapping_bytes(snapshot.files["model_card.json"], "model_card.json")

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
        try:
            snapshot = self._load_verified_snapshot(model_id)
        except (OSError, ValueError) as error:
            raise ValueError(f"model bundle integrity verification failed: {model_id}") from error
        model_card = self._read_json_mapping_bytes(
            snapshot.files["model_card.json"],
            "model_card.json",
        )
        config = self._read_json_mapping_bytes(snapshot.files["config.json"], "config.json")
        signature = self._read_json_mapping_bytes(
            snapshot.files["signature.json"],
            "signature.json",
        )
        feature_document = self._read_json_mapping_bytes(
            snapshot.files["feature_order.json"],
            "feature_order.json",
        )
        preprocessing = self._read_json_mapping_bytes(
            snapshot.files["preprocessing.json"],
            "preprocessing.json",
        )
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

    def _load_verified_snapshot(self, model_id: str) -> _VerifiedBundleSnapshot:
        """Read each bundle file once and verify those exact bytes against one checksum file."""
        self._validate_model_id(model_id)
        root = self._root / model_id
        checksums = (root / "checksums.sha256").read_bytes()
        expected_files = {
            "config.json",
            "feature_order.json",
            "model_card.json",
            "preprocessing.json",
            "signature.json",
            "validation_predictions.parquet",
            *(f"seed-{seed}.pt" for seed in REQUIRED_SEEDS),
        }
        files: dict[str, bytes] = {}
        for line in checksums.decode("utf-8").splitlines():
            digest, separator, relative = line.partition("  ")
            if (
                not separator
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or relative in files
            ):
                raise ValueError("model bundle checksum manifest is malformed")
            path = (root / relative).resolve()
            if root.resolve() not in path.parents or not path.is_file():
                raise ValueError("model bundle checksum path is invalid")
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError("model bundle checksum verification failed")
            files[relative] = content
        if set(files) != expected_files:
            raise ValueError("model bundle checksum manifest is incomplete")
        return _VerifiedBundleSnapshot(
            digest=hashlib.sha256(checksums).hexdigest(),
            files=files,
        )

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
    def _read_json_mapping_bytes(content: bytes, filename: str) -> Mapping[str, object]:
        loaded = json.loads(content.decode("utf-8"))
        if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
            raise ValueError(f"model bundle metadata is not a string-keyed mapping: {filename}")
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
