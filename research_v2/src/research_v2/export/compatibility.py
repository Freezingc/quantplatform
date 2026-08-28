"""Write deterministic V1-compatible prediction views without touching V1 paths."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import pandas as pd
import yaml

_MODEL_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_MANIFEST_NAME = "manifest.json"
_CSV_NAME = "predictions.csv"
_PKL_NAME = "predictions.pkl"


class UnsafeExportPath(ValueError):
    """Raised when an export root would overlap a protected V1 output root."""


@dataclass(frozen=True)
class CompatibilityContract:
    csv_index_column: str
    csv_index_source: str
    csv_signal_date_format: str
    csv_value_column: str
    csv_value_source: str
    csv_float_precision: int
    pkl_columns: tuple[tuple[str, str], ...]
    pkl_signal_date_format: str
    pickle_protocol: int
    contract_hash: str


@dataclass(frozen=True)
class CompatibilityExport:
    csv_path: Path
    pkl_path: Path
    manifest_path: Path


def export_compatibility_predictions(
    predictions: pd.DataFrame,
    *,
    model_id: str,
    model_version: int,
    signal_date: date,
    source_prediction_data_version: str,
    artifact_root: Path,
    legacy_output_roots: tuple[Path, ...] = (),
) -> CompatibilityExport:
    """Publish one complete V2-owned compatibility export through an exclusive atomic link."""
    contract = load_compatibility_contract(_default_contract_path())
    _validate_identity(model_id, model_version, signal_date, source_prediction_data_version)
    canonical = _canonical_predictions(predictions, signal_date)
    destination, model_directory = _open_export_model_directory(
        artifact_root, model_id, signal_date, legacy_output_roots
    )
    staging_name = f".{destination.name}.{uuid.uuid4().hex}.tmp"
    staging_directory: int | None = None
    published = False
    try:
        if _entry_exists(model_directory, destination.name):
            raise FileExistsError(f"compatibility export is immutable: {destination}")
        staging_directory = _create_directory(model_directory, staging_name)
        _write_csv_at(staging_directory, canonical, signal_date, contract)
        _write_pickle_at(staging_directory, canonical, signal_date, contract)
        manifest = _manifest(
            model_id=model_id,
            model_version=model_version,
            signal_date=signal_date,
            source_prediction_data_version=source_prediction_data_version,
            row_count=len(canonical),
            files={
                _CSV_NAME: _sha256_at(staging_directory, _CSV_NAME),
                _PKL_NAME: _sha256_at(staging_directory, _PKL_NAME),
            },
            contract_hash=contract.contract_hash,
        )
        _write_text_at(staging_directory, _MANIFEST_NAME, _canonical_json(manifest) + "\n")
        os.fsync(staging_directory)
        try:
            # An exclusive symlink prevents replacing a concurrent empty destination.
            os.symlink(
                staging_name,
                destination.name,
                target_is_directory=True,
                dir_fd=model_directory,
            )
            published = True
            os.fsync(model_directory)
        except FileExistsError as error:
            raise FileExistsError(f"compatibility export is immutable: {destination}") from error
    finally:
        if staging_directory is not None:
            os.close(staging_directory)
        if not published:
            _remove_directory(model_directory, staging_name)
        os.close(model_directory)
    return CompatibilityExport(
        csv_path=destination / _CSV_NAME,
        pkl_path=destination / _PKL_NAME,
        manifest_path=destination / _MANIFEST_NAME,
    )


def verify_compatibility_export(manifest_path: Path) -> bool:
    """Verify an immutable compatibility export using only its manifest and file hashes."""
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"compatibility manifest cannot be loaded: {manifest_path}") from error
    if not isinstance(document, dict):
        raise ValueError("compatibility manifest is invalid")
    expected = {
        "files",
        "model_id",
        "model_version",
        "row_count",
        "schema_version",
        "signal_date",
        "source_prediction_data_version",
        "contract_hash",
    }
    if set(document) != expected or document.get("schema_version") != 1:
        raise ValueError("compatibility manifest schema is invalid")
    signal_date = _parse_date(document.get("signal_date"))
    _validate_identity(
        document.get("model_id"),
        document.get("model_version"),
        signal_date,
        document.get("source_prediction_data_version"),
    )
    _validate_manifest_layout(manifest_path, document["model_id"], signal_date)
    _validate_publication_link(manifest_path)
    contract_hash = document.get("contract_hash")
    if not isinstance(contract_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", contract_hash):
        raise ValueError("compatibility manifest contract hash is invalid")
    if contract_hash != _sha256(_default_contract_path()):
        return False
    row_count = document.get("row_count")
    files = document.get("files")
    if (
        isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count < 1
        or not isinstance(files, dict)
        or set(files) != {_CSV_NAME, _PKL_NAME}
    ):
        raise ValueError("compatibility manifest contents are invalid")
    for filename, expected_hash in files.items():
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValueError("compatibility manifest file hash is invalid")
        if _sha256(manifest_path.parent / filename) != expected_hash:
            return False
    try:
        return len(pd.read_csv(manifest_path.parent / _CSV_NAME)) == row_count
    except (OSError, UnicodeDecodeError, pd.errors.ParserError):
        return False


def load_compatibility_contract(path: Path) -> CompatibilityContract:
    """Load the frozen V1 mapping and reject incomplete or unknown contract fields."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"compatibility contract cannot be loaded: {path}") from error
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "csv", "pkl"}:
        raise ValueError("compatibility contract schema is invalid")
    csv = raw.get("csv")
    pkl = raw.get("pkl")
    if (
        raw.get("schema_version") != 1
        or not isinstance(csv, dict)
        or set(csv) != {"index", "signal_date_header", "value"}
        or not isinstance(pkl, dict)
        or set(pkl) != {"columns", "protocol"}
    ):
        raise ValueError("compatibility contract schema is invalid")
    csv_index = csv.get("index")
    csv_header = csv.get("signal_date_header")
    csv_value = csv.get("value")
    columns = pkl.get("columns")
    if (
        not isinstance(csv_index, dict)
        or csv_index != {"name": "ts_code", "source": "ts_code", "transform": "identity"}
        or not isinstance(csv_header, dict)
        or csv_header
        != {"source": "signal_date", "transform": "date_format", "format": "%Y%m%d"}
        or not isinstance(csv_value, dict)
        or csv_value.get("name") != "score"
        or csv_value.get("source") != "score"
        or csv_value.get("transform") != "fixed_decimal"
        or isinstance(csv_value.get("float_precision"), bool)
        or csv_value.get("float_precision") != 10
        or not isinstance(columns, dict)
        or tuple(columns.items()) != (
            ("instrument", {"source": "ts_code", "transform": "identity"}),
            (
                "datetime",
                {
                    "source": "signal_date",
                    "transform": "date_format",
                    "format": "%Y-%m-%d",
                },
            ),
            ("score", {"source": "score", "transform": "identity"}),
        )
        or pkl.get("protocol") != 4
    ):
        raise ValueError("compatibility contract values are invalid")
    return CompatibilityContract(
        csv_index_column=csv_index["name"],
        csv_index_source=csv_index["source"],
        csv_signal_date_format=csv_header["format"],
        csv_value_column=csv_value["name"],
        csv_value_source=csv_value["source"],
        csv_float_precision=csv_value["float_precision"],
        pkl_columns=tuple((target, mapping["source"]) for target, mapping in columns.items()),
        pkl_signal_date_format=columns["datetime"]["format"],
        pickle_protocol=pkl["protocol"],
        contract_hash=_sha256(path),
    )


def _canonical_predictions(predictions: pd.DataFrame, signal_date: date) -> pd.DataFrame:
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("compatibility export predictions must be a DataFrame")
    required = {"signal_date", "ts_code", "score"}
    if missing := sorted(required.difference(predictions.columns)):
        raise ValueError(f"compatibility export predictions are missing columns: {missing}")
    frame = predictions.loc[:, ["signal_date", "ts_code", "score"]].copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.date
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    if (
        frame.empty
        or frame["signal_date"].nunique() != 1
        or frame["signal_date"].iloc[0] != signal_date
        or frame["ts_code"].str.strip().eq("").any()
        or frame.duplicated(["signal_date", "ts_code"]).any()
        or not all(math.isfinite(value) for value in frame["score"])
    ):
        raise ValueError("compatibility export predictions are invalid")
    return frame.sort_values("ts_code", kind="mergesort", ignore_index=True)


def _compatibility_csv(
    predictions: pd.DataFrame, signal_date: date, contract: CompatibilityContract
) -> pd.DataFrame:
    header = signal_date.strftime(contract.csv_signal_date_format)
    sources = {"ts_code": predictions["ts_code"], "score": predictions["score"]}
    return pd.DataFrame(
        {
            contract.csv_index_column: sources[contract.csv_index_source],
            header: sources[contract.csv_value_source],
        }
    )


def _compatibility_pickle(
    predictions: pd.DataFrame, signal_date: date, contract: CompatibilityContract
) -> pd.DataFrame:
    sources = {
        "ts_code": predictions["ts_code"],
        "signal_date": signal_date.strftime(contract.pkl_signal_date_format),
        "score": predictions["score"],
    }
    return pd.DataFrame({target: sources[source] for target, source in contract.pkl_columns})


def _manifest(
    *,
    model_id: str,
    model_version: int,
    signal_date: date,
    source_prediction_data_version: str,
    row_count: int,
    files: dict[str, str],
    contract_hash: str,
) -> dict[str, object]:
    return {
        "files": files,
        "model_id": model_id,
        "model_version": model_version,
        "row_count": row_count,
        "schema_version": 1,
        "signal_date": signal_date.isoformat(),
        "source_prediction_data_version": source_prediction_data_version,
        "contract_hash": contract_hash,
    }


def _open_export_model_directory(
    artifact_root: Path,
    model_id: str,
    signal_date: date,
    legacy_output_roots: tuple[Path, ...],
) -> tuple[Path, int]:
    resolved_root = artifact_root.resolve()
    requested = resolved_root / "exports" / "v2" / model_id / signal_date.isoformat()
    resolved_parent = requested.parent.resolve()
    if not resolved_parent.is_relative_to(resolved_root):
        raise UnsafeExportPath(
            f"compatibility export destination is outside artifact root: {resolved_root}"
        )
    destination = resolved_parent / requested.name
    _require_safe_destination(destination, legacy_output_roots)
    try:
        resolved_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        message = f"compatibility export artifact root cannot be created: {resolved_root}"
        raise UnsafeExportPath(message) from error

    current = _open_directory(resolved_root)
    try:
        for part in ("exports", "v2", model_id):
            child = _open_or_create_directory(current, part)
            os.close(current)
            current = child
    except Exception:
        os.close(current)
        raise
    return destination, current


def _open_directory(path: Path) -> int:
    try:
        return os.open(path, os.O_RDONLY | _directory_flags())
    except OSError as error:
        raise UnsafeExportPath(f"compatibility export cannot open directory: {path}") from error


def _open_or_create_directory(parent_directory: int, name: str) -> int:
    with suppress(FileExistsError):
        os.mkdir(name, mode=0o755, dir_fd=parent_directory)
    try:
        return os.open(name, os.O_RDONLY | _directory_flags(), dir_fd=parent_directory)
    except OSError as error:
        raise UnsafeExportPath(f"compatibility export path component is unsafe: {name}") from error


def _create_directory(parent_directory: int, name: str) -> int:
    os.mkdir(name, mode=0o755, dir_fd=parent_directory)
    try:
        return os.open(name, os.O_RDONLY | _directory_flags(), dir_fd=parent_directory)
    except OSError:
        os.rmdir(name, dir_fd=parent_directory)
        raise


def _entry_exists(parent_directory: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_directory, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _write_csv_at(
    directory: int,
    predictions: pd.DataFrame,
    signal_date: date,
    contract: CompatibilityContract,
) -> None:
    with _open_text_writer(directory, _CSV_NAME) as output:
        _compatibility_csv(predictions, signal_date, contract).to_csv(
            output,
            index=False,
            float_format=f"%.{contract.csv_float_precision}f",
            lineterminator="\n",
        )
        _sync(output)


def _write_pickle_at(
    directory: int,
    predictions: pd.DataFrame,
    signal_date: date,
    contract: CompatibilityContract,
) -> None:
    with _open_binary_writer(directory, _PKL_NAME) as output:
        _compatibility_pickle(predictions, signal_date, contract).to_pickle(
            output, protocol=contract.pickle_protocol
        )
        _sync(output)


def _write_text_at(directory: int, filename: str, value: str) -> None:
    with _open_text_writer(directory, filename) as output:
        output.write(value)
        _sync(output)


def _open_text_writer(directory: int, filename: str) -> TextIO:
    return os.fdopen(_open_file_for_write(directory, filename), "w", encoding="utf-8", newline="")


def _open_binary_writer(directory: int, filename: str) -> BinaryIO:
    return os.fdopen(_open_file_for_write(directory, filename), "wb")


def _open_file_for_write(directory: int, filename: str) -> int:
    return os.open(
        filename,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | _no_follow_flag(),
        mode=0o644,
        dir_fd=directory,
    )


def _sync(output: TextIO | BinaryIO) -> None:
    output.flush()
    os.fsync(output.fileno())


def _sha256_at(directory: int, filename: str) -> str:
    try:
        descriptor = os.open(filename, os.O_RDONLY | _no_follow_flag(), dir_fd=directory)
    except OSError as error:
        raise ValueError(f"compatibility export file cannot be read: {filename}") from error
    with os.fdopen(descriptor, "rb") as source:
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_directory(parent_directory: int, name: str) -> None:
    try:
        directory = os.open(name, os.O_RDONLY | _directory_flags(), dir_fd=parent_directory)
    except OSError:
        return
    try:
        for filename in (_CSV_NAME, _PKL_NAME, _MANIFEST_NAME):
            with suppress(FileNotFoundError):
                os.unlink(filename, dir_fd=directory)
    finally:
        os.close(directory)
    with suppress(OSError):
        os.rmdir(name, dir_fd=parent_directory)


def _directory_flags() -> int:
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if not isinstance(directory_flag, int):
        raise UnsafeExportPath("compatibility export requires directory descriptor support")
    return directory_flag | _no_follow_flag()


def _no_follow_flag() -> int:
    no_follow_flag = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(no_follow_flag, int):
        raise UnsafeExportPath("compatibility export requires no-follow directory support")
    return no_follow_flag


def _require_safe_destination(destination: Path, legacy_output_roots: tuple[Path, ...]) -> None:
    if not legacy_output_roots:
        raise UnsafeExportPath("compatibility export requires explicit V1 output roots")
    for root in legacy_output_roots:
        resolved = root.resolve()
        if destination.is_relative_to(resolved):
            raise UnsafeExportPath(f"compatibility export would write under V1 root: {resolved}")


def _validate_identity(
    model_id: object,
    model_version: object,
    signal_date: date,
    source_prediction_data_version: object,
) -> None:
    if not isinstance(model_id, str) or not _MODEL_ID.fullmatch(model_id):
        raise ValueError("compatibility export model_id is invalid")
    if isinstance(model_version, bool) or not isinstance(model_version, int) or model_version < 1:
        raise ValueError("compatibility export model_version is invalid")
    if (
        not isinstance(source_prediction_data_version, str)
        or not source_prediction_data_version.strip()
    ):
        raise ValueError("compatibility export source data version is invalid")
    if not isinstance(signal_date, date):
        raise ValueError("compatibility export signal_date is invalid")


def _parse_date(value: object) -> date:
    if not isinstance(value, str):
        raise ValueError("compatibility manifest signal_date is invalid")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("compatibility manifest signal_date is invalid") from error


def _validate_manifest_layout(manifest_path: Path, model_id: str, signal_date: date) -> None:
    expected_parent = manifest_path.parent
    if (
        manifest_path.name != _MANIFEST_NAME
        or expected_parent.name != signal_date.isoformat()
        or expected_parent.parent.name != model_id
        or expected_parent.parent.parent.name != "v2"
        or expected_parent.parent.parent.parent.name != "exports"
    ):
        raise ValueError("compatibility manifest path is outside the V2 export layout")


def _validate_publication_link(manifest_path: Path) -> None:
    publication = manifest_path.parent
    if not publication.is_symlink():
        raise ValueError(
            "compatibility manifest is not published through an exclusive publication link"
        )
    try:
        target = os.readlink(publication)
    except OSError as error:
        raise ValueError("compatibility publication link cannot be read") from error
    expected = rf"\.{re.escape(publication.name)}\.[0-9a-f]{{32}}\.tmp"
    if not re.fullmatch(expected, target):
        raise ValueError(
            "compatibility manifest is not published through an exclusive publication link"
        )
    staging = publication.parent / target
    if staging.is_symlink() or not staging.is_dir():
        raise ValueError(
            "compatibility manifest is not published through an exclusive publication link"
        )


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"compatibility export file cannot be read: {path}") from error


def _default_contract_path() -> Path:
    return Path(__file__).with_name("v1_compatible.yaml")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
