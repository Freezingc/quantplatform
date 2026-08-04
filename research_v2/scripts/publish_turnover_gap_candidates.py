"""Preflight or explicitly publish the isolated turnover-gap candidate batch.

The default mode is local and write-free.  ``--publish`` is required before
the batch can be inserted into the new ``factor_repair_candidate`` V2 table;
the script never writes YesLab or a clean factor table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from research_v2.db.publish import StagedBatch, V2BatchPublisher
from research_v2.db.session import V2WriteSession
from research_v2.quality.engine import CheckResult, CheckStatus, Severity

EXPECTED_FACTOR_IDS = (
    "yeslab_377",
    "yeslab_380",
    "yeslab_382",
    "yeslab_385",
    "yeslab_387",
    "yeslab_388",
)
BLOCKED_FACTOR_IDS = frozenset({"yeslab_395"})
LOCKED_TEST_START = pd.Timestamp("2026-01-01").date()
_REQUIRED_COLUMNS = (
    "observation_date",
    "ts_code",
    "factor_id",
    "factor_version",
    "factor_name",
    "candidate_value",
    "source_value",
    "repair_method",
    "reason",
)
_BUSINESS_KEYS = ("observation_date", "ts_code", "factor_id", "factor_version")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        help="directory containing the CSV files listed by the manifest",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument(
        "--target-database",
        default=os.getenv("QRV2_DDB_DATABASE", "dfs://QuantResearchV2Test"),
    )
    return parser.parse_args(argv)


def load_candidate_scope(manifest_path: Path, candidate_dir: Path | None = None) -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("scope") != "TURNOVER_GAP_TUSHARE_REPAIR_AUDIT_V1":
        raise ValueError("manifest scope is not a turnover-gap audit")
    artifacts = manifest.get("candidate_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("manifest contains no candidate artifacts")
    factor_ids = tuple(sorted(str(item.get("factor_id")) for item in artifacts))
    if factor_ids != tuple(sorted(EXPECTED_FACTOR_IDS)):
        raise ValueError(f"candidate scope must contain exactly six approved IDs: {factor_ids}")
    if BLOCKED_FACTOR_IDS.intersection(factor_ids):
        raise ValueError("blocked factor is present in candidate scope")
    frames: list[pd.DataFrame] = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ValueError("candidate artifact entry is not an object")
        factor_id = str(artifact["factor_id"])
        raw_path = Path(str(artifact["path"]))
        path = (
            raw_path
            if raw_path.is_file()
            else (candidate_dir or manifest_path.parent) / raw_path.name
        )
        if not path.is_file():
            raise FileNotFoundError(f"candidate artifact is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != str(artifact.get("sha256")):
            raise ValueError(f"candidate artifact hash mismatch: {factor_id}")
        frame = pd.read_csv(path)
        missing = sorted(set(_REQUIRED_COLUMNS).difference(frame.columns))
        if missing:
            raise ValueError(f"candidate {factor_id} is missing columns: {missing}")
        if len(frame) != int(artifact.get("rows", -1)):
            raise ValueError(f"candidate row count mismatch: {factor_id}")
        if set(frame["factor_id"].astype(str)) != {factor_id}:
            raise ValueError(f"candidate factor ID mismatch: {factor_id}")
        frames.append(frame.loc[:, list(_REQUIRED_COLUMNS)].copy())
    result = pd.concat(frames, ignore_index=True)
    result["observation_date"] = pd.to_datetime(
        result["observation_date"], errors="raise"
    ).dt.date
    result["ts_code"] = result["ts_code"].astype(str)
    result["factor_id"] = result["factor_id"].astype(str)
    result["factor_version"] = pd.to_numeric(
        result["factor_version"], errors="raise"
    ).astype(int)
    result["candidate_value"] = pd.to_numeric(result["candidate_value"], errors="coerce")
    result["source_value"] = pd.to_numeric(result["source_value"], errors="coerce")
    if not result["candidate_value"].notna().all():
        raise ValueError("candidate values must all be finite")
    if result["source_value"].notna().any():
        raise ValueError("candidate scope must not overwrite finite source values")
    if result["observation_date"].max() >= LOCKED_TEST_START:
        raise ValueError("candidate scope includes locked-test dates")
    if result.duplicated(list(_BUSINESS_KEYS)).any():
        raise ValueError("candidate scope contains duplicate business keys")
    if set(result["repair_method"]) != {"TUSHARE_TURNOVER_GAP"}:
        raise ValueError("candidate scope has an unexpected repair method")
    if set(result["reason"]) != {"TURNOVER_RATE_GAP"}:
        raise ValueError("candidate scope has an unexpected reason")
    return result.sort_values(list(_BUSINESS_KEYS), kind="mergesort").reset_index(drop=True)


def build_candidate_batch(frame: pd.DataFrame) -> tuple[StagedBatch, dict[str, object]]:
    if frame.empty:
        raise ValueError("candidate frame cannot be empty")
    canonical = frame.to_json(orient="records", date_format="iso", double_precision=15)
    data_version = hashlib.sha256(("turnover-gap-candidate:" + canonical).encode()).hexdigest()
    run_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "quant-research-v2/" + data_version))
    source_watermark = hashlib.sha256(("turnover-gap-source:" + canonical).encode()).hexdigest()
    created_at = datetime.now(UTC).replace(tzinfo=None)
    staged = frame.copy(deep=True)
    staged["candidate_hash"] = staged.apply(
        lambda row: hashlib.sha256(
            f"{row.observation_date}|{row.ts_code}|{row.factor_id}|{row.candidate_value:.17g}".encode()
        ).hexdigest(),
        axis=1,
    )
    staged["run_id"] = run_id
    staged["data_version"] = data_version
    staged["created_at"] = created_at
    staged["source_watermark_id"] = source_watermark
    staged["record_hash"] = staged.apply(
        lambda row: hashlib.sha256(
            "|".join(
                str(row[column])
                for column in (
                    "observation_date",
                    "ts_code",
                    "factor_id",
                    "factor_version",
                    "candidate_value",
                    "candidate_hash",
                    "data_version",
                )
            ).encode()
        ).hexdigest(),
        axis=1,
    )
    batch = StagedBatch.create(
        object_name="factor_repair_candidate",
        logical_partition="2020-09",
        data_version=data_version,
        frame=staged,
        business_keys=_BUSINESS_KEYS,
        partition_date_column="observation_date",
    )
    identity = {
        "run_id": run_id,
        "data_version": data_version,
        "source_watermark_id": source_watermark,
        "created_at": created_at.isoformat(),
        "row_count": len(staged),
        "factor_ids": sorted(set(staged["factor_id"])),
        "logical_partition": "2020-09",
    }
    return batch, identity


def preflight(frame: pd.DataFrame) -> tuple[StagedBatch, dict[str, object], list[CheckResult]]:
    batch, identity = build_candidate_batch(frame)
    batch.validate()
    checks = [
        CheckResult("candidate_scope_exact_six", Severity.BLOCKER, CheckStatus.PASS),
        CheckResult("candidate_values_finite", Severity.BLOCKER, CheckStatus.PASS),
        CheckResult("candidate_no_overwrite", Severity.BLOCKER, CheckStatus.PASS),
        CheckResult("candidate_keys_unique", Severity.BLOCKER, CheckStatus.PASS),
        CheckResult("candidate_not_locked_test", Severity.BLOCKER, CheckStatus.PASS),
    ]
    return batch, identity, checks


def _connect(target_database: str) -> V2WriteSession:
    import dolphindb as ddb  # type: ignore[import-untyped]

    client = ddb.session()
    client.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    return V2WriteSession(client, target_database=target_database)


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    frame = load_candidate_scope(args.manifest, args.candidate_dir)
    batch, identity, checks = preflight(frame)
    report: dict[str, object] = {
        "scope": "TURNOVER_GAP_CANDIDATE_PREPARE_V1",
        "database_writes": False,
        "publish_requested": bool(args.publish),
        "checks": [
            {
                "name": check.check_name,
                "severity": check.severity.value,
                "status": check.status.value,
            }
            for check in checks
        ],
        "identity": identity,
    }
    if args.publish:
        target = _connect(str(args.target_database))
        # ``preflight`` validates its own batch.  Build a fresh CREATED batch
        # for the publisher, which performs the final validation immediately
        # before upload and pointer advancement.
        publish_batch, publish_identity = build_candidate_batch(frame)
        identity_keys = ("run_id", "data_version", "source_watermark_id", "row_count")
        if any(publish_identity[key] != identity[key] for key in identity_keys):
            raise RuntimeError("candidate identity changed between preflight and publish")
        V2BatchPublisher(target).publish(publish_batch, checks)
        report["database_writes"] = True
        report["published"] = True
    else:
        report["published"] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return report


if __name__ == "__main__":
    main()
