"""Dry-run-first publication for the compact current factor-quality registry."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Protocol

import pandas as pd

from research_v2.db.session import V2WriteSession
from research_v2.quality.latest import (
    build_quality_latest_frame,
    select_model_eligible,
    validate_quality_latest_frame,
)

_TARGET_OBJECT = "factor_quality_latest"
_STAGE_NAME = "factorQualityLatestStage"
_DEFAULT_CATALOG_PATH = Path("research_v2/configs/factors/source_catalog.yaml")
_DEFAULT_SCOREBOARD_PATH = Path(
    "research_v2/artifacts/diagnostics/tushare_shsz_factor_quality_scoreboard_v4.json"
)
_DEFAULT_TRAIN_SUMMARY_PATH = Path(
    "research_v2/artifacts/diagnostics/"
    "tushare_canonical_formal9_unique7_train_2020_2024_v1/quality_summary.json"
)
_DEFAULT_VALIDATION_SUMMARY_PATH = Path(
    "research_v2/artifacts/diagnostics/"
    "tushare_canonical_formal9_unique7_validation_2025_v1/quality_summary.json"
)
_DEFAULT_TRAIN_SIMILARITY_PATH = Path(
    "research_v2/artifacts/diagnostics/"
    "tushare_canonical_formal9_unique7_train_2020_2024_v1/similarity_summary.json"
)
_DEFAULT_VALIDATION_SIMILARITY_PATH = Path(
    "research_v2/artifacts/diagnostics/"
    "tushare_canonical_formal9_unique7_validation_2025_v1/similarity_summary.json"
)
_DEFAULT_ALPHA95_PATH = Path(
    "research_v2/artifacts/diagnostics/"
    "tushare_alpha95_relative_amount_volatility_v1/quality_summary.json"
)


class QualityLatestSession(Protocol):
    """Minimal V2 session capability needed to replace the compact registry."""

    @property
    def target_database(self) -> str: ...

    def upload_frame(self, name: str, frame: pd.DataFrame) -> None: ...

    def execute(self, script: str) -> object: ...

    def query(self, script: str) -> object: ...


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit dry-run, publication, or read-only eligibility actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effective-date", type=date.fromisoformat)
    parser.add_argument("--evidence-commit")
    parser.add_argument("--alpha484-review", type=Path)
    parser.add_argument("--catalog", type=Path, default=_DEFAULT_CATALOG_PATH)
    parser.add_argument("--scoreboard", type=Path, default=_DEFAULT_SCOREBOARD_PATH)
    parser.add_argument("--train-summary", type=Path, default=_DEFAULT_TRAIN_SUMMARY_PATH)
    parser.add_argument(
        "--validation-summary", type=Path, default=_DEFAULT_VALIDATION_SUMMARY_PATH
    )
    parser.add_argument(
        "--train-similarity", type=Path, default=_DEFAULT_TRAIN_SIMILARITY_PATH
    )
    parser.add_argument(
        "--validation-similarity", type=Path, default=_DEFAULT_VALIDATION_SIMILARITY_PATH
    )
    parser.add_argument("--alpha95", type=Path, default=_DEFAULT_ALPHA95_PATH)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--publish", action="store_true")
    mode.add_argument("--model-eligible", action="store_true")
    args = parser.parse_args(argv)
    if not args.model_eligible:
        missing = [
            flag
            for flag, value in (
                ("--effective-date", args.effective_date),
                ("--evidence-commit", args.evidence_commit),
                ("--alpha484-review", args.alpha484_review),
            )
            if value is None
        ]
        if missing:
            parser.error(f"normal refresh requires: {', '.join(missing)}")
    return args


def load_formal_scoreboard(
    *,
    scoreboard_path: Path,
    train_summary_path: Path,
    validation_summary_path: Path,
    train_similarity_path: Path,
    validation_similarity_path: Path,
) -> dict[str, object]:
    """Normalize audited formal-nine artifacts into the row-builder contract."""
    scoreboard = _read_mapping_json(scoreboard_path, "formal scoreboard")
    factor_ids = tuple(str(value) for value in _required_sequence(scoreboard, "factor_ids"))
    if not factor_ids or len(set(factor_ids)) != len(factor_ids):
        raise ValueError("formal scoreboard factor_ids must be non-empty and unique")
    monitoring_ids = {
        str(value) for value in _required_sequence(scoreboard, "monitoring_factor_ids")
    }
    if not monitoring_ids.issubset(factor_ids):
        raise ValueError("formal scoreboard monitoring IDs are not a factor-ID subset")
    score_rows = _index_rows(_required_sequence(scoreboard, "rows"), "formal scoreboard")
    if set(score_rows) != set(factor_ids):
        raise ValueError("formal scoreboard rows do not match its factor IDs")
    train_rows = _index_rows(
        _required_sequence(_read_mapping_json(train_summary_path, "train summary"), "summaries"),
        "train summary",
    )
    validation_rows = _index_rows(
        _required_sequence(
            _read_mapping_json(validation_summary_path, "validation summary"), "summaries"
        ),
        "validation summary",
    )
    missing_train = sorted(set(factor_ids).difference(train_rows))
    missing_validation = sorted(set(factor_ids).difference(validation_rows))
    if missing_train or missing_validation:
        raise ValueError(
            "formal summaries are incomplete: "
            f"train={missing_train} validation={missing_validation}"
        )
    train_pairs = _read_json_array(train_similarity_path, "train similarity")
    validation_pairs = _read_json_array(validation_similarity_path, "validation similarity")
    rows: list[dict[str, object]] = []
    for factor_id in factor_ids:
        score = score_rows[factor_id]
        train = train_rows[factor_id]
        validation = validation_rows[factor_id]
        direction = _integer(train, "direction")
        if direction not in {-1, 1}:
            raise ValueError(f"formal factor direction must be -1 or 1: {factor_id}")
        admission_status = str(_required(score, "admission_status", f"score {factor_id}"))
        if admission_status not in {"PASS", "PASS_WITH_MONITORING"}:
            raise ValueError(f"formal factor has unsupported admission status: {factor_id}")
        rows.append(
            {
                "factor_id": factor_id,
                "status": admission_status,
                "monitoring_required": factor_id in monitoring_ids,
                "reason_code": "FORMAL_NINE_SCOREBOARD",
                "frozen_direction": direction,
                "train_valid_days": _integer(train, "valid_day_count"),
                "train_signed_rank_ic": _number(train, "signed_mean_rank_ic"),
                "train_icir": _signed_icir(train, direction),
                "train_directional_share": _number(train, "directional_day_share"),
                "annual_gate_years": _integer(score, "annual_abs_icir_pass_years"),
                "train_max_similarity": _maximum_similarity(
                    factor_id, factor_ids, train_pairs
                ),
                "validation_valid_days": _integer(validation, "valid_day_count"),
                "validation_signed_rank_ic": _number(
                    validation, "signed_mean_rank_ic"
                ),
                "validation_icir": _signed_icir(validation, direction),
                "validation_directional_share": _number(
                    validation, "directional_day_share"
                ),
                "validation_max_similarity": _maximum_similarity(
                    factor_id, factor_ids, validation_pairs
                ),
                "evidence_path": str(scoreboard_path),
            }
        )
    return {"rows": rows}


def build_frame_from_artifacts(
    *,
    catalog_path: Path,
    scoreboard_path: Path,
    train_summary_path: Path,
    validation_summary_path: Path,
    train_similarity_path: Path,
    validation_similarity_path: Path,
    alpha95_path: Path,
    alpha484_path: Path,
    effective_date: date,
    evidence_commit: str,
) -> pd.DataFrame:
    """Build a validated-input frame from local catalog and audit artifacts only."""
    catalog = _read_catalog(catalog_path)
    scoreboard = load_formal_scoreboard(
        scoreboard_path=scoreboard_path,
        train_summary_path=train_summary_path,
        validation_summary_path=validation_summary_path,
        train_similarity_path=train_similarity_path,
        validation_similarity_path=validation_similarity_path,
    )
    return build_quality_latest_frame(
        catalog=catalog,
        scoreboard=scoreboard,
        alpha95=_read_mapping_json(alpha95_path, "alpha95 quality summary"),
        alpha484=_read_mapping_json(alpha484_path, "alpha484 admission review"),
        effective_date=effective_date,
        evidence_commit=evidence_commit,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Build a dry-run, publish a validated replacement, or read eligible rows."""
    args = parse_args(argv)
    if args.model_eligible:
        client, session = _connect_v2()
        try:
            result = session.query(
                "select * from loadTable("
                f"'{session.target_database}', '{_TARGET_OBJECT}')"
            )
            if not isinstance(result, pd.DataFrame):
                raise RuntimeError("quality latest reader did not return a DataFrame")
            eligible = select_model_eligible(result)
        finally:
            client.close()
        print(eligible.to_json(orient="records", date_format="iso"))
        return 0

    frame = build_frame_from_artifacts(
        catalog_path=args.catalog,
        scoreboard_path=args.scoreboard,
        train_summary_path=args.train_summary,
        validation_summary_path=args.validation_summary,
        train_similarity_path=args.train_similarity,
        validation_similarity_path=args.validation_similarity,
        alpha95_path=args.alpha95,
        alpha484_path=args.alpha484_review,
        effective_date=args.effective_date,
        evidence_commit=args.evidence_commit,
    )
    if args.publish:
        client, session = _connect_v2()
        try:
            status = refresh_latest(session, frame, publish=True)
        finally:
            client.close()
    else:
        status = refresh_latest(None, frame, publish=False)  # type: ignore[arg-type]
    print(
        json.dumps(
            {
                "status": status,
                "row_count": int(len(frame)),
                "eligible_count": int(frame["model_eligible"].sum()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _connect_v2() -> tuple[object, V2WriteSession]:
    import dolphindb as ddb  # type: ignore[import-untyped]

    client = ddb.session()
    client.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    return client, V2WriteSession(client, target_database="dfs://QuantResearchV2Test")


def refresh_latest(
    session: QualityLatestSession, frame: pd.DataFrame, *, publish: bool
) -> str:
    """Validate then optionally replace only ``factor_quality_latest`` rows."""
    validated = validate_quality_latest_frame(frame, expected_catalog_count=487)
    if not publish:
        return "dry-run"
    session.upload_frame(_STAGE_NAME, validated)
    target = f"loadTable('{session.target_database}', '{_TARGET_OBJECT}')"
    session.execute(f"delete from {target}")
    session.execute(f"tableInsert{{{target}}}({_STAGE_NAME})")
    session.execute(f"undef(`{_STAGE_NAME})")
    _verify_published(session)
    return "published"


def _verify_published(session: QualityLatestSession) -> None:
    result = session.query(
        "select count(*) as row_count, count(distinct factor_id) as unique_factor_count, "
        "sum(int(model_eligible)) as eligible_count from loadTable("
        f"'{session.target_database}', '{_TARGET_OBJECT}')"
    )
    required = {"row_count", "unique_factor_count", "eligible_count"}
    if not isinstance(result, pd.DataFrame) or len(result) != 1 or not required.issubset(
        result.columns
    ):
        raise RuntimeError("quality latest post-write check returned an invalid frame")
    row = result.iloc[0]
    observed = tuple(
        int(row[column])
        for column in ("row_count", "unique_factor_count", "eligible_count")
    )
    if observed != (491, 491, 9):
        raise RuntimeError(
            "quality latest post-write invariant failed: "
            f"rows={observed[0]} unique={observed[1]} eligible={observed[2]}"
        )


def _read_mapping_json(path: Path, label: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON for {label}: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _read_catalog(path: Path) -> pd.DataFrame:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - declared project dependency
        raise RuntimeError("PyYAML is required to read the factor catalog") from error
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read factor catalog: {path}") from error
    records = payload.get("factors") if isinstance(payload, Mapping) else payload
    if not isinstance(records, list) or not all(isinstance(item, Mapping) for item in records):
        raise ValueError("factor catalog must contain a factors object array")
    catalog = pd.DataFrame(records)
    if "factor_id" not in catalog.columns:
        raise ValueError("factor catalog is missing factor_id")
    return catalog.loc[:, ["factor_id"]].copy()


def _read_json_array(path: Path, label: str) -> tuple[Mapping[str, object], ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON for {label}: {path}") from error
    if not isinstance(payload, list) or not all(isinstance(row, Mapping) for row in payload):
        raise ValueError(f"{label} must be a JSON object array: {path}")
    return tuple(payload)


def _required(source: Mapping[str, object], key: str, label: str) -> object:
    if key not in source:
        raise ValueError(f"{label} is missing {key}")
    return source[key]


def _required_sequence(source: Mapping[str, object], key: str) -> Sequence[object]:
    value = _required(source, key, "JSON object")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"JSON object field {key} must be a sequence")
    return value


def _index_rows(
    values: Sequence[object], label: str
) -> dict[str, Mapping[str, object]]:
    rows: dict[str, Mapping[str, object]] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} rows must be objects")
        factor_id = str(_required(value, "factor_id", label))
        if not factor_id or factor_id in rows:
            raise ValueError(f"{label} has missing or duplicate factor_id")
        rows[factor_id] = value
    return rows


def _number(source: Mapping[str, object], key: str) -> float:
    value = _required(source, key, "metric row")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"metric {key} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"metric {key} must be finite")
    return number


def _integer(source: Mapping[str, object], key: str) -> int:
    value = _number(source, key)
    if not value.is_integer():
        raise ValueError(f"metric {key} must be an integer")
    return int(value)


def _signed_icir(source: Mapping[str, object], direction: int) -> float:
    if "signed_icir" in source:
        return _number(source, "signed_icir")
    return _number(source, "icir") * direction


def _maximum_similarity(
    factor_id: str,
    factor_ids: Sequence[str],
    pairs: Sequence[Mapping[str, object]],
) -> float:
    allowed = set(factor_ids)
    values: list[float] = []
    for pair in pairs:
        left = str(_required(pair, "factor_id_left", "similarity row"))
        right = str(_required(pair, "factor_id_right", "similarity row"))
        if factor_id not in {left, right} or left not in allowed or right not in allowed:
            continue
        values.append(_number(pair, "median_abs_spearman"))
    return max(values, default=0.0)


if __name__ == "__main__":
    raise SystemExit(main())
