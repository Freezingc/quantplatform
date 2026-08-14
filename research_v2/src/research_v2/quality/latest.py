"""Current-state factor-quality rows for model feature screening.

This module intentionally models only the latest quality conclusion.  It does
not create factor sets, version pointers, or historical audit records.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import date
from numbers import Integral, Real
from typing import Final

import pandas as pd

QUALITY_LATEST_COLUMNS: Final[tuple[str, ...]] = (
    "factor_id",
    "source_factor_id",
    "effective_date",
    "decision",
    "model_eligible",
    "monitoring_required",
    "reason_code",
    "market_scope",
    "label_id",
    "train_start",
    "train_end",
    "validation_start",
    "validation_end",
    "frozen_direction",
    "train_valid_days",
    "train_signed_rank_ic",
    "train_icir",
    "train_directional_share",
    "annual_gate_years",
    "train_max_similarity",
    "validation_valid_days",
    "validation_signed_rank_ic",
    "validation_icir",
    "validation_directional_share",
    "validation_max_similarity",
    "evidence_path",
    "evidence_commit",
)
FORBIDDEN_LINEAGE_COLUMNS: Final[frozenset[str]] = frozenset(
    {"run_id", "data_version", "created_at", "source_watermark_id", "record_hash"}
)
MODEL_ELIGIBLE_DECISIONS: Final[frozenset[str]] = frozenset(
    {"APPROVED", "APPROVED_MONITOR"}
)
ALPHA95_CANDIDATE_IDS: Final[tuple[str, ...]] = (
    "yeslab_163_relative_amount_volatility_10_v1",
    "yeslab_163_relative_amount_volatility_20_v1",
    "yeslab_163_relative_amount_volatility_40_v1",
)
ALPHA484_CANDIDATE_ID: Final[str] = "yeslab_484_tushare_shsz_v1"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_TRAIN_START: Final[date] = date(2020, 1, 2)
_TRAIN_END: Final[date] = date(2024, 12, 31)
_VALIDATION_START: Final[date] = date(2025, 1, 2)
_VALIDATION_END: Final[date] = date(2025, 12, 31)


def empty_quality_row(factor_id: str, effective_date: date) -> dict[str, object]:
    """Return the explicit current-state default for one catalog factor."""
    normalized_id = _normalize_factor_id(factor_id)
    if not isinstance(effective_date, date):
        raise ValueError("effective_date must be a date")
    return {
        "factor_id": normalized_id,
        "source_factor_id": normalized_id,
        "effective_date": effective_date,
        "decision": "UNSCREENED",
        "model_eligible": False,
        "monitoring_required": False,
        "reason_code": "NO_COMPLETE_STRICT_PREDICTIVE_EVIDENCE",
        "market_scope": "SHSZ",
        "label_id": "forward_open_5d:v1",
        "train_start": _TRAIN_START,
        "train_end": _TRAIN_END,
        "validation_start": _VALIDATION_START,
        "validation_end": _VALIDATION_END,
        "frozen_direction": pd.NA,
        "train_valid_days": pd.NA,
        "train_signed_rank_ic": math.nan,
        "train_icir": math.nan,
        "train_directional_share": math.nan,
        "annual_gate_years": pd.NA,
        "train_max_similarity": math.nan,
        "validation_valid_days": pd.NA,
        "validation_signed_rank_ic": math.nan,
        "validation_icir": math.nan,
        "validation_directional_share": math.nan,
        "validation_max_similarity": math.nan,
        "evidence_path": "",
        "evidence_commit": "",
    }


def build_quality_latest_frame(
    *,
    catalog: pd.DataFrame,
    scoreboard: Mapping[str, object],
    alpha95: Mapping[str, object],
    alpha484: Mapping[str, object],
    effective_date: date,
    evidence_commit: str,
) -> pd.DataFrame:
    """Build current quality rows without mutating a database or factor set."""
    if "factor_id" not in catalog.columns:
        raise ValueError("catalog is missing factor_id")
    if not _COMMIT.fullmatch(evidence_commit):
        raise ValueError("evidence_commit must be a lowercase 40-hex SHA")
    catalog_ids = tuple(_normalize_factor_id(value) for value in catalog["factor_id"])
    if not catalog_ids or len(set(catalog_ids)) != len(catalog_ids):
        raise ValueError("catalog factor IDs must be non-empty and unique")

    rows = {factor_id: empty_quality_row(factor_id, effective_date) for factor_id in catalog_ids}
    for row in _mapping_rows(scoreboard, "rows", "scoreboard"):
        factor_id = _normalize_factor_id(_required(row, "factor_id", "scoreboard row"))
        if factor_id not in rows:
            raise ValueError(f"scoreboard factor is not in the catalog: {factor_id}")
        status = str(_required(row, "status", f"scoreboard {factor_id}"))
        if status not in {"PASS", "PASS_WITH_MONITORING"}:
            raise ValueError(f"scoreboard factor has unsupported status: {factor_id}={status}")
        decision = "APPROVED_MONITOR" if status == "PASS_WITH_MONITORING" else "APPROVED"
        rows[factor_id] = _with_metrics(
            rows[factor_id],
            source_factor_id=factor_id,
            decision=decision,
            model_eligible=True,
            monitoring_required=bool(
                row.get("monitoring_required", decision == "APPROVED_MONITOR")
            ),
            reason_code=str(row.get("reason_code", "FORMAL_NINE_SCOREBOARD")),
            metrics=row,
            evidence_path=str(row.get("evidence_path", "")),
            evidence_commit=evidence_commit,
        )

    alpha95_candidates = _mapping(alpha95.get("candidates"), "alpha95 candidates")
    observed_alpha95 = set(map(str, alpha95_candidates))
    if observed_alpha95 != set(ALPHA95_CANDIDATE_IDS):
        raise ValueError("alpha95 payload must contain exactly the three canonical candidates")
    for factor_id in ALPHA95_CANDIDATE_IDS:
        candidate = _mapping(alpha95_candidates[factor_id], f"alpha95 candidate {factor_id}")
        training = _mapping(candidate.get("train_2020_2024"), f"alpha95 training {factor_id}")
        validation = _mapping(
            candidate.get("validation_2025", {}), f"alpha95 validation {factor_id}"
        )
        row = empty_quality_row(factor_id, effective_date)
        rows[factor_id] = _with_metrics(
            row,
            source_factor_id="yeslab_163",
            decision="REJECT",
            model_eligible=False,
            monitoring_required=False,
            reason_code="ALPHA95_REJECT_TRAINING_GATE",
            metrics={**training, **_validation_prefixed(validation)},
            evidence_path=str(alpha95.get("evidence_path", "")),
            evidence_commit=evidence_commit,
        )

    candidate_id = str(alpha484.get("candidate_id", ALPHA484_CANDIDATE_ID))
    if candidate_id != ALPHA484_CANDIDATE_ID:
        raise ValueError("alpha484 candidate identity does not match the canonical contract")
    review_status = str(_required(alpha484, "status", "alpha484 review"))
    if review_status == "READY_FOR_EXPLICIT_ADMISSION_REVIEW":
        decision = "PENDING_EXPLICIT_ADMISSION"
    elif review_status.startswith("REJECT"):
        decision = "REJECT"
    else:
        decision = "DEFER"
    training = _mapping(alpha484.get("training"), "alpha484 training")
    validation = _mapping(alpha484.get("validation"), "alpha484 validation")
    alpha484_metrics = {
        **training,
        **_validation_prefixed(validation),
        "train_max_similarity": _value(
            _mapping(alpha484.get("train_similarity"), "alpha484 train similarity"),
            "maximum_median_abs_spearman",
        ),
        "validation_max_similarity": _value(
            _mapping(alpha484.get("validation_similarity"), "alpha484 validation similarity"),
            "maximum_median_abs_spearman",
        ),
    }
    row = empty_quality_row(candidate_id, effective_date)
    rows[candidate_id] = _with_metrics(
        row,
        source_factor_id=str(alpha484.get("source_factor_id", "yeslab_484")),
        decision=decision,
        model_eligible=False,
        monitoring_required=False,
        reason_code=review_status,
        metrics=alpha484_metrics,
        evidence_path=str(alpha484.get("evidence_path", "")),
        evidence_commit=evidence_commit,
    )
    return pd.DataFrame(list(rows.values()), columns=QUALITY_LATEST_COLUMNS).sort_values(
        "factor_id", kind="mergesort"
    ).reset_index(drop=True)


def validate_quality_latest_frame(
    frame: pd.DataFrame, *, expected_catalog_count: int
) -> pd.DataFrame:
    """Fail closed on a malformed current-state quality replacement frame."""
    forbidden = sorted(FORBIDDEN_LINEAGE_COLUMNS.intersection(frame.columns))
    if forbidden:
        raise ValueError(f"quality latest frame has forbidden columns: {forbidden}")
    if tuple(frame.columns) != QUALITY_LATEST_COLUMNS:
        raise ValueError("quality latest frame columns do not match the compact contract")
    if expected_catalog_count < 1:
        raise ValueError("expected_catalog_count must be positive")
    expected_count = expected_catalog_count + len(ALPHA95_CANDIDATE_IDS) + 1
    if len(frame) != expected_count:
        raise ValueError(f"quality latest row count must be {expected_count}")
    if frame["factor_id"].isna().any() or frame["factor_id"].duplicated().any():
        raise ValueError("quality latest factor IDs must be non-empty and unique")
    eligible = frame.loc[frame["model_eligible"].fillna(False).astype(bool)].copy()
    if len(eligible) != 9:
        raise ValueError("quality latest frame must contain exactly nine eligible factors")
    if not eligible["decision"].isin(MODEL_ELIGIBLE_DECISIONS).all():
        raise ValueError("eligible factors must have an approved decision")
    if not eligible["market_scope"].eq("SHSZ").all() or not eligible["label_id"].eq(
        "forward_open_5d:v1"
    ).all():
        raise ValueError("eligible factors must use the canonical SHSZ five-day label scope")
    noneligible = frame.loc[~frame["model_eligible"].fillna(False).astype(bool)]
    if noneligible["decision"].isin(MODEL_ELIGIBLE_DECISIONS).any():
        raise ValueError("approved decisions must be model eligible")
    _validate_required_metrics(frame)
    return frame.copy(deep=True)


def select_model_eligible(frame: pd.DataFrame) -> pd.DataFrame:
    """Return typed, factor-ID-sorted eligible rows without constructing a factor set."""
    forbidden = sorted(FORBIDDEN_LINEAGE_COLUMNS.intersection(frame.columns))
    if forbidden:
        raise ValueError(f"quality latest frame has forbidden columns: {forbidden}")
    if tuple(frame.columns) != QUALITY_LATEST_COLUMNS:
        raise ValueError("quality latest frame columns do not match the compact contract")
    if frame["factor_id"].duplicated().any():
        raise ValueError("quality latest factor IDs must be unique")
    eligible = frame.loc[frame["model_eligible"].fillna(False).astype(bool)].copy()
    if not eligible["decision"].isin(MODEL_ELIGIBLE_DECISIONS).all():
        raise ValueError("eligible factors must have an approved decision")
    if not eligible["market_scope"].eq("SHSZ").all() or not eligible["label_id"].eq(
        "forward_open_5d:v1"
    ).all():
        raise ValueError("eligible factors must use the canonical SHSZ five-day label scope")
    return eligible.sort_values("factor_id", kind="mergesort").reset_index(drop=True)


def _with_metrics(
    row: Mapping[str, object],
    *,
    source_factor_id: str,
    decision: str,
    model_eligible: bool,
    monitoring_required: bool,
    reason_code: str,
    metrics: Mapping[str, object],
    evidence_path: str,
    evidence_commit: str,
) -> dict[str, object]:
    result = dict(row)
    result.update(
        {
            "source_factor_id": _normalize_factor_id(source_factor_id),
            "decision": decision,
            "model_eligible": model_eligible,
            "monitoring_required": monitoring_required,
            "reason_code": reason_code,
            "frozen_direction": _value(metrics, "frozen_direction", "direction"),
            "train_valid_days": _value(metrics, "train_valid_days", "valid_day_count"),
            "train_signed_rank_ic": _value(
                metrics, "train_signed_rank_ic", "signed_mean_rank_ic"
            ),
            "train_icir": _value(metrics, "train_icir", "signed_icir"),
            "train_directional_share": _value(
                metrics, "train_directional_share", "directional_day_share"
            ),
            "annual_gate_years": _value(
                metrics, "annual_gate_years", "annual_double_gate_pass_years"
            ),
            "train_max_similarity": _value(metrics, "train_max_similarity", "max_similarity"),
            "validation_valid_days": _value(metrics, "validation_valid_days"),
            "validation_signed_rank_ic": _value(
                metrics, "validation_signed_rank_ic"
            ),
            "validation_icir": _value(metrics, "validation_icir"),
            "validation_directional_share": _value(
                metrics, "validation_directional_share"
            ),
            "validation_max_similarity": _value(metrics, "validation_max_similarity"),
            "evidence_path": evidence_path,
            "evidence_commit": evidence_commit,
        }
    )
    return result


def _validation_prefixed(metrics: Mapping[str, object]) -> dict[str, object]:
    return {
        "validation_valid_days": _value(metrics, "valid_day_count"),
        "validation_signed_rank_ic": _value(metrics, "signed_mean_rank_ic"),
        "validation_icir": _value(metrics, "signed_icir"),
        "validation_directional_share": _value(metrics, "directional_day_share"),
        "validation_max_similarity": _value(metrics, "max_similarity"),
    }


def _validate_required_metrics(frame: pd.DataFrame) -> None:
    train_columns = (
        "frozen_direction",
        "train_valid_days",
        "train_signed_rank_ic",
        "train_icir",
        "train_directional_share",
        "annual_gate_years",
        "train_max_similarity",
    )
    validation_columns = (
        "validation_valid_days",
        "validation_signed_rank_ic",
        "validation_icir",
        "validation_directional_share",
        "validation_max_similarity",
    )
    evaluated = frame["decision"].isin(
        {"APPROVED", "APPROVED_MONITOR", "PENDING_EXPLICIT_ADMISSION", "REJECT", "DEFER"}
    )
    for column in train_columns:
        if not _all_finite(frame.loc[evaluated, column]):
            raise ValueError(f"evaluated factors require finite training metric: {column}")
    complete_validation = frame["decision"].isin(
        {"APPROVED", "APPROVED_MONITOR", "PENDING_EXPLICIT_ADMISSION", "DEFER"}
    )
    for column in validation_columns:
        if not _all_finite(frame.loc[complete_validation, column]):
            raise ValueError(f"completed reviews require finite validation metric: {column}")


def _all_finite(values: pd.Series) -> bool:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric.map(lambda value: math.isfinite(float(value)))
    return bool(numeric.notna().all() and finite.all())


def _mapping_rows(
    source: Mapping[str, object], key: str, label: str
) -> tuple[Mapping[str, object], ...]:
    values = source.get(key)
    if not isinstance(values, list):
        raise ValueError(f"{label} is missing list field: {key}")
    return tuple(_mapping(value, f"{label} {key}") for value in values)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _required(source: Mapping[str, object], key: str, label: str) -> object:
    if key not in source:
        raise ValueError(f"{label} is missing {key}")
    return source[key]


def _value(source: Mapping[str, object], *keys: str) -> object:
    for key in keys:
        if key in source:
            return source[key]
    return math.nan


def _normalize_factor_id(value: object) -> str:
    if isinstance(value, bool):
        raise ValueError("factor ID cannot be boolean")
    if isinstance(value, Integral):
        return f"yeslab_{int(value)}"
    if isinstance(value, Real) and float(value).is_integer():
        return f"yeslab_{int(value)}"
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("factor ID cannot be empty")
    if normalized.isdigit():
        return f"yeslab_{int(normalized)}"
    return normalized
