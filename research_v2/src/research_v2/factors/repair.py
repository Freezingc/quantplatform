"""Write-free market-gap repair and exact GP factor reconstruction helpers.

The repair path is deliberately narrow.  It preserves every primary market
row, adds only missing business keys from a secondary extract, reconstructs
only the two approved GP expressions, and fills only non-finite source factor
values.  It never overwrites an existing finite source observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Final

import numpy as np
import pandas as pd


class FactorRepairError(ValueError):
    """Raised when a repair candidate violates its source or key contract."""


_MARKET_COLUMNS: Final[tuple[str, ...]] = (
    "trade_date",
    "ts_code",
    "close",
    "amount",
    "pct_chg",
)
_FACTOR_COLUMNS: Final[tuple[str, ...]] = (
    "trade_date",
    "ts_code",
    "factor_name",
    "value",
)
_CANDIDATE_COLUMNS: Final[tuple[str, ...]] = (
    "trade_date",
    "ts_code",
    "candidate_value",
)
_BUSINESS_KEYS: Final[tuple[str, ...]] = ("trade_date", "ts_code")
_SOURCE_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "trade_date",
    "ts_code",
    "factor_name",
)


class RepairStatus(StrEnum):
    """Decision states emitted by the pure repair overlay contract."""

    PASS = "REPAIR_AUDIT_PASS"
    PASS_WITH_QUARANTINE = "REPAIR_AUDIT_PASS_WITH_UNRESOLVED_QUARANTINE"
    BLOCKED = "REPAIR_BLOCKED"
    PENDING_RECHECK = "REPAIRED_PENDING_QUALITY_RECHECK"


@dataclass(frozen=True)
class FactorRepairAudit:
    """Finite-value and overlap evidence for one source/candidate pair."""

    source_rows: int
    source_finite_rows: int
    candidate_finite_rows: int
    overlap_finite_rows: int
    fillable_rows: int
    remaining_nonfinite_rows: int
    mismatch_rows: int
    max_abs_diff: float
    max_rel_diff: float


@dataclass(frozen=True)
class SourceKeyAudit:
    """Deterministic duplicate and finite-value counts for a factor source."""

    source_rows: int
    exact_duplicate_groups: int
    exact_duplicate_rows: int
    conflict_groups: int
    nonfinite_rows: int
    finite_rows: int


@dataclass(frozen=True)
class RepairOverlayResult:
    """Clean and quarantined rows produced by :func:`apply_repair_overlay`."""

    status: RepairStatus
    clean: pd.DataFrame
    quarantine: pd.DataFrame
    source_audit: SourceKeyAudit
    candidate_audit: SourceKeyAudit | None
    filled_rows: int
    added_rows: int
    conflict_groups: int
    exact_duplicate_groups: int
    pit_violation_rows: int
    unresolved_rows: int
    mismatch_rows: int
    max_abs_diff: float
    max_rel_diff: float

    @property
    def audit(self) -> SourceKeyAudit:
        """Compatibility alias for callers that refer to the source audit."""

        return self.source_audit

    @property
    def source_key_audit(self) -> SourceKeyAudit:
        """Explicit alias used by audit/persistence callers."""

        return self.source_audit

    @property
    def candidate_key_audit(self) -> SourceKeyAudit | None:
        """Explicit alias for candidate key evidence."""

        return self.candidate_audit


def audit_factor_source(source: pd.DataFrame) -> SourceKeyAudit:
    """Audit factor source keys without selecting a conflicting value.

    The source business key is ``(trade_date, ts_code, factor_name)``. Exact
    duplicate rows are safe to collapse. A key with two unequal finite values
    is a conflict and cannot be represented by one clean observation.
    """

    normalized = _normalize_overlay_source(source)
    return _audit_source_frame(normalized)


def apply_repair_overlay(
    source: pd.DataFrame,
    candidate: pd.DataFrame | None,
    *,
    factor_name: str,
    signal_date: date | None = None,
    abs_tolerance: float = 1e-8,
    rel_tolerance: float = 1e-6,
) -> RepairOverlayResult:
    """Apply a no-overwrite, point-in-time-safe candidate overlay.

    Exact duplicate source and candidate rows are collapsed deterministically.
    Any distinct finite duplicate values, overlap mismatches, or post-signal
    candidates are quarantined and block the repair. A finite source value is
    always retained; only a unique finite candidate can replace a non-finite
    source value.
    """

    if not factor_name.strip():
        raise FactorRepairError("factor_name cannot be empty")
    if abs_tolerance < 0 or rel_tolerance < 0:
        raise FactorRepairError("repair tolerances cannot be negative")
    normalized_signal_date = (
        pd.Timestamp(signal_date).date() if signal_date is not None else None
    )

    source_normalized = _normalize_overlay_source(source)
    if not (source_normalized["factor_name"] == factor_name).all():
        raise FactorRepairError("source contains an unexpected factor name")
    source_audit = _audit_source_frame(source_normalized)

    candidate_normalized: pd.DataFrame | None
    candidate_audit: SourceKeyAudit | None
    if candidate is None:
        candidate_normalized = None
        candidate_audit = None
    else:
        candidate_normalized = _normalize_overlay_candidate(candidate)
        candidate_audit = _audit_candidate_frame(candidate_normalized, factor_name)

    quarantine_rows: list[dict[str, object]] = []
    source_representatives: dict[tuple[date, str], pd.Series] = {}
    source_conflict_keys: set[tuple[date, str]] = set()

    for key, group in source_normalized.groupby(
        list(_BUSINESS_KEYS), sort=False, dropna=False
    ):
        business_key = _as_business_key(key)
        finite = np.isfinite(group["value"].to_numpy(dtype=float))
        finite_values = group.loc[finite, "value"].to_numpy(dtype=float)
        if _has_distinct_finite_values(finite_values):
            source_conflict_keys.add(business_key)
            for _, row in group.iterrows():
                quarantine_rows.append(_source_quarantine_row(row, "CONFLICT_QUARANTINE"))
            continue
        finite_rows = group.loc[finite]
        source_representatives[business_key] = (
            finite_rows.iloc[0] if not finite_rows.empty else group.iloc[0]
        )

    candidate_rows_by_key: dict[tuple[date, str], pd.Series] = {}
    candidate_conflict_keys: set[tuple[date, str]] = set()
    candidate_nonfinite_keys: set[tuple[date, str]] = set()
    pit_violation_rows = 0
    if candidate_normalized is not None:
        admissible_candidate = candidate_normalized
        if normalized_signal_date is not None:
            pit_mask = candidate_normalized["trade_date"] > normalized_signal_date
            pit_rows = candidate_normalized.loc[pit_mask]
            pit_violation_rows = len(pit_rows)
            for _, row in pit_rows.iterrows():
                quarantine_rows.append(
                    _candidate_quarantine_row(
                        row,
                        "PIT_DATE_VIOLATION",
                        factor_name=factor_name,
                    )
                )
            admissible_candidate = candidate_normalized.loc[~pit_mask]

        for key, group in admissible_candidate.groupby(
            list(_BUSINESS_KEYS), sort=False, dropna=False
        ):
            business_key = _as_business_key(key)
            finite = np.isfinite(group["candidate_value"].to_numpy(dtype=float))
            finite_values = group.loc[finite, "candidate_value"].to_numpy(dtype=float)
            nonfinite_rows = group.loc[~finite]
            if not nonfinite_rows.empty:
                candidate_nonfinite_keys.add(business_key)
                candidate_conflict_keys.add(business_key)
                for _, row in nonfinite_rows.iterrows():
                    quarantine_rows.append(
                        _candidate_quarantine_row(
                            row,
                            "NONFINITE_CANDIDATE",
                            factor_name=factor_name,
                        )
                    )
                for _, row in group.loc[finite].iterrows():
                    quarantine_rows.append(
                        _candidate_quarantine_row(
                            row,
                            "CONFLICT_QUARANTINE",
                            factor_name=factor_name,
                        )
                    )
                continue
            if _has_distinct_finite_values(finite_values):
                candidate_conflict_keys.add(business_key)
                for _, row in group.iterrows():
                    quarantine_rows.append(
                        _candidate_quarantine_row(
                            row,
                            "CONFLICT_QUARANTINE",
                            factor_name=factor_name,
                        )
                    )
                continue
            finite_rows = group.loc[finite]
            if not finite_rows.empty:
                candidate_rows_by_key[business_key] = finite_rows.iloc[0]

    clean_rows: list[pd.Series] = []
    filled_rows = 0
    added_rows = 0
    unresolved_rows = 0
    mismatch_keys: set[tuple[date, str]] = set()
    mismatch_abs: list[float] = []
    mismatch_rel: list[float] = []
    for business_key, source_row in source_representatives.items():
        if (
            business_key in candidate_conflict_keys
            and business_key not in candidate_nonfinite_keys
        ):
            # Candidate rows are already quarantined above. Keep the source
            # out of clean so a conflicting candidate can never be selected.
            quarantine_rows.append(_source_quarantine_row(source_row, "CONFLICT_QUARANTINE"))
            continue

        source_value = float(source_row["value"])
        candidate_row = candidate_rows_by_key.get(business_key)
        candidate_value = (
            float(candidate_row["candidate_value"])
            if candidate_row is not None
            else float("nan")
        )
        source_finite = bool(np.isfinite(source_value))
        candidate_finite = bool(np.isfinite(candidate_value))

        if source_finite:
            clean_rows.append(source_row)
            if candidate_finite:
                abs_diff = abs(source_value - candidate_value)
                rel_diff = abs_diff / max(abs(source_value), 1e-12)
                mismatch_abs.append(abs_diff)
                mismatch_rel.append(rel_diff)
                if abs_diff > abs_tolerance and rel_diff > rel_tolerance:
                    mismatch_keys.add(business_key)
                    quarantine_row = _candidate_quarantine_row(
                        candidate_row,  # type: ignore[arg-type]
                        "CONFLICT_QUARANTINE",
                        factor_name=factor_name,
                        source_value=source_value,
                    )
                    quarantine_rows.append(quarantine_row)
            continue

        if candidate_finite:
            repaired = source_row.copy()
            repaired["value"] = candidate_value
            clean_rows.append(repaired)
            filled_rows += 1
            continue

        # A rejected future candidate is represented by its PIT quarantine;
        # the source key still has no admissible finite evidence and remains
        # visible in the unresolved quarantine as well.
        unresolved_rows += 1
        quarantine_rows.append(_source_quarantine_row(source_row, "NO_FINITE_CANDIDATE"))

    source_keys = set(source_representatives) | source_conflict_keys
    for business_key, candidate_row in candidate_rows_by_key.items():
        if business_key in source_keys or business_key in candidate_conflict_keys:
            continue
        clean_rows.append(
            pd.Series(
                {
                    "trade_date": business_key[0],
                    "ts_code": business_key[1],
                    "factor_name": factor_name,
                    "value": candidate_row["candidate_value"],
                }
            )
        )
        added_rows += 1

    clean = _clean_overlay_frame(clean_rows)
    quarantine = _quarantine_frame(quarantine_rows)
    conflict_keys = source_conflict_keys | candidate_conflict_keys | mismatch_keys
    blocked = bool(conflict_keys or pit_violation_rows)
    if blocked:
        status = RepairStatus.BLOCKED
    elif unresolved_rows or not quarantine.empty:
        status = RepairStatus.PASS_WITH_QUARANTINE
    elif filled_rows or added_rows:
        status = RepairStatus.PENDING_RECHECK
    else:
        status = RepairStatus.PASS

    return RepairOverlayResult(
        status=status,
        clean=clean,
        quarantine=quarantine,
        source_audit=source_audit,
        candidate_audit=candidate_audit,
        filled_rows=filled_rows,
        added_rows=added_rows,
        conflict_groups=len(conflict_keys),
        exact_duplicate_groups=source_audit.exact_duplicate_groups,
        pit_violation_rows=pit_violation_rows,
        unresolved_rows=unresolved_rows,
        mismatch_rows=len(mismatch_keys),
        max_abs_diff=max(mismatch_abs, default=0.0),
        max_rel_diff=max(mismatch_rel, default=0.0),
    )


def _normalize_overlay_source(source: pd.DataFrame) -> pd.DataFrame:
    """Normalize a source frame for audit/overlay use."""

    required_keys = ["trade_date", "ts_code", "factor_name"]
    if source.loc[:, required_keys].isna().any().any():
        raise FactorRepairError("factor source business keys cannot be null")
    normalized = _normalize_factor_source(source)
    if normalized.loc[:, list(_SOURCE_KEY_COLUMNS)].isna().any().any():
        raise FactorRepairError("factor source business keys cannot be null")
    normalized["factor_name"] = normalized["factor_name"].astype(str)
    normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")
    return normalized


def _normalize_overlay_candidate(candidate: pd.DataFrame) -> pd.DataFrame:
    """Normalize candidates while retaining duplicate rows for classification."""

    missing = sorted(set(_CANDIDATE_COLUMNS).difference(candidate.columns))
    if missing:
        raise FactorRepairError(f"factor candidate is missing columns: {missing}")
    if candidate.loc[:, list(_BUSINESS_KEYS)].isna().any().any():
        raise FactorRepairError("factor candidate business keys cannot be null")
    normalized = candidate.loc[:, list(_CANDIDATE_COLUMNS)].copy(deep=True)
    normalized["trade_date"] = pd.to_datetime(
        normalized["trade_date"], errors="raise"
    ).dt.date
    normalized["ts_code"] = normalized["ts_code"].astype(str)
    if normalized.loc[:, list(_BUSINESS_KEYS)].isna().any().any():
        raise FactorRepairError("factor candidate business keys cannot be null")
    normalized["candidate_value"] = pd.to_numeric(
        normalized["candidate_value"], errors="coerce"
    )
    return normalized


def _audit_source_frame(frame: pd.DataFrame) -> SourceKeyAudit:
    """Compute key counts for a normalized source frame."""

    duplicate = frame.duplicated(list(_SOURCE_KEY_COLUMNS), keep=False)
    exact_duplicate_groups = 0
    exact_duplicate_rows = 0
    conflict_groups = 0
    for _, group in frame.loc[duplicate].groupby(
        list(_SOURCE_KEY_COLUMNS), sort=False, dropna=False
    ):
        finite_values = group.loc[
            np.isfinite(group["value"].to_numpy(dtype=float)), "value"
        ].to_numpy(dtype=float)
        if _has_distinct_finite_values(finite_values):
            conflict_groups += 1
        elif len(group.drop_duplicates(list(_SOURCE_KEY_COLUMNS) + ["value"])) == len(group):
            # This branch is only reached for groups that differ in a value
            # which is not finite; all-finite identical rows are counted too.
            pass
        if len(group.drop_duplicates(list(_SOURCE_KEY_COLUMNS) + ["value"])) == 1:
            exact_duplicate_groups += 1
            exact_duplicate_rows += len(group) - 1
    values = frame["value"].to_numpy(dtype=float)
    finite_rows = int(np.isfinite(values).sum())
    return SourceKeyAudit(
        source_rows=len(frame),
        exact_duplicate_groups=exact_duplicate_groups,
        exact_duplicate_rows=exact_duplicate_rows,
        conflict_groups=conflict_groups,
        nonfinite_rows=len(frame) - finite_rows,
        finite_rows=finite_rows,
    )


def _audit_candidate_frame(frame: pd.DataFrame, factor_name: str) -> SourceKeyAudit:
    """Compute source-shaped audit counts for candidate rows."""

    source_shape = frame.rename(columns={"candidate_value": "value"}).copy()
    source_shape["factor_name"] = factor_name
    return _audit_source_frame(source_shape.loc[:, list(_FACTOR_COLUMNS)])


def _has_distinct_finite_values(values: np.ndarray) -> bool:
    """Return whether finite values contain more than one exact value."""

    if len(values) < 2:
        return False
    return len(np.unique(values)) > 1


def _as_business_key(key: object) -> tuple[date, str]:
    """Convert pandas groupby keys to the stable business-key type."""

    if not isinstance(key, tuple):
        key = (key,)
    return key[0], str(key[1])


def _source_quarantine_row(
    row: pd.Series,
    reason: str,
    *,
    candidate_value: float = float("nan"),
) -> dict[str, object]:
    return {
        "trade_date": row["trade_date"],
        "ts_code": row["ts_code"],
        "factor_name": row["factor_name"],
        "source_value": row["value"],
        "candidate_value": candidate_value,
        "reason": reason,
    }


def _candidate_quarantine_row(
    row: pd.Series,
    reason: str,
    *,
    factor_name: str,
    source_value: float = float("nan"),
) -> dict[str, object]:
    return {
        "trade_date": row["trade_date"],
        "ts_code": row["ts_code"],
        "factor_name": factor_name,
        "source_value": source_value,
        "candidate_value": row["candidate_value"],
        "reason": reason,
    }


def _clean_overlay_frame(rows: list[pd.Series]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(_FACTOR_COLUMNS))
    clean = pd.DataFrame(rows).loc[:, list(_FACTOR_COLUMNS)]
    return clean.sort_values(list(_FACTOR_COLUMNS[:2]), kind="mergesort").reset_index(
        drop=True
    )


def _quarantine_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "trade_date",
        "ts_code",
        "factor_name",
        "source_value",
        "candidate_value",
        "reason",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    quarantine = pd.DataFrame(rows).loc[:, columns]
    return quarantine.sort_values(
        ["trade_date", "ts_code", "reason"], kind="mergesort"
    ).reset_index(drop=True)


def fill_missing_market_rows(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
) -> pd.DataFrame:
    """Add secondary rows absent from the primary market extract.

    Exact duplicate rows are removed first.  A remaining duplicate business
    key is treated as a blocking conflict rather than silently selecting one
    row.  Overlapping keys always retain the primary row, so this function is
    suitable for a secondary-only reconciliation workflow.
    """

    primary_normalized = _normalize_market(primary, "primary")
    secondary_normalized = _normalize_market(secondary, "secondary")
    primary_keys = pd.MultiIndex.from_frame(primary_normalized.loc[:, list(_BUSINESS_KEYS)])
    secondary_keys = pd.MultiIndex.from_frame(secondary_normalized.loc[:, list(_BUSINESS_KEYS)])
    missing = ~secondary_keys.isin(primary_keys)
    additions = secondary_normalized.loc[missing].copy()
    repaired = pd.concat([primary_normalized, additions], ignore_index=True)
    return repaired.sort_values(list(_BUSINESS_KEYS), kind="mergesort").reset_index(drop=True)


def compute_gp022_candidate(market: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct ``ts_max_60(pct_chg) * amount`` exactly as the calculator."""

    normalized = _normalize_market(market, "market")
    amount = _market_matrix(normalized, "amount")
    pct_chg = _market_matrix(normalized, "pct_chg")
    with np.errstate(over="ignore", invalid="ignore"):
        candidate = (pct_chg.rolling(60).max() * amount).replace(
            [np.inf, -np.inf], np.nan
        )
    return _candidate_long(candidate)


def compute_gp064_candidate(market: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the recovered GP_064 Python calculator expression.

    The implementation follows the original calculator's pandas semantics:
    rolling standard deviation uses its default sample ``ddof=1`` and rolling
    covariance/variance are computed over a five-observation window.
    """

    normalized = _normalize_market(market, "market")
    close = _market_matrix(normalized, "close")
    amount = _market_matrix(normalized, "amount")
    pct_chg = _market_matrix(normalized, "pct_chg")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        v1 = amount - 7.894736842105264
        v2 = v1.rolling(10).std()
        v3 = v2.rolling(60).mean()
        v5 = close.rolling(5).mean()
        v6 = v3.rolling(5).mean()
        v7 = close.rolling(5).cov(v3) / close.rolling(5).var().replace(0, np.nan)
        v4 = (v3 - v6) - v7 * (close - v5)
        v9 = np.exp(np.sign(v4)).replace([np.inf, -np.inf], np.nan)
        v10 = pct_chg.rolling(10).sum()
        v11 = (v9**v10).replace([np.inf, -np.inf], np.nan)
        candidate = (v11 + amount).replace([np.inf, -np.inf], np.nan)
    return _candidate_long(candidate)


def audit_factor_candidate(
    source: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    abs_tolerance: float = 1e-8,
    rel_tolerance: float = 1e-6,
) -> FactorRepairAudit:
    """Return overlap, fillability, and finite-value mismatch evidence."""

    source_normalized = _normalize_factor_source(source)
    candidate_normalized = _normalize_candidate(candidate)
    merged = source_normalized.merge(
        candidate_normalized,
        on=list(_BUSINESS_KEYS),
        how="left",
        validate="many_to_one",
    )
    source_values = pd.to_numeric(merged["value"], errors="coerce").to_numpy(dtype=float)
    candidate_values = pd.to_numeric(
        merged["candidate_value"], errors="coerce"
    ).to_numpy(dtype=float)
    source_finite = np.isfinite(source_values)
    candidate_finite = np.isfinite(candidate_values)
    overlap = source_finite & candidate_finite
    fillable = ~source_finite & candidate_finite
    remaining = ~source_finite & ~candidate_finite
    if overlap.any():
        abs_diff = np.abs(source_values[overlap] - candidate_values[overlap])
        rel_diff = abs_diff / np.maximum(np.abs(source_values[overlap]), 1e-12)
        mismatch = (abs_diff > abs_tolerance) & (rel_diff > rel_tolerance)
        max_abs = float(np.max(abs_diff))
        max_rel = float(np.max(rel_diff))
        mismatch_rows = int(np.sum(mismatch))
    else:
        max_abs = 0.0
        max_rel = 0.0
        mismatch_rows = 0
    return FactorRepairAudit(
        source_rows=len(source_normalized),
        source_finite_rows=int(source_finite.sum()),
        candidate_finite_rows=int(candidate_finite.sum()),
        overlap_finite_rows=int(overlap.sum()),
        fillable_rows=int(fillable.sum()),
        remaining_nonfinite_rows=int(remaining.sum()),
        mismatch_rows=mismatch_rows,
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
    )


def patch_factor_source(
    source: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    factor_name: str,
) -> pd.DataFrame:
    """Fill non-finite source values from a candidate without overwriting finite rows."""

    if not factor_name.strip():
        raise FactorRepairError("factor_name cannot be empty")
    source_normalized = _normalize_factor_source(source)
    candidate_normalized = _normalize_candidate(candidate)
    if not (source_normalized["factor_name"].astype(str) == factor_name).all():
        raise FactorRepairError("source contains an unexpected factor name")
    merged = source_normalized.merge(
        candidate_normalized,
        on=list(_BUSINESS_KEYS),
        how="left",
        validate="many_to_one",
    )
    source_values = pd.to_numeric(merged["value"], errors="coerce").to_numpy(dtype=float)
    candidate_values = pd.to_numeric(
        merged["candidate_value"], errors="coerce"
    ).to_numpy(dtype=float)
    fillable = ~np.isfinite(source_values) & np.isfinite(candidate_values)
    source_normalized["value"] = np.where(fillable, candidate_values, source_values)
    return source_normalized.loc[:, list(_FACTOR_COLUMNS)]


def _normalize_market(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = sorted(set(_MARKET_COLUMNS).difference(frame.columns))
    if missing:
        raise FactorRepairError(f"{label} market frame is missing columns: {missing}")
    normalized = frame.loc[:, list(_MARKET_COLUMNS)].copy(deep=True)
    normalized["trade_date"] = pd.to_datetime(normalized["trade_date"], errors="raise").dt.date
    normalized["ts_code"] = normalized["ts_code"].astype(str)
    if normalized.loc[:, list(_BUSINESS_KEYS)].isna().any().any():
        raise FactorRepairError(f"{label} market keys cannot be null")
    normalized = normalized.drop_duplicates(keep="first")
    if normalized.duplicated(list(_BUSINESS_KEYS), keep=False).any():
        raise FactorRepairError(f"{label} contains conflicting duplicate market keys")
    for column in _MARKET_COLUMNS[2:]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def _normalize_factor_source(source: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(_FACTOR_COLUMNS).difference(source.columns))
    if missing:
        raise FactorRepairError(f"factor source is missing columns: {missing}")
    normalized = source.loc[:, list(_FACTOR_COLUMNS)].copy(deep=True)
    normalized["trade_date"] = pd.to_datetime(normalized["trade_date"], errors="raise").dt.date
    normalized["ts_code"] = normalized["ts_code"].astype(str)
    return normalized


def _normalize_candidate(candidate: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(_CANDIDATE_COLUMNS).difference(candidate.columns))
    if missing:
        raise FactorRepairError(f"factor candidate is missing columns: {missing}")
    normalized = candidate.loc[:, list(_CANDIDATE_COLUMNS)].copy(deep=True)
    normalized["trade_date"] = pd.to_datetime(normalized["trade_date"], errors="raise").dt.date
    normalized["ts_code"] = normalized["ts_code"].astype(str)
    if normalized.duplicated(list(_BUSINESS_KEYS), keep=False).any():
        raise FactorRepairError("duplicate candidate keys")
    normalized["candidate_value"] = pd.to_numeric(
        normalized["candidate_value"], errors="coerce"
    )
    return normalized


def _market_matrix(market: pd.DataFrame, column: str) -> pd.DataFrame:
    matrix = market.pivot(index="trade_date", columns="ts_code", values=column).sort_index()
    matrix.index.name = "trade_date"
    matrix.columns.name = "ts_code"
    return matrix


def _candidate_long(matrix: pd.DataFrame) -> pd.DataFrame:
    long = matrix.reset_index().melt(
        id_vars=["trade_date"],
        var_name="ts_code",
        value_name="candidate_value",
    )
    return long.sort_values(list(_BUSINESS_KEYS), kind="mergesort").reset_index(drop=True)
