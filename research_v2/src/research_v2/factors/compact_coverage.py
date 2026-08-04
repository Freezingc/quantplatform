"""Point-in-time factor coverage on the actual compact research universe."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np
import pandas as pd

_KEYS: Final[tuple[str, str]] = ("signal_date", "ts_code")


def summarize_compact_coverage(
    universe: pd.DataFrame,
    labels: pd.DataFrame,
    source: pd.DataFrame,
    *,
    factor_id: str,
    factor_name: str,
    logical_partition: str,
    minimum_overall_ratio: float = 0.80,
    minimum_daily_ratio: float = 0.70,
    minimum_p05_ratio: float = 0.70,
    minimum_monthly_ratio: float = 0.70,
) -> Mapping[str, object]:
    """Return deterministic finite coverage after PIT universe/label intersection.

    Exact source revisions are counted and collapsed.  A remaining business-key
    conflict blocks the result instead of selecting a value.  The denominator
    is the intersection of active universe members and valid labels, so broad
    market-wide missingness cannot be mistaken for model-input missingness.
    """

    _require_columns(universe, {"signal_date", "ts_code", "is_member"}, "universe")
    _require_columns(labels, {"signal_date", "ts_code", "label_valid"}, "labels")
    _require_columns(source, {"trade_date", "ts_code", "value"}, "source")
    universe_frame = _normalize_keys(universe, "signal_date", "universe")
    labels_frame = _normalize_keys(labels, "signal_date", "labels")
    source_frame = _normalize_keys(source, "trade_date", "source")
    universe_frame = universe_frame.loc[universe_frame["is_member"].fillna(False)].copy()
    labels_frame = labels_frame.loc[labels_frame["label_valid"].fillna(False)].copy()
    universe_frame = universe_frame.drop_duplicates(list(_KEYS)).loc[:, list(_KEYS)]
    labels_frame = labels_frame.drop_duplicates(list(_KEYS)).loc[:, list(_KEYS)]
    mature = universe_frame.merge(labels_frame, on=list(_KEYS), how="inner", validate="one_to_one")

    source_rows = len(source_frame)
    source_frame["value"] = pd.to_numeric(source_frame["value"], errors="coerce")
    source_frame = source_frame.drop_duplicates().reset_index(drop=True)
    exact_duplicate_rows = source_rows - len(source_frame)
    key_counts = source_frame.groupby(list(_KEYS), sort=False).size()
    conflict_keys = int(key_counts.gt(1).sum())
    if conflict_keys:
        return _result(
            factor_id=factor_id,
            factor_name=factor_name,
            logical_partition=logical_partition,
            source_rows=source_rows,
            source_unique_rows=len(source_frame),
            source_exact_duplicate_rows=exact_duplicate_rows,
            source_conflict_keys=conflict_keys,
            compact_rows=len(mature),
            compact_finite_rows=0,
            compact_finite_ratio=0.0,
            minimum_daily_finite_ratio=0.0,
            p05_daily_finite_ratio=0.0,
            minimum_monthly_finite_ratio=0.0,
            decision_status="COMPACT_DQ_BLOCKED",
            reason_codes=("SOURCE_VALUE_CONFLICT",),
        )
    source_frame = source_frame.drop_duplicates(list(_KEYS), keep="last")
    merged = mature.merge(
        source_frame.loc[:, [*list(_KEYS), "value"]],
        on=list(_KEYS),
        how="left",
        validate="one_to_one",
    )
    finite = np.isfinite(merged["value"].to_numpy(dtype=float))
    compact_rows = len(merged)
    compact_finite_rows = int(finite.sum())
    overall_ratio = compact_finite_rows / compact_rows if compact_rows else 0.0
    daily = merged.assign(_finite=finite).groupby("signal_date", sort=True)['_finite'].mean()
    monthly = daily.groupby(
        daily.index.to_series().map(lambda value: value.strftime("%Y-%m"))
    ).mean()
    minimum_daily = float(daily.min()) if not daily.empty else 0.0
    p05_daily = float(daily.quantile(0.05)) if not daily.empty else 0.0
    minimum_monthly = float(monthly.min()) if not monthly.empty else 0.0
    gate = bool(
        compact_rows
        and overall_ratio >= minimum_overall_ratio
        and minimum_daily >= minimum_daily_ratio
        and p05_daily >= minimum_p05_ratio
        and minimum_monthly >= minimum_monthly_ratio
    )
    reasons = ("COMPACT_FINITE_COVERAGE_PASS",) if gate else ("LOW_COMPACT_FINITE_COVERAGE",)
    return _result(
        factor_id=factor_id,
        factor_name=factor_name,
        logical_partition=logical_partition,
        source_rows=source_rows,
        source_unique_rows=len(source_frame),
        source_exact_duplicate_rows=exact_duplicate_rows,
        source_conflict_keys=0,
        compact_rows=compact_rows,
        compact_finite_rows=compact_finite_rows,
        compact_finite_ratio=overall_ratio,
        minimum_daily_finite_ratio=minimum_daily,
        p05_daily_finite_ratio=p05_daily,
        minimum_monthly_finite_ratio=minimum_monthly,
        decision_status=("COMPACT_DQ_PASS_PENDING_SIMILARITY" if gate else "COMPACT_DQ_FAIL"),
        reason_codes=reasons,
    )


def _result(**values: object) -> Mapping[str, object]:
    return values


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"{label} is missing columns: {missing}")


def _normalize_keys(frame: pd.DataFrame, date_column: str, label: str) -> pd.DataFrame:
    normalized = frame.copy(deep=True)
    normalized["signal_date"] = pd.to_datetime(
        normalized[date_column], errors="raise"
    ).dt.date
    normalized["ts_code"] = normalized["ts_code"].astype(str)
    if normalized.loc[:, list(_KEYS)].isna().any().any():
        raise ValueError(f"{label} business keys cannot be null")
    return normalized
