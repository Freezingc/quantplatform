"""Point-in-time dataset inputs backed by the latest compact V2 tables."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final

import pandas as pd

from research_v2.compact.publisher import CompactPublishSession, read_latest_partition
from research_v2.datasets.builder import (
    DatasetBuildRequest,
    FactorInput,
    LockedDatasetError,
    LockedTestAccess,
    build_point_in_time_dataset,
)
from research_v2.factors.admission import (
    FactorAdmissionContract,
    admit_feature_frame,
)


class CompactDataUnavailable(RuntimeError):
    """Raised when a requested month has no published compact batch."""


_DATE_COLUMNS: Final[dict[str, str]] = {
    "universe_daily_latest": "signal_date",
    "label_daily_latest": "signal_date",
    "factor_observation_clean_latest": "signal_date",
    "tradability_daily_latest": "trade_date",
}


@dataclass(frozen=True)
class CompactDatasetReader:
    """Read manifest-selected compact data and build a PIT training frame.

    The reader is intentionally fail-closed: a missing month is an error rather
    than an empty join. This prevents a partially populated latest table from
    silently changing the training interval or factor coverage.
    """

    session: CompactPublishSession

    def read_range(
        self,
        *,
        object_name: str,
        start_date: date,
        end_date: date,
        columns: tuple[str, ...] = (),
        factor_keys: tuple[tuple[str, int], ...] = (),
        locked_access: LockedTestAccess | None = None,
    ) -> pd.DataFrame:
        """Read every manifest-published month intersecting a closed date range."""
        if start_date > end_date:
            raise ValueError("start_date cannot be after end_date")
        if object_name == "label_daily_latest":
            if locked_access is None and end_date >= date(2026, 1, 1):
                raise LockedDatasetError(
                    "ordinary compact reads cannot consume 2026 FINAL or future labels"
                )
            if locked_access is not None and (
                start_date.year != locked_access.locked_year
                or end_date.year != locked_access.locked_year
            ):
                raise LockedDatasetError(
                    "locked label reads must stay within the authorized 2026 segment"
                )
            if locked_access is not None:
                locked_access.verify()
        try:
            date_column = _DATE_COLUMNS[object_name]
        except KeyError as error:
            raise ValueError(f"unsupported compact object: {object_name}") from error

        monthly_frames: list[pd.DataFrame] = []
        for logical_partition in _monthly_partitions(start_date, end_date):
            frame = read_latest_partition(
                self.session,
                object_name=object_name,
                logical_partition=logical_partition,
                columns=columns,
                factor_keys=factor_keys,
            )
            if frame.empty:
                raise CompactDataUnavailable(
                    f"no published compact batch for {object_name}/{logical_partition}"
                )
            if date_column not in frame.columns:
                raise CompactDataUnavailable(
                    f"compact batch is missing its date column: {object_name}/{date_column}"
                )
            dates = pd.to_datetime(frame[date_column], errors="raise").dt.date
            scoped = frame.loc[(dates >= start_date) & (dates <= end_date)].copy()
            if not scoped.empty:
                monthly_frames.append(scoped)

        if not monthly_frames:
            raise CompactDataUnavailable(
                f"compact range has no rows: {object_name}/{start_date.isoformat()}"
                f"..{end_date.isoformat()}"
            )
        return pd.concat(monthly_frames, ignore_index=True)

    def build_point_in_time_dataset(
        self,
        *,
        start_date: date,
        end_date: date,
        factor_members: tuple[FactorInput, ...],
        segment_end: date,
        request: DatasetBuildRequest,
        admission_contract: FactorAdmissionContract,
        minimum_finite_ratio: float = 0.70,
    ) -> pd.DataFrame:
        """Build a PIT frame exclusively from latest compact tables.

        The contract is required here (unlike the low-level in-memory join),
        making this the production-facing training boundary.
        """
        if admission_contract is None:  # defensive runtime guard for untyped callers
            raise ValueError("admission_contract is required for compact dataset builds")
        authorized_locked_test = (
            request.fold_id == "FINAL"
            and request.split_role == "TEST"
            and request.locked_access is not None
        )
        if authorized_locked_test:
            if start_date.year != 2026 or end_date.year != 2026:
                raise LockedDatasetError(
                    "locked FINAL dataset reads must stay within the authorized 2026 segment"
                )
        elif end_date >= date(2026, 1, 1):
            raise LockedDatasetError(
                "ordinary compact dataset reads cannot consume 2026 FINAL or future dates"
            )
        factor_keys = tuple((member.factor_id, member.factor_version) for member in factor_members)
        universe = self.read_range(
            object_name="universe_daily_latest",
            start_date=start_date,
            end_date=end_date,
            columns=("batch_id", "signal_date", "ts_code", "is_member"),
        )
        universe = universe.loc[universe["is_member"].astype(bool)].copy()
        factors = self.read_range(
            object_name="factor_observation_clean_latest",
            start_date=start_date,
            end_date=end_date,
            columns=(
                "batch_id",
                "signal_date",
                "ts_code",
                "factor_id",
                "factor_version",
                "value",
                "available_date",
            ),
            factor_keys=factor_keys,
        )
        # Perform the hard feature-coverage gate before reading locked labels.
        # A failed factor contract must not consume the one-time FINAL/TEST
        # label capability merely to discover that its inputs are unusable.
        _preflight_factor_admission(
            universe,
            factors,
            factor_members=factor_members,
            request=request,
            admission_contract=admission_contract,
            minimum_finite_ratio=minimum_finite_ratio,
        )
        labels = self.read_range(
            object_name="label_daily_latest",
            start_date=start_date,
            end_date=end_date,
            columns=(
                "batch_id",
                "signal_date",
                "ts_code",
                "raw_return",
                "label_valid",
                "exit_date",
            ),
            locked_access=request.locked_access if authorized_locked_test else None,
        )
        return build_point_in_time_dataset(
            universe,
            factors,
            labels,
            factor_members=factor_members,
            segment_end=segment_end,
            request=request,
            admission_contract=admission_contract,
            minimum_finite_ratio=minimum_finite_ratio,
        )


def _preflight_factor_admission(
    universe: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    factor_members: tuple[FactorInput, ...],
    request: DatasetBuildRequest,
    admission_contract: FactorAdmissionContract,
    minimum_finite_ratio: float,
) -> None:
    """Validate factor identity, PIT availability, and coverage without labels."""
    keys = ["signal_date", "ts_code"]
    result = universe.loc[:, keys].copy()
    result["signal_date"] = pd.to_datetime(result["signal_date"], errors="raise").dt.date
    normalized_factors = factors.copy()
    normalized_factors["signal_date"] = pd.to_datetime(
        normalized_factors["signal_date"], errors="raise"
    ).dt.date
    if normalized_factors.duplicated(
        ["signal_date", "ts_code", "factor_id", "factor_version"]
    ).any():
        raise ValueError("factors contain duplicate business keys")
    feature_names = [member.feature_name for member in factor_members]
    for member in factor_members:
        member_rows = normalized_factors.loc[
            (normalized_factors["factor_id"].astype(str) == member.factor_id)
            & (normalized_factors["factor_version"].astype(int) == member.factor_version),
            keys + ["value", "available_date"],
        ].copy()
        member_rows["available_date"] = pd.to_datetime(
            member_rows["available_date"], errors="raise"
        ).dt.date
        if member_rows["available_date"].isna().any():
            raise ValueError(f"factor has null available_date: {member.factor_id}")
        if (member_rows["available_date"] > member_rows["signal_date"]).any():
            raise ValueError(f"factor is not available by signal date: {member.factor_id}")
        result = result.merge(
            member_rows.loc[:, keys + ["value"]].rename(columns={"value": member.feature_name}),
            on=keys,
            how="left",
            validate="one_to_one",
        )
    admitted = result.loc[:, keys + feature_names]
    admit_feature_frame(
        admission_contract,
        admitted,
        role=request.split_role,
        minimum_finite_ratio=minimum_finite_ratio,
    )


def _monthly_partitions(start_date: date, end_date: date) -> tuple[str, ...]:
    current = pd.Period(start_date, freq="M")
    end = pd.Period(end_date, freq="M")
    partitions: list[str] = []
    while current <= end:
        partitions.append(str(current))
        current += 1
    return tuple(partitions)
