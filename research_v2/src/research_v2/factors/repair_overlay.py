"""Bounded, read-only repair providers for technical factor observations.

Providers in this module are deliberately small adapters around the pure
helpers in :mod:`research_v2.factors.repair`.  A provider may return evidence
for one factor/month, but it cannot write a source table or overwrite a
finite source value.  ``build_repair_overlay`` is the only assembly boundary;
it applies the no-overwrite/PIT decision contract and adds deterministic
lineage columns to the returned frames.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

from research_v2.contracts.factor import FactorDefinition
from research_v2.factors.repair import (
    FactorRepairError,
    RepairStatus,
    apply_repair_overlay,
    compute_gp022_candidate,
    compute_gp064_candidate,
    fill_missing_market_rows,
)
from research_v2.factors.repair import RepairOverlayResult as RepairDecision
from research_v2.pit.reconciliation import reconcile_market

_MARKET_COLUMNS = ("trade_date", "ts_code", "close", "amount", "pct_chg")
_CANDIDATE_COLUMNS = ("trade_date", "ts_code", "candidate_value")


class RepairProvider(Protocol):
    """Contract for one bounded factor/month candidate provider.

    Implementations receive a single source frame and return either ``None``
    or a long-form candidate frame containing exactly the three candidate
    columns.  Keyword arguments are optional context so lightweight providers
    can implement only ``provide(source)``.
    """

    method: str

    def provide(
        self,
        source: pd.DataFrame,
        *,
        factor_id: str | None = None,
        factor_name: str | None = None,
        logical_partition: str | None = None,
        source_watermark_id: str | None = None,
    ) -> pd.DataFrame | None: ...


@dataclass(frozen=True)
class RepairOverlayBuildResult:
    """Decision plus annotated clean/quarantine frames for one factor/month."""

    status: RepairStatus
    clean: pd.DataFrame
    quarantine: pd.DataFrame
    repair_method: str
    source_watermark_id: str
    candidate_hash: str
    decision: RepairDecision
    provider: RepairProvider

    @property
    def audit(self) -> RepairDecision:
        """Expose the pure decision under the conventional audit name."""

        return self.decision

    @property
    def overlay(self) -> RepairDecision:
        """Compatibility alias for callers that call the decision an overlay."""

        return self.decision

    @property
    def filled_rows(self) -> int:
        return self.decision.filled_rows

    @property
    def added_rows(self) -> int:
        return self.decision.added_rows

    @property
    def unresolved_rows(self) -> int:
        return self.decision.unresolved_rows

    @property
    def pit_violation_rows(self) -> int:
        return self.decision.pit_violation_rows


class NullRepairProvider:
    """Fail-closed provider used when no bounded evidence is registered."""

    method = "NO_EVIDENCE_BACKFILL"
    source_watermark_id = ""

    def __init__(self) -> None:
        self.reasons: list[str] = []

    @property
    def repair_method(self) -> str:
        return self.method

    @property
    def last_reason(self) -> str | None:
        return self.reasons[-1] if self.reasons else None

    @property
    def events(self) -> list[str]:
        """Compatibility alias for consumers that call recorded reasons events."""

        return self.reasons

    def provide(
        self,
        source: pd.DataFrame,
        **_: object,
    ) -> None:
        del source
        self.reasons.append("NO_EVIDENCE_BACKFILL")
        return None

    # A callable alias keeps the provider convenient for source runners and
    # supports the same contract as ``provide`` without fabricating evidence.
    __call__ = provide


class MarketCandidateProvider:
    """Reconcile a secondary market extract without replacing primary rows.

    A secondary watermark is mandatory.  The market operations are delegated
    to the existing no-overwrite helpers; the provider only retains their
    in-memory evidence.  A market frame alone does not imply a factor formula,
    so ``provide`` returns ``None`` unless an explicit candidate column is
    supplied or a known GP factor is requested.
    """

    method = "MARKET_SECONDARY_RECONCILIATION"

    def __init__(
        self,
        secondary: pd.DataFrame,
        *market_frames: pd.DataFrame,
        secondary_watermark_id: str | None = None,
        secondary_watermark: str | None = None,
        primary_market: pd.DataFrame | None = None,
        primary: pd.DataFrame | None = None,
    ) -> None:
        if len(market_frames) > 1:
            raise TypeError("at most one positional primary/secondary market frame is allowed")
        if market_frames:
            # Accept the intuitive ``(primary, secondary)`` form in addition
            # to the documented ``(secondary, primary_market=...)`` form.
            primary_market = secondary
            secondary = market_frames[0]
        if primary is not None:
            if primary_market is not None:
                raise TypeError("primary and primary_market cannot both be supplied")
            primary_market = primary
        watermark = secondary_watermark_id or secondary_watermark
        if not watermark or not str(watermark).strip():
            raise ValueError("secondary watermark is required")
        self.secondary = secondary.copy(deep=True)
        self.secondary_watermark_id = str(watermark)
        self.source_watermark_id = self.secondary_watermark_id
        self.primary_market = (
            primary_market.copy(deep=True) if primary_market is not None else None
        )
        self.last_market_frame: pd.DataFrame | None = None
        self.last_reconciliation: pd.DataFrame | None = None

    @property
    def secondary_watermark(self) -> str:
        return self.secondary_watermark_id

    @property
    def repair_method(self) -> str:
        return self.method

    def provide(
        self,
        source: pd.DataFrame,
        *,
        factor_name: str | None = None,
        **_: object,
    ) -> pd.DataFrame | None:
        primary = self._market_frame(source)
        if primary is None:
            raise ValueError(
                "market candidate provider requires market columns or primary_market"
            )
        # Keep both calls explicit: reconciliation is an audit side effect,
        # while fill_missing_market_rows is the bounded row assembly.
        self.last_market_frame = fill_missing_market_rows(primary, self.secondary)
        self.last_reconciliation = reconcile_market(primary, self.secondary)

        # Callers that already supplied calculated candidate values can use the
        # provider as a market-key gate.  We never treat arbitrary market fields
        # as a factor value.
        if set(_CANDIDATE_COLUMNS).issubset(source.columns):
            return source.loc[:, list(_CANDIDATE_COLUMNS)].copy(deep=True)
        if factor_name == "GP_022":
            return compute_gp022_candidate(self.last_market_frame)
        if factor_name == "GP_064":
            return compute_gp064_candidate(self.last_market_frame)
        return None

    __call__ = provide

    def _market_frame(self, source: pd.DataFrame) -> pd.DataFrame | None:
        if set(_MARKET_COLUMNS).issubset(source.columns):
            return source.loc[:, list(_MARKET_COLUMNS)].copy(deep=True)
        if self.primary_market is not None:
            return self.primary_market.copy(deep=True)
        return None


class KnownFormulaCandidateProvider:
    """Reconstruct only the approved GP_022 and GP_064 expressions."""

    method = "KNOWN_FORMULA_RECONSTRUCTION"

    def __init__(
        self,
        market: pd.DataFrame | None = None,
        *,
        market_frame: pd.DataFrame | None = None,
        source_watermark_id: str = "known-formula-market",
        abs_tolerance: float = 1e-8,
        rel_tolerance: float = 1e-6,
    ) -> None:
        if not source_watermark_id.strip():
            raise ValueError("source watermark cannot be empty")
        resolved_market = market if market is not None else market_frame
        if resolved_market is None:
            raise ValueError("known formula provider requires a market frame")
        self.market = resolved_market.copy(deep=True)
        self.source_watermark_id = source_watermark_id
        self.abs_tolerance = abs_tolerance
        self.rel_tolerance = rel_tolerance

    @property
    def repair_method(self) -> str:
        return self.method

    def provide(
        self,
        source: pd.DataFrame,
        *,
        factor_name: str | None = None,
        **_: object,
    ) -> pd.DataFrame | None:
        name = factor_name
        if name is None and "factor_name" in source.columns and not source.empty:
            names = source["factor_name"].dropna().astype(str).unique().tolist()
            if len(names) == 1:
                name = names[0]
        calculators = {"GP_022": compute_gp022_candidate, "GP_064": compute_gp064_candidate}
        if name not in calculators:
            raise ValueError("known formula provider only supports GP_022 and GP_064")
        candidate = calculators[name](self.market)
        self._reject_overlap_mismatch(source, candidate)
        return candidate

    __call__ = provide

    def _reject_overlap_mismatch(
        self,
        source: pd.DataFrame,
        candidate: pd.DataFrame,
    ) -> None:
        required = {"trade_date", "ts_code", "value"}
        if not required.issubset(source.columns):
            return
        left = source.loc[:, ["trade_date", "ts_code", "value"]].copy()
        left["trade_date"] = pd.to_datetime(left["trade_date"], errors="raise").dt.date
        left["ts_code"] = left["ts_code"].astype(str)
        right = candidate.copy()
        right["trade_date"] = pd.to_datetime(right["trade_date"], errors="raise").dt.date
        right["ts_code"] = right["ts_code"].astype(str)
        merged = left.merge(right, on=["trade_date", "ts_code"], how="inner")
        if merged.empty:
            return
        source_values = pd.to_numeric(merged["value"], errors="coerce").to_numpy(float)
        candidate_values = pd.to_numeric(
            merged["candidate_value"], errors="coerce"
        ).to_numpy(float)
        finite = np.isfinite(source_values) & np.isfinite(candidate_values)
        if not finite.any():
            return
        abs_diff = np.abs(source_values[finite] - candidate_values[finite])
        rel_diff = abs_diff / np.maximum(np.abs(source_values[finite]), 1e-12)
        mismatch = (abs_diff > self.abs_tolerance) & (rel_diff > self.rel_tolerance)
        if mismatch.any():
            raise FactorRepairError("known formula overlap mismatch")


# Explicit by-factor registration.  Providers need market dependencies at
# runtime, so the default registry stores the approved method names; callers
# may pass instantiated providers through ``providers``.
EXPLICIT_FACTOR_PROVIDER_METHODS: Mapping[str, str] = {
    "yeslab_391": "GP_022",
    "yeslab_433": "GP_064",
}
REPAIR_PROVIDER_MAPPING = EXPLICIT_FACTOR_PROVIDER_METHODS


def provider_for_factor(
    factor_id: str,
    providers: Mapping[str, RepairProvider] | None = None,
) -> RepairProvider:
    """Return the explicitly registered provider, or a fail-closed null one."""

    if factor_id not in EXPLICIT_FACTOR_PROVIDER_METHODS:
        return NullRepairProvider()
    if providers is not None:
        provider = providers.get(factor_id)
        if provider is not None:
            return provider
    return NullRepairProvider()


def build_repair_provider_mapping(
    providers: Mapping[str, RepairProvider] | None = None,
    *,
    factor_ids: tuple[str, ...] = (),
) -> dict[str, RepairProvider]:
    """Materialize an explicit factor-ID map with null defaults.

    The map is intentionally not inferred from factor names.  A caller must
    provide instantiated dependencies for the two approved formula IDs; every
    requested unknown/missing ID receives a fresh ``NullRepairProvider``.
    """

    requested = set(factor_ids).union(providers or {})
    return {factor_id: provider_for_factor(factor_id, providers) for factor_id in sorted(requested)}


def build_repair_overlay(
    source: pd.DataFrame,
    provider: RepairProvider | None = None,
    *,
    definition: FactorDefinition | None = None,
    factor_name: str | None = None,
    factor_id: str | None = None,
    logical_partition: str | None = None,
    source_watermark_id: str | None = None,
    source_watermark: str | None = None,
    signal_date: date | None = None,
    abs_tolerance: float = 1e-8,
    rel_tolerance: float = 1e-6,
) -> RepairOverlayBuildResult:
    """Request one candidate, apply the pure repair decision, and annotate it."""

    active_provider: RepairProvider = provider or NullRepairProvider()
    resolved_name = factor_name or (definition.display_name if definition else None)
    if (
        not resolved_name
        and "factor_name" in source.columns
        and not source.empty
    ):
        names = source["factor_name"].dropna().astype(str).unique().tolist()
        if len(names) == 1:
            resolved_name = names[0]
    if not resolved_name:
        raise ValueError("factor_name or a definition is required")
    candidate = _invoke_provider(
        active_provider,
        source,
        factor_id=factor_id or (definition.factor_id if definition else None),
        factor_name=resolved_name,
        logical_partition=logical_partition,
        source_watermark_id=source_watermark_id,
    )
    if candidate is not None and not isinstance(candidate, pd.DataFrame):
        raise TypeError("repair provider must return a DataFrame or None")
    candidate, scope_quarantine, scope_blocking = _restrict_candidate_to_partition(
        candidate,
        factor_name=resolved_name,
        logical_partition=logical_partition,
    )
    decision = apply_repair_overlay(
        source,
        candidate,
        factor_name=resolved_name,
        signal_date=signal_date,
        abs_tolerance=abs_tolerance,
        rel_tolerance=rel_tolerance,
    )
    if not scope_quarantine.empty:
        combined_quarantine = (
            scope_quarantine.copy(deep=True)
            if decision.quarantine.empty
            else pd.concat([decision.quarantine, scope_quarantine], ignore_index=True)
        )
        decision = replace(
            decision,
            status=(
                RepairStatus.BLOCKED
                if scope_blocking or decision.status is RepairStatus.BLOCKED
                else RepairStatus.PASS_WITH_QUARANTINE
            ),
            quarantine=combined_quarantine,
        )
    method = str(getattr(active_provider, "method", active_provider.__class__.__name__))
    watermark = source_watermark_id or source_watermark or str(
        getattr(active_provider, "source_watermark_id", "")
    )
    candidate_digest = _candidate_hash(candidate)
    clean = decision.clean.copy(deep=True)
    quarantine = decision.quarantine.copy(deep=True)
    if isinstance(active_provider, NullRepairProvider) and not quarantine.empty:
        quarantine["reason"] = quarantine["reason"].replace(
            {"NO_FINITE_CANDIDATE": "NO_EVIDENCE_BACKFILL"}
        )
    for frame in (clean, quarantine):
        frame["repair_method"] = method
        frame["source_watermark_id"] = watermark
        frame["repair_source_watermark_id"] = watermark
        frame["candidate_hash"] = candidate_digest
        frame["repair_candidate_hash"] = candidate_digest
        frame["repair_status"] = decision.status.value
        frame["decision_status"] = decision.status.value
    return RepairOverlayBuildResult(
        status=decision.status,
        clean=clean,
        quarantine=quarantine,
        repair_method=method,
        source_watermark_id=watermark,
        candidate_hash=candidate_digest,
        decision=decision,
        provider=active_provider,
    )


def _restrict_candidate_to_partition(
    candidate: pd.DataFrame | None,
    *,
    factor_name: str,
    logical_partition: str | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame, bool]:
    """Drop candidate warmup rows and return them as date-scope quarantine."""

    quarantine_columns = [
        "trade_date",
        "ts_code",
        "factor_name",
        "source_value",
        "candidate_value",
        "reason",
    ]
    if candidate is None or logical_partition is None:
        return candidate, pd.DataFrame(columns=quarantine_columns), False
    try:
        period = pd.Period(logical_partition, freq="M")
    except (TypeError, ValueError) as error:
        raise ValueError("logical_partition must use YYYY-MM") from error
    if str(period) != logical_partition:
        raise ValueError("logical_partition must use YYYY-MM")
    missing = sorted(set(_CANDIDATE_COLUMNS).difference(candidate.columns))
    if missing:
        raise FactorRepairError(f"factor candidate is missing columns: {missing}")
    dates = pd.to_datetime(candidate["trade_date"], errors="raise").dt.date
    lower = period.start_time.date()
    upper = period.end_time.date()
    in_scope = (dates >= lower) & (dates <= upper)
    out_of_scope = candidate.loc[~in_scope, list(_CANDIDATE_COLUMNS)].copy(deep=True)
    if out_of_scope.empty:
        return candidate, pd.DataFrame(columns=quarantine_columns), False
    out_of_scope["trade_date"] = dates.loc[out_of_scope.index]
    out_of_scope["ts_code"] = out_of_scope["ts_code"].astype(str)
    scope_quarantine = out_of_scope.assign(
        factor_name=factor_name,
        source_value=np.nan,
        reason=np.where(
            out_of_scope["trade_date"] > upper,
            "DATE_SCOPE_VIOLATION",
            "WARMUP_OUT_OF_SCOPE",
        ),
    ).loc[:, quarantine_columns]
    scope_quarantine = scope_quarantine.sort_values(
        ["trade_date", "ts_code"], kind="mergesort"
    ).reset_index(drop=True)
    blocking = bool((out_of_scope["trade_date"] > upper).any())
    return candidate.loc[in_scope].copy(deep=True), scope_quarantine, blocking


def _invoke_provider(
    provider: RepairProvider,
    source: pd.DataFrame,
    **kwargs: object,
) -> pd.DataFrame | None:
    target: Any
    if hasattr(provider, "provide"):
        target = provider.provide
    elif hasattr(provider, "candidate"):
        target = cast(Any, provider).candidate
    elif callable(provider):
        target = provider
    else:
        raise TypeError("repair provider must expose provide(source) or be callable")
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return cast(pd.DataFrame | None, target(source, **kwargs))
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    supported = kwargs if accepts_kwargs else {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return cast(pd.DataFrame | None, target(source, **supported))


def _candidate_hash(candidate: pd.DataFrame | None) -> str:
    if candidate is None or candidate.empty:
        return hashlib.sha256(b"EMPTY_CANDIDATE").hexdigest()
    frame = candidate.loc[
        :, [column for column in _CANDIDATE_COLUMNS if column in candidate]
    ].copy()
    if "trade_date" in frame:
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.strftime(
            "%Y-%m-%d"
        )
    frame = frame.sort_values(list(frame.columns), kind="mergesort").reset_index(drop=True)
    records = frame.to_dict(orient="records")
    canonical = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Public aliases used by callers that name the assembly result differently.
RepairOverlayResult = RepairOverlayBuildResult
RepairOverlayAssembly = RepairOverlayBuildResult


__all__ = [
    "EXPLICIT_FACTOR_PROVIDER_METHODS",
    "KnownFormulaCandidateProvider",
    "MarketCandidateProvider",
    "NullRepairProvider",
    "RepairOverlayAssembly",
    "RepairOverlayBuildResult",
    "RepairOverlayResult",
    "RepairProvider",
    "RepairStatus",
    "build_repair_provider_mapping",
    "build_repair_overlay",
    "provider_for_factor",
]
