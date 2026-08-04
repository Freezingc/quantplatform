from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from research_v2.contracts.factor import AvailabilityPolicy, FactorDefinition, FactorType
from research_v2.factors.repair import RepairStatus
from research_v2.factors.repair_overlay import (
    KnownFormulaCandidateProvider,
    MarketCandidateProvider,
    NullRepairProvider,
    build_repair_overlay,
    build_repair_provider_mapping,
    provider_for_factor,
)
from research_v2.factors.source_backfill import (
    FactorBackfillError,
    FactorMonth,
    build_factor_month_context,
    iter_clean_factor_batches,
    iter_factor_materialization_batches,
)


def _definition(factor_id: str = "yeslab_22", display_name: str = "GP_022") -> FactorDefinition:
    return FactorDefinition(
        factor_id=factor_id,
        factor_version=1,
        display_name=display_name,
        source_database="dfs://YesLab",
        source_table="factor_data1",
        source_value_column="value",
        source_date_column="trade_date",
        source_code_column="ts_code",
        factor_type=FactorType.TECHNICAL,
        frequency="daily",
        expected_direction=1,
        availability_policy=AvailabilityPolicy.SAME_DAY_CLOSE,
        max_staleness_calendar_days=0,
        unit="raw",
        owner="test",
    )


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [date(2025, 1, 2), date(2025, 1, 2)],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "factor_name": ["GP_022", "GP_022"],
            "value": [np.nan, 4.0],
        }
    )


class FiniteProvider:
    method = "TEST_CANDIDATE"

    def provide(self, source: pd.DataFrame, **_: object) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_date": [date(2025, 1, 2)],
                "ts_code": ["000001.SZ"],
                "candidate_value": [2.0],
            }
        )


def test_overlay_provider_fills_only_nonfinite_values_and_annotates_lineage() -> None:
    result = build_repair_overlay(
        _source(),
        FiniteProvider(),
        definition=_definition(),
        source_watermark_id="primary-w1",
    )

    assert result.status is RepairStatus.PENDING_RECHECK
    assert result.filled_rows == 1
    assert result.clean.loc[0, "value"] == 2.0
    assert result.clean.loc[0, "repair_method"] == "TEST_CANDIDATE"
    assert result.clean.loc[0, "source_watermark_id"] == "primary-w1"
    assert result.clean.loc[0, "candidate_hash"]
    assert result.clean.loc[0, "repair_status"] == RepairStatus.PENDING_RECHECK.value


def test_null_provider_records_no_evidence_and_never_fabricates() -> None:
    provider = NullRepairProvider()
    result = build_repair_overlay(
        _source(), provider, definition=_definition(), source_watermark_id="primary-w1"
    )

    assert provider.reasons == ["NO_EVIDENCE_BACKFILL"]
    assert result.clean.loc[:, ["ts_code", "value"]].to_dict("records") == [
        {"ts_code": "000002.SZ", "value": 4.0}
    ]
    assert result.quarantine["reason"].tolist() == ["NO_EVIDENCE_BACKFILL"]


def test_unknown_factor_uses_null_provider() -> None:
    provider = provider_for_factor("yeslab_999")
    assert isinstance(provider, NullRepairProvider)


def test_provider_mapping_rejects_injected_unknown_factor_ids() -> None:
    injected = FiniteProvider()
    provider = provider_for_factor("yeslab_999", {"yeslab_999": injected})
    assert isinstance(provider, NullRepairProvider)
    mapped = build_repair_provider_mapping(
        {"yeslab_999": injected}, factor_ids=("yeslab_999",)
    )
    assert isinstance(mapped["yeslab_999"], NullRepairProvider)


def test_market_provider_requires_secondary_watermark() -> None:
    with pytest.raises(ValueError, match="secondary watermark"):
        MarketCandidateProvider(pd.DataFrame())


def test_market_provider_delegates_reconciliation_and_adds_only_missing_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    primary = pd.DataFrame(
        {
            "trade_date": [date(2025, 1, 2)],
            "ts_code": ["000001.SZ"],
            "close": [10.0],
            "amount": [100.0],
            "pct_chg": [1.0],
        }
    )
    secondary = pd.DataFrame(
        {
            "trade_date": [date(2025, 1, 2), date(2025, 1, 3)],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "close": [10.0, 11.0],
            "amount": [100.0, 110.0],
            "pct_chg": [1.0, 1.0],
        }
    )
    from research_v2.factors import repair_overlay as module

    original_fill = module.fill_missing_market_rows
    original_reconcile = module.reconcile_market

    def fill(*args: object, **kwargs: object) -> pd.DataFrame:
        calls.append("fill")
        return original_fill(*args, **kwargs)

    def reconcile(*args: object, **kwargs: object) -> pd.DataFrame:
        calls.append("reconcile")
        return original_reconcile(*args, **kwargs)

    monkeypatch.setattr(module, "fill_missing_market_rows", fill)
    monkeypatch.setattr(module, "reconcile_market", reconcile)
    provider = MarketCandidateProvider(secondary, secondary_watermark_id="secondary-w1")
    result = provider.provide(primary)

    assert calls == ["fill", "reconcile"]
    assert result is None
    assert provider.last_market_frame is not None
    assert len(provider.last_market_frame) == 2


def test_known_formula_provider_rejects_overlap_mismatch() -> None:
    market = pd.DataFrame(
        {
            "trade_date": pd.date_range("2020-01-01", periods=60),
            "ts_code": ["000001.SZ"] * 60,
            "close": [10.0] * 60,
            "amount": [100.0] * 60,
            "pct_chg": [1.0] * 60,
        }
    )
    source = pd.DataFrame(
        {
            "trade_date": [date(2020, 2, 29)],
            "ts_code": ["000001.SZ"],
            "factor_name": ["GP_022"],
            "value": [999.0],
        }
    )
    provider = KnownFormulaCandidateProvider(market)

    with pytest.raises(ValueError, match="overlap mismatch"):
        provider.provide(source, factor_name="GP_022")


@pytest.mark.parametrize(
    ("factor_id", "factor_name"),
    [("yeslab_391", "GP_022"), ("yeslab_433", "GP_064")],
)
def test_known_formula_warmup_rows_are_dropped_without_quarantine(
    factor_id: str, factor_name: str
) -> None:
    dates = pd.bdate_range("2024-07-01", "2025-01-31")
    market = pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": ["000001.SZ"] * len(dates),
            "close": np.linspace(10.0, 20.0, len(dates)),
            "amount": np.linspace(100.0, 200.0, len(dates)),
            "pct_chg": [0.1] * len(dates),
        }
    )
    source = pd.DataFrame(
        {
            "trade_date": [date(2025, 1, 2)],
            "ts_code": ["000001.SZ"],
            "factor_name": [factor_name],
            "value": [np.nan],
        }
    )
    provider = KnownFormulaCandidateProvider(market)
    result = build_repair_overlay(
        source,
        provider,
        factor_id=factor_id,
        factor_name=factor_name,
        logical_partition="2025-01",
    )

    assert result.status is RepairStatus.PENDING_RECHECK
    assert result.filled_rows == 1
    assert result.quarantine.empty
    assert "WARMUP_OUT_OF_SCOPE" not in set(result.quarantine.get("reason", ()))


def test_known_formula_post_month_rows_remain_blocking() -> None:
    dates = pd.bdate_range("2024-07-01", "2025-02-03")
    market = pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": ["000001.SZ"] * len(dates),
            "close": np.linspace(10.0, 20.0, len(dates)),
            "amount": np.linspace(100.0, 200.0, len(dates)),
            "pct_chg": [0.1] * len(dates),
        }
    )
    source = pd.DataFrame(
        {
            "trade_date": [date(2025, 1, 2)],
            "ts_code": ["000001.SZ"],
            "factor_name": ["GP_022"],
            "value": [np.nan],
        }
    )

    result = build_repair_overlay(
        source,
        KnownFormulaCandidateProvider(market),
        factor_id="yeslab_391",
        factor_name="GP_022",
        logical_partition="2025-01",
    )

    assert result.status is RepairStatus.BLOCKED
    assert "DATE_SCOPE_VIOLATION" in set(result.quarantine["reason"])


@pytest.mark.parametrize(
    ("factor_id", "factor_name"),
    [("yeslab_391", "GP_022"), ("yeslab_433", "GP_064")],
)
def test_known_formula_all_nan_warmup_is_unresolved_not_blocking(
    factor_id: str, factor_name: str
) -> None:
    market = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-12-01", periods=10),
            "ts_code": ["000001.SZ"] * 10,
            "close": np.linspace(10.0, 11.0, 10),
            "amount": np.linspace(100.0, 110.0, 10),
            "pct_chg": [0.1] * 10,
        }
    )
    source = pd.DataFrame(
        {
            "trade_date": [date(2025, 1, 10)],
            "ts_code": ["000001.SZ"],
            "factor_name": [factor_name],
            "value": [np.nan],
        }
    )

    result = build_repair_overlay(
        source,
        KnownFormulaCandidateProvider(market),
        factor_id=factor_id,
        factor_name=factor_name,
        logical_partition="2025-01",
    )

    assert result.status is RepairStatus.PASS_WITH_QUARANTINE
    assert result.quarantine["reason"].tolist() == ["NO_FINITE_CANDIDATE"]
    assert result.decision.conflict_groups == 0


def test_known_formula_in_month_nonfinite_candidate_is_unresolved() -> None:
    market = pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=10),
            "ts_code": ["000001.SZ"] * 10,
            "close": np.linspace(10.0, 11.0, 10),
            "amount": np.linspace(100.0, 110.0, 10),
            "pct_chg": [0.1] * 10,
        }
    )
    source = pd.DataFrame(
        {
            "trade_date": [date(2025, 1, 10)],
            "ts_code": ["000001.SZ"],
            "factor_name": ["GP_022"],
            "value": [np.nan],
        }
    )

    result = build_repair_overlay(
        source,
        KnownFormulaCandidateProvider(market),
        factor_id="yeslab_391",
        factor_name="GP_022",
        logical_partition="2025-01",
    )

    assert result.status is RepairStatus.PASS_WITH_QUARANTINE
    assert result.quarantine["reason"].tolist() == ["NO_FINITE_CANDIDATE"]


def test_known_formula_nonfinite_candidate_does_not_quarantine_finite_source() -> None:
    market = pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=10),
            "ts_code": ["000001.SZ"] * 10,
            "close": np.linspace(10.0, 11.0, 10),
            "amount": np.linspace(100.0, 110.0, 10),
            "pct_chg": [0.1] * 10,
        }
    )
    source = pd.DataFrame(
        {
            "trade_date": [date(2025, 1, 10)],
            "ts_code": ["000001.SZ"],
            "factor_name": ["GP_022"],
            "value": [4.0],
        }
    )

    result = build_repair_overlay(
        source,
        KnownFormulaCandidateProvider(market),
        factor_id="yeslab_391",
        factor_name="GP_022",
        logical_partition="2025-01",
    )

    assert result.status is RepairStatus.PASS
    assert result.quarantine.empty
    assert result.clean["value"].tolist() == [4.0]


def test_overlay_quarantines_duplicate_candidate_keys() -> None:
    class DuplicateProvider:
        method = "DUPLICATE_CANDIDATE"

        def provide(self, source: pd.DataFrame, **_: object) -> pd.DataFrame:
            del source
            return pd.DataFrame(
                {
                    "trade_date": [date(2025, 1, 2), date(2025, 1, 2)],
                    "ts_code": ["000001.SZ", "000001.SZ"],
                    "candidate_value": [2.0, 3.0],
                }
            )

    result = build_repair_overlay(
        _source(),
        DuplicateProvider(),
        definition=_definition(),
        logical_partition="2025-01",
    )
    assert result.status is RepairStatus.BLOCKED
    assert set(result.quarantine["reason"]) == {"CONFLICT_QUARANTINE"}
    assert result.clean.loc[result.clean["ts_code"] == "000001.SZ"].empty


def test_overlay_quarantines_future_same_day_candidate() -> None:
    class FutureProvider:
        method = "FUTURE_CANDIDATE"

        def provide(self, source: pd.DataFrame, **_: object) -> pd.DataFrame:
            del source
            return pd.DataFrame(
                {
                    "trade_date": [date(2025, 1, 3)],
                    "ts_code": ["000001.SZ"],
                    "candidate_value": [2.0],
                }
            )

    result = build_repair_overlay(
        _source(),
        FutureProvider(),
        definition=_definition(),
        logical_partition="2025-01",
        signal_date=date(2025, 1, 2),
    )
    assert result.status is RepairStatus.BLOCKED
    assert result.quarantine["reason"].tolist() == ["NO_FINITE_CANDIDATE", "PIT_DATE_VIOLATION"]
    assert result.clean.loc[result.clean["ts_code"] == "000001.SZ"].empty


def test_overlay_quarantines_candidate_outside_factor_month() -> None:
    class WarmupProvider:
        method = "WARMUP_CANDIDATE"

        def provide(self, source: pd.DataFrame, **_: object) -> pd.DataFrame:
            del source
            return pd.DataFrame(
                {
                    "trade_date": [date(2024, 12, 31), date(2025, 1, 2)],
                    "ts_code": ["000003.SZ", "000001.SZ"],
                    "candidate_value": [9.0, 2.0],
                }
            )

    result = build_repair_overlay(
        _source(),
        WarmupProvider(),
        definition=_definition(),
        logical_partition="2025-01",
    )
    assert result.status is RepairStatus.PASS_WITH_QUARANTINE
    assert "WARMUP_OUT_OF_SCOPE" in set(result.quarantine["reason"])
    assert "000003.SZ" not in set(result.clean["ts_code"])
    assert "000001.SZ" in set(result.clean["ts_code"])


def test_warmup_only_scope_quarantine_promotes_pass_with_quarantine_status() -> None:
    class WarmupOnlyProvider:
        method = "WARMUP_ONLY"

        def provide(self, source: pd.DataFrame, **_: object) -> pd.DataFrame:
            del source
            return pd.DataFrame(
                {
                    "trade_date": [date(2024, 12, 31)],
                    "ts_code": ["000003.SZ"],
                    "candidate_value": [9.0],
                }
            )

    source = _source().assign(value=[1.0, 4.0])
    result = build_repair_overlay(
        source,
        WarmupOnlyProvider(),
        definition=_definition(),
        logical_partition="2025-01",
    )
    assert result.status is RepairStatus.PASS_WITH_QUARANTINE
    assert result.quarantine["reason"].tolist() == ["WARMUP_OUT_OF_SCOPE"]


def test_warmup_scope_quarantine_does_not_downgrade_in_month_conflict() -> None:
    class ConflictWithWarmupProvider:
        method = "CONFLICT_WITH_WARMUP"

        def provide(self, source: pd.DataFrame, **_: object) -> pd.DataFrame:
            del source
            return pd.DataFrame(
                {
                    "trade_date": [
                        date(2024, 12, 31),
                        date(2025, 1, 2),
                        date(2025, 1, 2),
                    ],
                    "ts_code": ["000003.SZ", "000001.SZ", "000001.SZ"],
                    "candidate_value": [9.0, 2.0, 3.0],
                }
            )

    result = build_repair_overlay(
        _source(),
        ConflictWithWarmupProvider(),
        definition=_definition(),
        logical_partition="2025-01",
    )
    assert result.status is RepairStatus.BLOCKED
    assert {"WARMUP_OUT_OF_SCOPE", "CONFLICT_QUARANTINE"}.issubset(
        set(result.quarantine["reason"])
    )


def test_candidate_hash_includes_rejected_warmup_rows() -> None:
    class CandidateProvider:
        method = "HASH_SCOPE"

        def __init__(self, include_warmup: bool) -> None:
            self.include_warmup = include_warmup

        def provide(self, source: pd.DataFrame, **_: object) -> pd.DataFrame:
            del source
            rows = [
                {
                    "trade_date": date(2025, 1, 2),
                    "ts_code": "000001.SZ",
                    "candidate_value": 2.0,
                }
            ]
            if self.include_warmup:
                rows.append(
                    {
                        "trade_date": date(2024, 12, 31),
                        "ts_code": "000001.SZ",
                        "candidate_value": 1.0,
                    }
                )
            return pd.DataFrame(rows)

    definition = _definition()
    in_month = build_repair_overlay(
        _source(),
        CandidateProvider(False),
        definition=definition,
        logical_partition="2025-01",
    )
    with_warmup = build_repair_overlay(
        _source(),
        CandidateProvider(True),
        definition=definition,
        logical_partition="2025-01",
    )
    assert in_month.candidate_hash != with_warmup.candidate_hash


def test_source_backfill_allows_pre_month_formula_warmup_and_materializes_in_month_fill() -> None:
    class WarmupProvider:
        method = "WARMUP_CANDIDATE"

        def provide(self, source: pd.DataFrame, **_: object) -> pd.DataFrame:
            del source
            return pd.DataFrame(
                {
                    "trade_date": [date(2024, 10, 1), date(2025, 1, 2)],
                    "ts_code": ["000099.SZ", "000001.SZ"],
                    "candidate_value": [9.0, 2.0],
                }
            )

    class Session:
        def query(self, _: str) -> object:
            return _source()

    definition = _definition()
    month = FactorMonth.from_partition("2025-01")
    context = build_factor_month_context(month, (definition,))
    batches = list(
        iter_clean_factor_batches(
            Session(),
            (definition,),
            month,
            context,
            repair_provider=WarmupProvider(),
        )
    )
    assert len(batches) == 1
    assert batches[0].frame.loc[:, "value"].tolist() == [2.0, 4.0]


def test_known_formula_warmup_rows_are_non_blocking_for_january_materialization() -> None:
    class Session:
        def query(self, _: str) -> object:
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "trade_date": [date(2025, 1, 2), date(2025, 1, 2)],
                    "factor_name": ["GP_022", "GP_022"],
                    "value": [np.nan, 4.0],
                }
            )

    market = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-09-25", periods=100),
            "ts_code": ["000001.SZ"] * 100,
            "close": [10.0] * 100,
            "amount": [100.0] * 100,
            "pct_chg": [1.0] * 100,
        }
    )
    provider = KnownFormulaCandidateProvider(market)
    definition = _definition()
    month = FactorMonth.from_partition("2025-01")
    context = build_factor_month_context(month, (definition,))
    batches = list(
        iter_clean_factor_batches(
            Session(),
            (definition,),
            month,
            context,
            repair_provider=provider,
        )
    )
    assert batches[0].frame.loc[:, "value"].tolist() == [100.0, 100.0, 4.0]
    assert batches[0].frame["signal_date"].min() == date(2025, 1, 1)


def test_source_backfill_rejects_post_month_candidate_before_publish() -> None:
    class FutureProvider:
        method = "FUTURE_CANDIDATE"

        def provide(self, source: pd.DataFrame, **_: object) -> pd.DataFrame:
            del source
            return pd.DataFrame(
                {
                    "trade_date": [date(2025, 2, 1)],
                    "ts_code": ["000003.SZ"],
                    "candidate_value": [9.0],
                }
            )

    class Session:
        def query(self, _: str) -> object:
            return _source()

    definition = _definition()
    month = FactorMonth.from_partition("2025-01")
    context = build_factor_month_context(month, (definition,))
    with pytest.raises(FactorBackfillError, match="repair overlay blocked"):
        list(
            iter_clean_factor_batches(
                Session(),
                (definition,),
                month,
                context,
                repair_provider=FutureProvider(),
            )
        )


def test_source_backfill_repair_hook_runs_before_materialization_and_keeps_business_columns(
) -> None:
    class Session:
        def query(self, _: str) -> object:
            return _source()

    definition = _definition()
    month = FactorMonth.from_partition("2025-01")
    context = build_factor_month_context(month, (definition,))
    baseline = list(iter_factor_materialization_batches(Session(), (definition,), month, context))
    repaired = list(
        iter_factor_materialization_batches(
            Session(),
            (definition,),
            month,
            context,
            repair_provider=FiniteProvider(),
        )
    )
    business = ["signal_date", "ts_code", "factor_id", "factor_version"]
    baseline_again = list(
        iter_factor_materialization_batches(Session(), (definition,), month, context)
    )
    assert baseline[0].clean_batch.frame.loc[:, business].equals(
        baseline_again[0].clean_batch.frame.loc[:, business]
    )
    assert len(repaired[0].clean_batch.frame.loc[:, business]) == 2
    assert repaired[0].clean_batch.frame.loc[
        repaired[0].clean_batch.frame["ts_code"] == "000001.SZ", "value"
    ].tolist() == [2.0]
