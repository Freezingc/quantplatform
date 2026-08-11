from __future__ import annotations

from dataclasses import replace

import pytest

from research_v2.contracts.factor import (
    AvailabilityPolicy,
    FactorDefinition,
    FactorStatus,
    FactorType,
)
from research_v2.factors.registry import FactorRegistry, ImmutableDefinitionError


def _definition() -> FactorDefinition:
    return FactorDefinition(
        factor_id="turnover_20d",
        factor_version=1,
        display_name="20-day turnover",
        source_database="dfs://YesLab",
        source_table="factor_data1",
        source_value_column="turnover_20d",
        source_date_column="trade_date",
        source_code_column="ts_code",
        factor_type=FactorType.TECHNICAL,
        frequency="DAILY",
        expected_direction=1,
        availability_policy=AvailabilityPolicy.SAME_DAY_CLOSE,
        max_staleness_calendar_days=0,
        unit="ratio",
        owner="research",
    )


def test_definition_change_requires_a_new_factor_version() -> None:
    registry = FactorRegistry()
    definition = _definition()

    registry.register(definition)

    with pytest.raises(ImmutableDefinitionError, match="definition hash"):
        registry.register(replace(definition, unit="percent"))


def test_metadata_changes_do_not_change_definition_hash_or_active_state() -> None:
    registry = FactorRegistry()
    definition = _definition()

    registry.register(definition)
    registry.transition("turnover_20d", 1, FactorStatus.VALIDATED)
    registry.transition("turnover_20d", 1, FactorStatus.ACTIVE)

    active = registry.get("turnover_20d", 1)
    assert active.status is FactorStatus.ACTIVE
    revised_metadata = replace(definition, display_name="Turnover", owner="other")
    assert active.definition_hash == revised_metadata.definition_hash


def test_technical_factor_rejects_staleness_and_non_yeslab_source() -> None:
    with pytest.raises(ValueError, match="staleness"):
        replace(_definition(), max_staleness_calendar_days=1)
    with pytest.raises(ValueError, match="YesLab"):
        replace(_definition(), source_database="dfs://Other")
