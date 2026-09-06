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


def _definition_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = dict(
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
    kwargs.update(overrides)
    return kwargs


def _definition() -> FactorDefinition:
    return FactorDefinition(**_definition_kwargs())  # type: ignore[arg-type]


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


# --- C1.2's enumerated rejections that had no test -------------------------------


def test_an_unknown_factor_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="factor_type"):
        FactorDefinition(**_definition_kwargs(factor_type="BOGUS"))  # type: ignore[arg-type]


def test_a_string_factor_type_cannot_bypass_the_technical_staleness_rule() -> None:
    """``factor_type is FactorType.TECHNICAL`` is False for the equal *string*.

    A YAML catalog stores ``factor_type: TECHNICAL`` as a string, so a loader that forgets
    to coerce registers a technical factor carrying 30 days of staleness -- and nothing
    downstream notices, because the definition looks entirely valid. Measured before the
    fix: accepted, with ``max_staleness_calendar_days=30``.
    """
    with pytest.raises(ValueError, match="FactorType"):
        FactorDefinition(  # type: ignore[arg-type]
            **_definition_kwargs(factor_type="TECHNICAL", max_staleness_calendar_days=30)
        )


def test_string_enums_cannot_bypass_the_fundamental_availability_rule() -> None:
    """The look-ahead case, and the most consequential of the bypasses.

    Availability is what keeps an announcement out of the window before it existed. A
    fundamental factor accepted as SAME_DAY_CLOSE is usable on its announcement date, i.e.
    before the market could have reacted to it. Both fields have to be strings for the
    bypass to happen -- with a real ``FactorType.FUNDAMENTAL`` the existing check still
    fires -- which is why this case is written with both as strings: that is what a YAML
    loader produces. Measured before the fix: accepted. The factor_type check fires first,
    so that is the message; the next test pins the availability_policy check on its own.
    """
    with pytest.raises(ValueError, match="FactorType"):
        FactorDefinition(  # type: ignore[arg-type]
            **_definition_kwargs(
                factor_type="FUNDAMENTAL",
                availability_policy="SAME_DAY_CLOSE",
                max_staleness_calendar_days=180,
            )
        )


def test_a_string_availability_policy_is_rejected_even_with_a_valid_factor_type() -> None:
    """Isolates the second membership check, which the case above cannot reach."""
    with pytest.raises(ValueError, match="AvailabilityPolicy"):
        FactorDefinition(  # type: ignore[arg-type]
            **_definition_kwargs(
                factor_type=FactorType.FUNDAMENTAL,
                availability_policy="NEXT_TRADING_DAY_AFTER_ANNOUNCEMENT_X",
                max_staleness_calendar_days=180,
            )
        )


def test_a_fundamental_factor_without_announcement_availability_is_rejected() -> None:
    with pytest.raises(ValueError, match="announcement-based"):
        FactorDefinition(
            **_definition_kwargs(
                factor_type=FactorType.FUNDAMENTAL,
                availability_policy=AvailabilityPolicy.SAME_DAY_CLOSE,
                max_staleness_calendar_days=180,
            )
        )


def test_a_definition_without_an_owner_is_rejected() -> None:
    with pytest.raises(ValueError, match="owner"):
        FactorDefinition(**_definition_kwargs(owner=""))


def test_registering_an_identical_definition_twice_returns_the_same_entry() -> None:
    """Duplicate (id, version) with an unchanged definition is idempotent, not an error.

    The error case is a *changed* definition under the same identity, which
    ``test_definition_change_requires_a_new_factor_version`` covers. Returning the existing
    entry rather than raising is what makes registration re-runnable.
    """
    registry = FactorRegistry()

    first = registry.register(_definition())
    second = registry.register(_definition())

    assert second is first


# --- C1.4: the lifecycle itself, which nothing above exercised beyond one happy path ------------


_LIFECYCLE_PATH = {
    FactorStatus.DRAFT: (),
    FactorStatus.VALIDATED: (FactorStatus.VALIDATED,),
    FactorStatus.ACTIVE: (FactorStatus.VALIDATED, FactorStatus.ACTIVE),
    FactorStatus.RETIRED: (FactorStatus.VALIDATED, FactorStatus.ACTIVE, FactorStatus.RETIRED),
}


def _registry_at(status: FactorStatus) -> FactorRegistry:
    """A registry whose single factor has been walked to ``status`` by legal transitions only."""
    registry = FactorRegistry()
    registry.register(_definition())
    for step in _LIFECYCLE_PATH[status]:
        registry.transition("turnover_20d", 1, step)
    assert registry.get("turnover_20d", 1).status is status
    return registry


def test_a_newly_registered_factor_starts_in_draft() -> None:
    """Nothing is usable by accident: a factor must be walked to ACTIVE to be relied on."""
    registry = FactorRegistry()

    entry = registry.register(_definition())

    assert entry.status is FactorStatus.DRAFT


@pytest.mark.parametrize(
    ("start", "attempted"),
    [
        (FactorStatus.DRAFT, FactorStatus.ACTIVE),
        (FactorStatus.DRAFT, FactorStatus.DRAFT),
        (FactorStatus.VALIDATED, FactorStatus.DRAFT),
        (FactorStatus.VALIDATED, FactorStatus.VALIDATED),
        (FactorStatus.ACTIVE, FactorStatus.VALIDATED),
        (FactorStatus.ACTIVE, FactorStatus.DRAFT),
        (FactorStatus.ACTIVE, FactorStatus.ACTIVE),
        (FactorStatus.RETIRED, FactorStatus.DRAFT),
        (FactorStatus.RETIRED, FactorStatus.VALIDATED),
        (FactorStatus.RETIRED, FactorStatus.ACTIVE),
        (FactorStatus.RETIRED, FactorStatus.RETIRED),
    ],
)
def test_the_lifecycle_only_moves_forward(start: FactorStatus, attempted: FactorStatus) -> None:
    """C1.4's DRAFT -> VALIDATED -> ACTIVE -> RETIRED is one-way.

    Skipping VALIDATED would let a factor reach ACTIVE without ever having been checked, and any
    backward edge would let a definition that is already feeding models be silently re-opened.
    Re-transitioning to the current state is refused too: it is not a no-op, it is a caller that
    believes it is somewhere else.
    """
    registry = _registry_at(start)

    with pytest.raises(ValueError, match="invalid factor state transition"):
        registry.transition("turnover_20d", 1, attempted)

    assert registry.get("turnover_20d", 1).status is start, (
        "a refused transition must leave the entry where it was"
    )


@pytest.mark.parametrize(
    "start", [FactorStatus.DRAFT, FactorStatus.VALIDATED, FactorStatus.ACTIVE]
)
def test_any_live_state_may_be_retired_directly(start: FactorStatus) -> None:
    """Retirement is the escape hatch, so it must not require walking the rest of the lifecycle."""
    registry = _registry_at(start)

    entry = registry.transition("turnover_20d", 1, FactorStatus.RETIRED)

    assert entry.status is FactorStatus.RETIRED


def test_retired_is_terminal() -> None:
    registry = _registry_at(FactorStatus.RETIRED)

    for status in FactorStatus:
        with pytest.raises(ValueError, match="invalid factor state transition"):
            registry.transition("turnover_20d", 1, status)


def test_a_definition_cannot_be_changed_after_validation() -> None:
    """C1.4: once VALIDATED the semantic fields are closed; a change means a new factor_version."""
    registry = _registry_at(FactorStatus.VALIDATED)
    before = registry.get("turnover_20d", 1)

    with pytest.raises(ImmutableDefinitionError, match="definition hash"):
        registry.register(replace(_definition(), unit="percent"))

    after = registry.get("turnover_20d", 1)
    assert after.definition_hash == before.definition_hash
    assert after.status is FactorStatus.VALIDATED, (
        "a refused redefinition must not disturb the state either"
    )
    assert after.definition.unit == "ratio"


def test_a_definition_change_is_refused_even_before_validation() -> None:
    """Pins that the implementation is *stricter* than C1.4, deliberately.

    The spec closes the definition at VALIDATED; ``register`` refuses a changed hash from DRAFT
    onward. Recording the actual behaviour matters because it is the difference between "a draft
    can be edited in place" and "it cannot" -- and a draft that could be edited would make the
    definition hash an unstable key for anything that references it early.
    """
    registry = FactorRegistry()
    registry.register(_definition())

    with pytest.raises(ImmutableDefinitionError):
        registry.register(replace(_definition(), expected_direction=-1))


def test_transitioning_a_factor_that_was_never_registered_is_an_error() -> None:
    registry = FactorRegistry()

    with pytest.raises(KeyError, match="not registered"):
        registry.transition("turnover_20d", 1, FactorStatus.VALIDATED)


def test_a_new_version_lives_alongside_the_old_one_and_has_its_own_lifecycle() -> None:
    """C1.4's remedy for a needed change: a new version, not an edit -- and not a replacement."""
    registry = FactorRegistry()
    registry.register(_definition())
    registry.transition("turnover_20d", 1, FactorStatus.VALIDATED)
    registry.register(replace(_definition(), factor_version=2))

    assert registry.get("turnover_20d", 1).status is FactorStatus.VALIDATED
    assert registry.get("turnover_20d", 2).status is FactorStatus.DRAFT
    assert registry.get("turnover_20d", 1).definition_hash != (
        registry.get("turnover_20d", 2).definition_hash
    )
