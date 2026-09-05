from __future__ import annotations

import pytest

from research_v2.factors.factor_sets import (
    FactorSetStatus,
    FactorSetTransitionError,
    create_factor_set,
    transition_factor_set,
)


def _members() -> list[tuple[str, int, int, str]]:
    return [
        ("yeslab_2", 1, -1, "hash-2"),
        ("yeslab_1", 1, 1, "hash-1"),
    ]


def test_factor_set_id_and_feature_order_are_invariant_to_input_order() -> None:
    first = create_factor_set(_members(), selection_scope_id="fold-1", rules_version=1)
    second = create_factor_set(
        list(reversed(_members())),
        selection_scope_id="fold-1",
        rules_version=1,
    )

    assert first.factor_set_id == second.factor_set_id
    assert [member.factor_id for member in first.members] == ["yeslab_1", "yeslab_2"]
    assert [member.feature_order for member in first.members] == [1, 2]


def test_factor_set_rejects_multiple_versions_of_the_same_factor() -> None:
    with pytest.raises(ValueError, match="multiple versions"):
        create_factor_set(
            [("yeslab_1", 1, 1, "hash-1"), ("yeslab_1", 2, 1, "hash-2")],
            selection_scope_id="fold-1",
            rules_version=1,
        )


def test_factor_set_transitions_to_frozen_and_cannot_reopen() -> None:
    factor_set = create_factor_set(_members(), selection_scope_id="fold-1", rules_version=1)
    validated = transition_factor_set(factor_set, FactorSetStatus.VALIDATED)
    frozen = transition_factor_set(validated, FactorSetStatus.FROZEN)

    assert frozen.status is FactorSetStatus.FROZEN
    with pytest.raises(FactorSetTransitionError, match="FROZEN"):
        transition_factor_set(frozen, FactorSetStatus.VALIDATED)


# --- the C8.1 / C8.2 clauses the three tests above did not assert ----------------


def test_draft_cannot_skip_validation_to_reach_frozen() -> None:
    """VALIDATED is a required step, so a set cannot be frozen straight from DRAFT."""
    draft = create_factor_set(_members(), selection_scope_id="fold-1", rules_version=1)

    with pytest.raises(FactorSetTransitionError, match="DRAFT -> FROZEN"):
        transition_factor_set(draft, FactorSetStatus.FROZEN)


@pytest.mark.parametrize(
    "start",
    [FactorSetStatus.DRAFT, FactorSetStatus.VALIDATED, FactorSetStatus.FROZEN],
)
def test_retired_is_reachable_from_every_prior_state_and_is_terminal(
    start: FactorSetStatus,
) -> None:
    """C8.1 names RETIRED as the fourth state; without this it was untested."""
    factor_set = create_factor_set(_members(), selection_scope_id="fold-1", rules_version=1)
    if start is FactorSetStatus.VALIDATED:
        factor_set = transition_factor_set(factor_set, FactorSetStatus.VALIDATED)
    elif start is FactorSetStatus.FROZEN:
        factor_set = transition_factor_set(
            transition_factor_set(factor_set, FactorSetStatus.VALIDATED),
            FactorSetStatus.FROZEN,
        )

    retired = transition_factor_set(factor_set, FactorSetStatus.RETIRED)

    assert retired.status is FactorSetStatus.RETIRED
    for target in FactorSetStatus:
        with pytest.raises(FactorSetTransitionError):
            transition_factor_set(retired, target)


def test_freezing_preserves_members_order_and_versions_exactly() -> None:
    """C8.1: after FROZEN the members, their order and their versions are immutable."""
    draft = create_factor_set(_members(), selection_scope_id="fold-1", rules_version=1)

    frozen = transition_factor_set(
        transition_factor_set(draft, FactorSetStatus.VALIDATED), FactorSetStatus.FROZEN
    )

    assert frozen.members == draft.members
    assert frozen.factor_set_id == draft.factor_set_id
    assert frozen.content_hash == draft.content_hash
    assert [member.feature_order for member in frozen.members] == [1, 2]


@pytest.mark.parametrize(
    "changed",
    [
        [("yeslab_9", 1, -1, "hash-2"), ("yeslab_1", 1, 1, "hash-1")],   # factor_id
        [("yeslab_2", 2, -1, "hash-2"), ("yeslab_1", 1, 1, "hash-1")],   # factor_version
        [("yeslab_2", 1, 1, "hash-2"), ("yeslab_1", 1, 1, "hash-1")],    # frozen_direction
        [("yeslab_2", 1, -1, "hash-X"), ("yeslab_1", 1, 1, "hash-1")],   # definition_hash
        [("yeslab_2", 1, -1, "hash-2")],                                 # a member removed
        [
            ("yeslab_2", 1, -1, "hash-2"),
            ("yeslab_1", 1, 1, "hash-1"),
            ("yeslab_3", 1, 0, "hash-3"),                                # a member added
        ],
    ],
)
def test_changing_any_member_creates_a_new_factor_set_id(
    changed: list[tuple[str, int, int, str]],
) -> None:
    """C8.1: the id is content-addressed, so any membership change is a different set.

    This is what makes a re-adjudicated factor set -- adjudication 4 option (iii), which
    drops six of the frozen seventeen -- a new identity rather than a mutation of the old
    one, so evidence signed against the old id stays attributable to it.
    """
    original = create_factor_set(_members(), selection_scope_id="fold-1", rules_version=1)

    changed_set = create_factor_set(changed, selection_scope_id="fold-1", rules_version=1)

    assert changed_set.factor_set_id != original.factor_set_id


@pytest.mark.parametrize(
    ("scope", "rules"),
    [("fold-2", 1), ("fold-1", 2)],
)
def test_selection_scope_and_rules_version_are_part_of_the_identity(scope: str, rules: int) -> None:
    """C8.2 names both as hash inputs; a set selected under different rules is a different set."""
    original = create_factor_set(_members(), selection_scope_id="fold-1", rules_version=1)

    other = create_factor_set(_members(), selection_scope_id=scope, rules_version=rules)

    assert other.factor_set_id != original.factor_set_id


def test_a_member_definition_hash_change_alone_changes_the_id() -> None:
    """The registry definition hash is folded in, so a redefined factor cannot reuse a set.

    Same members, same order, same scope and rules -- only the definition behind one factor
    moved. Without the definition hash in the payload, a factor could be redefined in place
    and every artifact signed against the old set id would silently describe new data.
    """
    original = create_factor_set(_members(), selection_scope_id="fold-1", rules_version=1)

    redefined = create_factor_set(
        [("yeslab_2", 1, -1, "hash-2-redefined"), ("yeslab_1", 1, 1, "hash-1")],
        selection_scope_id="fold-1",
        rules_version=1,
    )

    assert redefined.factor_set_id != original.factor_set_id
    assert [m.factor_id for m in redefined.members] == [m.factor_id for m in original.members]
