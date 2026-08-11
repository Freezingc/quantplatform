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
