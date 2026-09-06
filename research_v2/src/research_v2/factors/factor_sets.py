"""Content-addressed immutable factor-set definitions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import StrEnum


class FactorSetStatus(StrEnum):
    DRAFT = "DRAFT"
    VALIDATED = "VALIDATED"
    FROZEN = "FROZEN"
    RETIRED = "RETIRED"


class FactorSetTransitionError(ValueError):
    """Raised when an immutable factor-set state transition is invalid."""


@dataclass(frozen=True)
class FactorSetMember:
    factor_id: str
    factor_version: int
    frozen_direction: int
    definition_hash: str
    feature_order: int


@dataclass(frozen=True)
class FactorSet:
    factor_set_id: str
    selection_scope_id: str
    rules_version: int
    members: tuple[FactorSetMember, ...]
    content_hash: str
    status: FactorSetStatus


def create_factor_set(
    members: list[tuple[str, int, int, str]],
    *,
    selection_scope_id: str,
    rules_version: int,
) -> FactorSet:
    """Build a deterministic DRAFT factor set from member identities the caller asserts.

    Nothing here is verified against a registry. ``members`` is a list of bare
    ``(factor_id, factor_version, frozen_direction, definition_hash)`` tuples and the checks
    below are shape checks: one version per factor, a positive version, a direction in
    {-1, 0, 1}, a non-empty hash. No ``FactorRegistry`` is consulted, so neither "this factor
    is ACTIVE" nor "this definition hash is the registry's" is established by a successful
    call -- both remain the caller's assertion.

    That is the honest scope of the content addressing. ``content_hash``, and therefore
    ``factor_set_id``, is a deterministic function of what the caller supplied, which makes
    the set immutable and independent of input order; it does not make the set
    *authoritative*. Building sets from registry-verified identities is an open decision, and
    an earlier version of this docstring claimed it was already settled by saying the members
    came from the active registry. The production frozen set in use today is a
    hand-adjudicated JSON artifact that never passed through this function.
    """
    if not members or not selection_scope_id or rules_version < 1:
        raise ValueError("factor set requires members, selection scope, and positive rules version")
    factor_ids = [member[0] for member in members]
    if len(set(factor_ids)) != len(factor_ids):
        raise ValueError("factor set cannot contain multiple versions of the same factor")
    has_invalid_member = any(
        version < 1 or direction not in {-1, 0, 1} or not definition_hash
        for _, version, direction, definition_hash in members
    )
    if has_invalid_member:
        raise ValueError("factor set member version, direction, and definition hash are invalid")
    ordered_raw = sorted(members, key=lambda member: (member[0], member[1]))
    ordered_members = tuple(
        FactorSetMember(
            factor_id=factor_id,
            factor_version=factor_version,
            frozen_direction=direction,
            definition_hash=definition_hash,
            feature_order=index,
        )
        for index, (factor_id, factor_version, direction, definition_hash) in enumerate(
            ordered_raw,
            start=1,
        )
    )
    payload = {
        "members": [
            {
                "definition_hash": member.definition_hash,
                "factor_id": member.factor_id,
                "factor_version": member.factor_version,
                "frozen_direction": member.frozen_direction,
            }
            for member in ordered_members
        ],
        "rules_version": rules_version,
        "selection_scope_id": selection_scope_id,
    }
    content_hash = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return FactorSet(
        factor_set_id=content_hash,
        selection_scope_id=selection_scope_id,
        rules_version=rules_version,
        members=ordered_members,
        content_hash=content_hash,
        status=FactorSetStatus.DRAFT,
    )


def transition_factor_set(factor_set: FactorSet, status: FactorSetStatus) -> FactorSet:
    """Apply the one-way lifecycle that protects frozen feature order and versions."""
    allowed = {
        FactorSetStatus.DRAFT: {FactorSetStatus.VALIDATED, FactorSetStatus.RETIRED},
        FactorSetStatus.VALIDATED: {FactorSetStatus.FROZEN, FactorSetStatus.RETIRED},
        FactorSetStatus.FROZEN: {FactorSetStatus.RETIRED},
        FactorSetStatus.RETIRED: set(),
    }
    if status not in allowed[factor_set.status]:
        raise FactorSetTransitionError(
            f"invalid factor set transition {factor_set.status} -> {status}"
        )
    return replace(factor_set, status=status)
