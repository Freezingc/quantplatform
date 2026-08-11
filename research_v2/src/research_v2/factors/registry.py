"""In-memory immutable registry for versioned factor definitions."""

from __future__ import annotations

from dataclasses import dataclass

from research_v2.contracts.factor import FactorDefinition, FactorStatus


class ImmutableDefinitionError(ValueError):
    """Raised when an existing factor version changes semantic definition."""


@dataclass(frozen=True)
class RegisteredFactor:
    definition: FactorDefinition
    status: FactorStatus

    @property
    def definition_hash(self) -> str:
        return self.definition.definition_hash


class FactorRegistry:
    """Fail-closed registry that makes factor semantic changes versioned."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, int], RegisteredFactor] = {}

    def register(self, definition: FactorDefinition) -> RegisteredFactor:
        key = (definition.factor_id, definition.factor_version)
        prior = self._entries.get(key)
        if prior is not None:
            if prior.definition_hash != definition.definition_hash:
                raise ImmutableDefinitionError(
                    "factor definition hash changed; create a new version"
                )
            return prior
        entry = RegisteredFactor(definition=definition, status=FactorStatus.DRAFT)
        self._entries[key] = entry
        return entry

    def transition(
        self,
        factor_id: str,
        factor_version: int,
        status: FactorStatus,
    ) -> RegisteredFactor:
        key = (factor_id, factor_version)
        current = self.get(factor_id, factor_version)
        allowed = {
            FactorStatus.DRAFT: {FactorStatus.VALIDATED, FactorStatus.RETIRED},
            FactorStatus.VALIDATED: {FactorStatus.ACTIVE, FactorStatus.RETIRED},
            FactorStatus.ACTIVE: {FactorStatus.RETIRED},
            FactorStatus.RETIRED: set(),
        }
        if status not in allowed[current.status]:
            raise ValueError(f"invalid factor state transition {current.status} -> {status}")
        updated = RegisteredFactor(definition=current.definition, status=status)
        self._entries[key] = updated
        return updated

    def get(self, factor_id: str, factor_version: int) -> RegisteredFactor:
        key = (factor_id, factor_version)
        if key not in self._entries:
            raise KeyError(f"factor is not registered: {factor_id}:{factor_version}")
        return self._entries[key]
