"""Compare-and-swap active model pointers for safe consumer switching and rollback."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActiveModelPointer:
    consumer_id: str
    environment: str
    model_id: str
    model_version: int
    pointer_version: int


class ActiveModelPointerStore:
    """Domain-level CAS store; database adapters must implement the same atomic condition."""

    def __init__(self) -> None:
        self._pointers: dict[tuple[str, str], ActiveModelPointer] = {}

    def get(self, consumer_id: str, environment: str) -> ActiveModelPointer | None:
        return self._pointers.get((consumer_id, environment))

    def compare_and_swap(
        self,
        *,
        consumer_id: str,
        environment: str,
        expected_version: int,
        model_id: str,
        model_version: int,
    ) -> ActiveModelPointer | None:
        """Update only after a matching version read; return None if another writer won."""
        invalid_identity = not consumer_id or not environment or not model_id
        invalid_version = expected_version < 0 or model_version < 1
        if invalid_identity or invalid_version:
            raise ValueError("active pointer identity or version is invalid")
        key = (consumer_id, environment)
        current = self._pointers.get(key)
        current_version = current.pointer_version if current is not None else 0
        if current_version != expected_version:
            return None
        updated = ActiveModelPointer(
            consumer_id=consumer_id,
            environment=environment,
            model_id=model_id,
            model_version=model_version,
            pointer_version=current_version + 1,
        )
        self._pointers[key] = updated
        return updated
