"""Fail-closed data-quality publishing decisions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class Severity(StrEnum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    BLOCKER = "BLOCKER"


class CheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"


@dataclass(frozen=True)
class CheckResult:
    check_name: str
    severity: Severity
    status: CheckStatus


def can_publish(
    results: Iterable[CheckResult],
    *,
    required_blocking_checks: frozenset[str] = frozenset(),
) -> bool:
    """Return whether a non-empty DQ run satisfies all publish gate invariants."""
    materialized = list(results)
    if not materialized or any(result.status is CheckStatus.ERROR for result in materialized):
        return False
    by_name = {result.check_name: result for result in materialized}
    if len(by_name) != len(materialized):
        return False
    has_failed_blocker = any(
        result.severity is Severity.BLOCKER and result.status is not CheckStatus.PASS
        for result in materialized
    )
    if has_failed_blocker:
        return False
    missing_required_blocker = any(
        name not in by_name or by_name[name].status is not CheckStatus.PASS
        for name in required_blocking_checks
    )
    return not missing_required_blocker
