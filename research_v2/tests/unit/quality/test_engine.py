from __future__ import annotations

from research_v2.quality.engine import CheckResult, CheckStatus, Severity, can_publish


def test_publish_requires_nonempty_passing_blocking_checks() -> None:
    assert not can_publish([])
    assert can_publish(
        [
            CheckResult("keys_unique", Severity.BLOCKER, CheckStatus.PASS),
            CheckResult("coverage", Severity.WARN, CheckStatus.FAIL),
        ]
    )


def test_blocker_failure_or_execution_error_prevents_publish() -> None:
    assert not can_publish(
        [CheckResult("keys_unique", Severity.BLOCKER, CheckStatus.FAIL)]
    )
    assert not can_publish(
        [CheckResult("schema", Severity.ERROR, CheckStatus.ERROR)]
    )
