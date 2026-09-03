"""Tests for the walk-forward driver's transient-failure resilience.

The 2026-09-03 run 2 died silently mid-extraction on a transient fault;
run_query_with_retry and the fold dataset cache exist so a single fault
cannot destroy a multi-day run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "run_real_walk_forward.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_real_walk_forward", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


class _FlakySession:
    def __init__(self, failures: int) -> None:
        self.failures_left = failures
        self.queries: list[str] = []

    def run(self, script: str) -> str:
        self.queries.append(script)
        if self.failures_left > 0:
            self.failures_left -= 1
            raise RuntimeError("connection reset")
        return f"result:{script}"


def test_retry_recovers_after_transient_failure() -> None:
    session = _FlakySession(failures=2)
    connects: list[str] = []

    def connect() -> object:
        connects.append("reconnect")
        return session

    final_session, frame = MODULE.run_query_with_retry(
        connect, session, "select 1", attempts=5, base_delay=0.0
    )
    assert frame == "result:select 1"
    assert final_session is session
    assert len(connects) == 2  # reconnected after each failure


def test_retry_raises_after_exhaustion() -> None:
    session = _FlakySession(failures=99)
    with pytest.raises(RuntimeError, match="query failed after 3 attempts"):
        MODULE.run_query_with_retry(
            lambda: session, session, "select 1", attempts=3, base_delay=0.0
        )


def test_retry_does_not_retry_on_success() -> None:
    session = _FlakySession(failures=0)
    _, frame = MODULE.run_query_with_retry(
        lambda: session, session, "select 42", attempts=5, base_delay=0.0
    )
    assert frame == "result:select 42"
    assert session.queries == ["select 42"]


def test_fold_cache_key_is_deterministic_and_version_sensitive() -> None:
    versions_a = {"label_daily": {"2020.01M": "v1"}, "universe_daily": {"2020.01M": "u1"}}
    versions_b = {"label_daily": {"2020.01M": "v2"}, "universe_daily": {"2020.01M": "u1"}}
    key1 = MODULE.fold_cache_key("F01", "train", versions_a)
    key2 = MODULE.fold_cache_key("F01", "train", dict(versions_a))
    assert key1 == key2
    assert MODULE.fold_cache_key("F01", "train", versions_b) != key1
    assert MODULE.fold_cache_key("F01", "validation", versions_a) != key1
    assert MODULE.fold_cache_key("F02", "train", versions_a) != key1
