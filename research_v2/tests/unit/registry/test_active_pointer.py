from __future__ import annotations

from research_v2.registry.active_pointer import ActiveModelPointerStore


def test_active_pointer_compare_and_swap_prevents_stale_overwrite() -> None:
    store = ActiveModelPointerStore()
    first = store.compare_and_swap(
        consumer_id="daily-shadow",
        environment="shadow",
        expected_version=0,
        model_id="model-a",
        model_version=1,
    )
    stale = store.compare_and_swap(
        consumer_id="daily-shadow",
        environment="shadow",
        expected_version=0,
        model_id="model-b",
        model_version=1,
    )
    second = store.compare_and_swap(
        consumer_id="daily-shadow",
        environment="shadow",
        expected_version=1,
        model_id="model-b",
        model_version=1,
    )

    assert first is not None
    assert stale is None
    assert second is not None
    assert store.get("daily-shadow", "shadow") == second
