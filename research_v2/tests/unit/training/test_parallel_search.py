from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from research_v2.training.cross_section_search import (
    CrossSectionConfigTask,
    CrossSectionSeedTask,
    run_cross_section_config_task,
    run_cross_section_seed_task,
)
from research_v2.training.ensemble import REQUIRED_SEEDS, build_seed_ensemble
from research_v2.training.parallel_search import (
    SearchTask,
    capped_worker_count,
    granularity_makespans,
    run_parallel_search,
    search_granularity,
)
from research_v2.training.static_batches import build_static_training_batches


def _delayed_result(payload: Mapping[str, object]) -> dict[str, object]:
    """Top-level and importable so a spawned worker executes real scheduler work."""
    time.sleep(float(payload["delay_seconds"]))
    return {"value": int(payload["value"])}


def _raising_result(payload: Mapping[str, object]) -> dict[str, object]:
    raise RuntimeError(f"worker failed for {payload['value']}")


def test_parallel_search_restores_grid_order_after_reverse_completion() -> None:
    """Removing index sorting would select/tie-break based on scheduler timing."""
    tasks = (
        SearchTask(index=0, payload={"delay_seconds": 0.05, "value": 10}),
        SearchTask(index=1, payload={"delay_seconds": 0.0, "value": 20}),
    )

    results = run_parallel_search(tasks, workers=2, run_task=_delayed_result)

    assert results == [(0, {"value": 10}), (1, {"value": 20})]


def test_parallel_search_caps_requested_workers_to_the_available_tasks() -> None:
    """Removing the cap would start idle heavyweight Torch workers."""
    assert capped_worker_count(task_count=1, requested_workers=48) == 1


@pytest.mark.parametrize(
    ("configs", "workers", "expected", "why"),
    [
        pytest.param(24, 20, "seed", "24 configs over 20 workers leaves a 4-config second round",
                     id="m2_grid_memory_capped_workers"),
        pytest.param(24, 24, "config", "one round either way; config tasks are fewer and smaller",
                     id="grid_exactly_fills_workers"),
        pytest.param(
            24, 48, "seed", "surplus workers idle under 3-fit tasks: 2 rounds beats 1x3",
            id="more_workers_than_configs",
        ),
        pytest.param(24, 72, "seed", "one round either way, but a seed task is one fit not three",
                     id="surplus_workers_one_round_each"),
        pytest.param(4, 4, "config", "tie at three fit-times goes to the coarser task",
                     id="tie_at_four_workers"),
        pytest.param(1, 3, "seed", "a one-config grid cannot fill three workers at config level",
                     id="single_config_three_workers"),
        pytest.param(1, 1, "config", "one worker, one config: nothing to split",
                     id="single_config_single_worker"),
        pytest.param(2, 3, "seed", "six fits over three workers is two rounds, not two of three",
                     id="two_configs_three_workers"),
        pytest.param(3, 3, "config", "tie goes to the coarser granularity",
                     id="tie_goes_to_config"),
    ],
)
def test_the_granularity_choice_minimises_makespan_not_worker_occupancy(
    configs: int, workers: int, expected: str, why: str
) -> None:
    """A config task is three fits and cannot be split, so filling the workers is the wrong test.

    The measured case is M2: 24 configs, and 20 workers because F4's window panels put each worker
    at ~21.6 GiB. Config-level runs ceil(24/20)=2 rounds of three fits = six fit-times; seed-level
    runs ceil(72/20)=4 rounds of one = four fit-times, a third less wall clock on a multi-day run.
    """
    assert search_granularity(
        config_count=configs, workers=workers, seeds_per_config=3
    ) == expected, why


def test_the_granularity_makespan_arithmetic_is_exposed_not_hidden() -> None:
    """The two candidate makespans are part of the contract, so a change is reviewable."""
    assert granularity_makespans(config_count=24, workers=20, seeds_per_config=3) == (6, 4)
    assert granularity_makespans(config_count=24, workers=24, seeds_per_config=3) == (3, 3)
    assert granularity_makespans(config_count=1, workers=3, seeds_per_config=3) == (3, 1)


@pytest.mark.parametrize(
    ("configs", "workers", "seeds"),
    [
        pytest.param(0, 4, 3, id="no_configs"),
        pytest.param(4, 0, 3, id="no_workers"),
        pytest.param(4, 4, 0, id="no_seeds"),
    ],
)
def test_a_degenerate_granularity_request_is_refused(
    configs: int, workers: int, seeds: int
) -> None:
    with pytest.raises(ValueError):
        search_granularity(config_count=configs, workers=workers, seeds_per_config=seeds)
def test_parallel_search_propagates_a_worker_exception() -> None:
    """Swallowing a worker failure would let the parent choose from a partial grid."""
    with pytest.raises(RuntimeError, match="worker failed for 10"):
        run_parallel_search(
            (SearchTask(index=0, payload={"value": 10}),),
            workers=1,
            run_task=_raising_result,
        )


def test_spawned_cross_section_worker_returns_a_complete_three_seed_result(tmp_path) -> None:
    """A package-level worker must survive spawn, not merely work in the parent process."""
    def frame(day: date) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "signal_date": [day] * 30,
                "ts_code": [f"S{index:03d}" for index in range(30)],
                "raw_return_5d": [float(index) for index in range(30)],
                "label_valid": [True] * 30,
                "f00": [float(index) for index in range(30)],
            }
        )

    train = frame(date(2020, 1, 2))
    validation = frame(date(2020, 1, 3))
    task = CrossSectionConfigTask(
        family="M1",
        config_index=0,
        config_payload={"hidden_dim": 8, "num_blocks": 1, "dropout": 0.0},
        factor_order=("f00",),
        train_batches=build_static_training_batches(train, factor_order=("f00",)),
        validation_batches=build_static_training_batches(validation, factor_order=("f00",)),
        validation_labels=validation,
        checkpoint_root=tmp_path / "checkpoints",
        checkpoint_prefix="m1-f1-c0",
        train_dataset_id="train",
        validation_dataset_id="validation",
        max_epochs=1,
        early_stopping_patience=10,
        torch_threads=1,
    )

    results = run_parallel_search(
        (SearchTask(index=0, payload={"task": task}),),
        workers=1,
        run_task=run_cross_section_config_task,
    )

    assert results[0][0] == 0
    result = results[0][1]
    assert result["config"] == task.config_payload
    assert set(result["seed_validation_rank_ic"]) == {20260801, 20260802, 20260803}
    assert math.isfinite(float(result["ensemble_validation_rank_ic"]))


def _static_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    def frame(day: date) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "signal_date": [day] * 30,
                "ts_code": [f"S{index:03d}" for index in range(30)],
                "raw_return_5d": [float(index) for index in range(30)],
                "label_valid": [True] * 30,
                "f00": [float(index) for index in range(30)],
            }
        )

    return frame(date(2020, 1, 2)), frame(date(2020, 1, 3))


def _seed_task(tmp_path: Path, seed: int) -> CrossSectionSeedTask:
    train, validation = _static_frames()
    return CrossSectionSeedTask(
        family="M1",
        config_payload={"hidden_dim": 8, "num_blocks": 1, "dropout": 0.0},
        factor_order=("f00",),
        seed=seed,
        train_batches=build_static_training_batches(train, factor_order=("f00",)),
        validation_batches=build_static_training_batches(validation, factor_order=("f00",)),
        validation_labels=validation,
        checkpoint_root=tmp_path / "checkpoints",
        checkpoint_prefix=f"m1-f1-final-{seed}",
        train_dataset_id="train",
        validation_dataset_id="validation",
        max_epochs=1,
        early_stopping_patience=10,
        torch_threads=1,
    )


def test_a_spawned_seed_worker_hands_back_a_checkpoint_the_parent_can_freeze(tmp_path) -> None:
    """The parent writes the bundle from these paths, so a fit it did not run must still be
    persistable -- a worker that trained the weights and kept the path would leave an ensemble
    that could be scored but never frozen."""
    seed = 20260802
    results = run_parallel_search(
        (SearchTask(index=seed, payload={"task": _seed_task(tmp_path, seed)}),),
        workers=1,
        run_task=run_cross_section_seed_task,
    )
    index, result = results[0]
    assert index == seed
    assert result["seed"] == seed
    path = Path(str(result["checkpoint_path"]))
    assert path.is_file()
    # The naming contract the serial final fits produce, and the one the bundle write relies on.
    assert path.parent.name.startswith(f"m1-f1-final-{seed}-seed-{seed}-epoch-")
    assert math.isfinite(float(result["validation_rank_ic"]))
    predictions = result["predictions"]
    assert {"signal_date", "ts_code", "percentile_rank"}.issubset(predictions.columns)
    assert len(predictions) == 30


def test_three_spawned_seed_fits_reassemble_into_the_seed_ensemble(tmp_path) -> None:
    """The whole claim of parallelising the final fit: the parent can do everything the serial
    loop did -- ensemble, worst seed, three checkpoint paths -- from results it did not compute."""
    tasks = tuple(
        SearchTask(index=seed, payload={"task": _seed_task(tmp_path, seed)})
        for seed in REQUIRED_SEEDS
    )
    results = run_parallel_search(
        tasks, workers=len(REQUIRED_SEEDS), run_task=run_cross_section_seed_task
    )
    assert [index for index, _ in results] == list(REQUIRED_SEEDS)
    predictions = {seed: result["predictions"] for seed, result in results}
    rank_ic = {seed: float(result["validation_rank_ic"]) for seed, result in results}
    ensemble = build_seed_ensemble(predictions, validation_rank_ic=rank_ic)
    assert ensemble.worst_seed in set(REQUIRED_SEEDS)
    assert not ensemble.predictions.empty
    assert all(Path(str(result["checkpoint_path"])).is_file() for _, result in results)
    # Distinct seeds must not collapse into one another's weights.
    assert len({str(result["checkpoint_path"]) for _, result in results}) == len(REQUIRED_SEEDS)


def test_a_seed_task_of_a_different_seed_is_a_different_fit(tmp_path) -> None:
    """Guard against a worker ignoring its seed and returning whichever fit it happened to run."""
    first, second = _seed_task(tmp_path, 20260801), _seed_task(tmp_path, 20260802)
    assert replace(first, seed=20260802).seed == second.seed
    results = dict(
        run_parallel_search(
            (
                SearchTask(index=20260801, payload={"task": first}),
                SearchTask(index=20260802, payload={"task": second}),
            ),
            workers=2,
            run_task=run_cross_section_seed_task,
        )
    )
    assert results[20260801]["seed"] == 20260801
    assert results[20260802]["seed"] == 20260802
    assert str(results[20260801]["checkpoint_path"]) != str(results[20260802]["checkpoint_path"])
