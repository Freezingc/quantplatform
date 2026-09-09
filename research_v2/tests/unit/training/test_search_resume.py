"""Tests for the config-search resume primitive.

The property that matters is that adoption is *earned*: a persisted search record may replace hours
of training only when every numerics-affecting input it was produced under is still the one this run
would use. The bindings are the content-addressed dataset ids, the factor order, the family and its
code version, the config payload and its hash, the config's effective budget and the runtime
descriptor -- and each one gets a rejection test, because the failure mode of a too-loose check is
not an error, it is a silently different model selected out of a mixture of two runs.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from research_v2.training.search_resume import SearchResume, torch_runtime_bindings

FACTOR_ORDER = ("f00", "f01", "f02")
CONFIG: dict[str, Any] = {
    "hidden_dim": 64,
    "num_blocks": 1,
    "dropout": 0.0,
    "learning_rate": 0.003,
}
RESULT: dict[str, Any] = {
    "config": CONFIG,
    "ensemble_validation_rank_ic": 0.10573829401712345,
    "seed_validation_rank_ic": {20260801: 0.1029, 20260802: 0.1057, 20260803: 0.0988},
    "worst_seed": 20260803,
}
RUNTIME: dict[str, object] = {
    "torch_version": "2.4.1",
    "torch_intraop_threads": 52,
    "torch_interop_threads": 5,
}
BUDGET: dict[str, Any] = {"max_epochs": 100, "early_stopping_patience": 10}


def _resume(tmp_path: Path, **overrides: Any) -> SearchResume:
    kwargs: dict[str, Any] = {
        "workspace": tmp_path,
        "fold_id": "F1",
        "family": "RESIDUAL_MLP",
        "model_code_version": "residual-mlp-1",
        "train_dataset_id": "ds-1:train",
        "validation_dataset_id": "ds-1:validation",
        "factor_order": FACTOR_ORDER,
        "runtime": RUNTIME,
        "enabled": True,
    }
    kwargs.update(overrides)
    return SearchResume(**kwargs)


def _persisted(tmp_path: Path, config_index: int = 0, **overrides: Any) -> Path:
    return _resume(tmp_path, **overrides).persist(
        config_index, CONFIG, RESULT, budget=BUDGET
    )


def _adopt(
    tmp_path: Path,
    index: int = 0,
    config: Any = CONFIG,
    budget: Any = BUDGET,
    **overrides: Any,
) -> Any:
    return _resume(tmp_path, **overrides).adopt(index, config, budget=budget)


def test_a_persisted_record_round_trips_exactly(tmp_path: Path) -> None:
    """Floats must survive bit-for-bit: a resumed run's report has to equal a fresh run's."""
    path = _persisted(tmp_path)
    assert path.exists()
    lookup = _adopt(tmp_path)
    assert lookup.record is not None
    assert lookup.reason == ""
    assert lookup.record == RESULT
    assert lookup.record["ensemble_validation_rank_ic"].hex() == RESULT[
        "ensemble_validation_rank_ic"
    ].hex()


def test_json_stringifies_integer_seed_keys_and_the_restore_puts_them_back(
    tmp_path: Path,
) -> None:
    path = _persisted(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert set(raw["result"]["seed_validation_rank_ic"]) == {"20260801", "20260802", "20260803"}
    record = _adopt(tmp_path).record
    assert record is not None
    assert set(record["seed_validation_rank_ic"]) == set(RESULT["seed_validation_rank_ic"])
    assert isinstance(record["worst_seed"], int)


def test_a_disabled_run_still_persists_but_never_adopts(tmp_path: Path) -> None:
    """Recording is unconditional; adopting is opt-in, so default behaviour cannot change."""
    path = _persisted(tmp_path, enabled=False)
    assert path.exists()
    lookup = _adopt(tmp_path, enabled=False)
    assert lookup.record is None
    assert "disabled" in lookup.reason


def test_an_absent_record_is_a_reason_not_an_error(tmp_path: Path) -> None:
    lookup = _adopt(tmp_path)
    assert lookup.record is None
    assert "no record" in lookup.reason


def test_the_record_path_is_bound_to_the_config_and_its_index(tmp_path: Path) -> None:
    """A grid that changed under a fold must not inherit the old grid's results."""
    path = _persisted(tmp_path, config_index=2)
    assert path.parent.name == "F1"
    assert path.name.startswith("c2-")
    assert _adopt(tmp_path, index=3).record is None
    other_config = _adopt(tmp_path, index=2, config=dict(CONFIG, hidden_dim=256))
    assert other_config.record is None
    assert "no record" in other_config.reason


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        pytest.param("family", "SET_ATTENTION", "family", id="family"),
        pytest.param("model_code_version", "residual-mlp-2", "model_code_version", id="code"),
        pytest.param("train_dataset_id", "ds-2:train", "train_dataset_id", id="train_ds"),
        pytest.param(
            "validation_dataset_id", "ds-2:validation", "validation_dataset_id", id="val_ds"
        ),
        pytest.param("factor_order", ("f00", "f01"), "factor_order", id="factors"),
    ],
)
def test_a_changed_binding_refuses_adoption(
    field: str, value: Any, match: str, tmp_path: Path
) -> None:
    _persisted(tmp_path)
    lookup = _adopt(tmp_path, **{field: value})
    assert lookup.record is None
    assert match in lookup.reason


@pytest.mark.parametrize(
    ("budget", "match"),
    [
        pytest.param({"max_epochs": 50}, "max_epochs", id="epochs"),
        pytest.param(
            {"max_epochs": 100, "early_stopping_patience": 20}, "patience", id="patience"
        ),
        pytest.param(dict(BUDGET, num_boost_round=500), "budget", id="extra_knob"),
        pytest.param(
            dict(BUDGET, hyperparameters={"num_threads": 4}), "budget", id="effective_hyperparams"
        ),
    ],
)
def test_a_changed_budget_refuses_adoption(
    budget: dict[str, Any], match: str, tmp_path: Path
) -> None:
    """The budget is per config: for the tree families it carries the effective hyperparameters,
    which include library-side defaults a config hash cannot see."""
    _persisted(tmp_path)
    lookup = _adopt(tmp_path, budget=budget)
    assert lookup.record is None
    assert match in lookup.reason


def test_a_budget_record_survives_the_json_round_trip(tmp_path: Path) -> None:
    """Nested effective hyperparameters must compare equal after being written and read back."""
    budget = {
        "num_boost_round": 500,
        "hyperparameters": {"num_leaves": 31, "learning_rate": 0.05, "num_threads": 8},
    }
    _resume(tmp_path).persist(0, CONFIG, RESULT, budget=budget)
    assert _adopt(tmp_path, budget=budget).record is not None


def test_another_fold_simply_has_no_records(tmp_path: Path) -> None:
    """Records live under search/<fold>/, so cross-fold adoption is impossible by construction."""
    _persisted(tmp_path)
    lookup = _adopt(tmp_path, fold_id="F2")
    assert lookup.record is None
    assert "no record" in lookup.reason


def test_a_record_copied_into_another_fold_directory_is_still_refused(tmp_path: Path) -> None:
    """The fold_id binding is the second layer: path separation is the first, not the only."""
    path = _persisted(tmp_path)
    other = tmp_path / "search" / "F2"
    other.mkdir(parents=True)
    shutil.copy(path, other / path.name)
    lookup = _adopt(tmp_path, fold_id="F2")
    assert lookup.record is None
    assert "fold_id" in lookup.reason


def test_a_changed_runtime_refuses_adoption(tmp_path: Path) -> None:
    """Thread count changes torch numerics, so a record from another regime is another result."""
    _persisted(tmp_path)
    lookup = _adopt(tmp_path, runtime=dict(RUNTIME, torch_intraop_threads=1))
    assert lookup.record is None
    assert "runtime" in lookup.reason


def test_a_record_whose_config_disagrees_with_its_own_filename_is_refused(tmp_path: Path) -> None:
    path = _persisted(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["config"] = dict(CONFIG, hidden_dim=256)
    path.write_text(json.dumps(document), encoding="utf-8")
    lookup = _adopt(tmp_path)
    assert lookup.record is None
    assert "config" in lookup.reason


def test_a_record_from_another_schema_is_refused(tmp_path: Path) -> None:
    path = _persisted(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["schema"] = "qrv2-search-record-0"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert "schema" in _adopt(tmp_path).reason


@pytest.mark.parametrize("body", ["not json", "[]", '"x"', "{}"])
def test_a_corrupt_record_is_refused_with_a_reason(tmp_path: Path, body: str) -> None:
    resume = _resume(tmp_path)
    path = resume.record_path(0, CONFIG)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    lookup = resume.adopt(0, CONFIG, budget=BUDGET)
    assert lookup.record is None
    assert lookup.reason


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda r: r.pop("worst_seed"), id="no_worst_seed"),
        pytest.param(lambda r: r.pop("ensemble_validation_rank_ic"), id="no_rank_ic"),
        pytest.param(lambda r: r.pop("seed_validation_rank_ic"), id="no_seed_map"),
        pytest.param(lambda r: r.update(ensemble_validation_rank_ic="0.1"), id="rank_ic_str"),
        pytest.param(lambda r: r.update(worst_seed="20260803"), id="worst_seed_str"),
        pytest.param(lambda r: r.update(seed_validation_rank_ic={20260801: 0.1}), id="seed_short"),
    ],
)
def test_an_incomplete_result_is_refused(tmp_path: Path, mutate: Any) -> None:
    """Mutate a genuinely persisted record's result only, so the binding checks pass first."""
    path = _persisted(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document["result"])
    path.write_text(json.dumps(document), encoding="utf-8")
    lookup = _adopt(tmp_path)
    assert lookup.record is None
    assert "result" in lookup.reason


def test_persist_refuses_a_result_it_could_not_read_back(tmp_path: Path) -> None:
    resume = _resume(tmp_path)
    with pytest.raises(ValueError, match="seed_validation_rank_ic"):
        resume.persist(
            0, CONFIG, dict(RESULT, seed_validation_rank_ic={20260801: 0.1}), budget=BUDGET
        )


def test_the_log_explains_both_outcomes(tmp_path: Path) -> None:
    """A transcript that does not say why a config was not trained is unauditable."""
    lines: list[str] = []
    _persisted(tmp_path, log=lines.append)
    _adopt(tmp_path, log=lines.append)
    _adopt(tmp_path, index=1, log=lines.append)
    assert any("resumed" in line for line in lines)
    assert any("no record" in line for line in lines)


def test_torch_runtime_bindings_describe_without_perturbing() -> None:
    """The descriptor must be an observation: pinning threads is an adjudicated change, not ours."""
    import torch

    before_threads = torch.get_num_threads()
    before_interop = torch.get_num_interop_threads()
    bindings = torch_runtime_bindings()
    assert set(bindings) == {
        "torch_version",
        "torch_intraop_threads",
        "torch_interop_threads",
    }
    assert bindings["torch_version"] == torch.__version__
    assert bindings["torch_intraop_threads"] == before_threads
    assert bindings["torch_interop_threads"] == before_interop
    # Observing must not configure: a pinned thread count is an adjudicated numerics change.
    assert torch.get_num_threads() == before_threads
    assert torch.get_num_interop_threads() == before_interop
