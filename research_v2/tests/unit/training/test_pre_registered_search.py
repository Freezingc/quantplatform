from __future__ import annotations

from research_v2.training.search import build_pre_registered_trials, select_validation_winner


def test_trials_are_content_ordered_and_capped_independent_of_input_mapping_order() -> None:
    first = build_pre_registered_trials(
        "RESIDUAL_MLP",
        {"hidden": (64, 128), "dropout": (0.0, 0.1), "lr": (0.001,)},
        maximum_trials=3,
    )
    second = build_pre_registered_trials(
        "RESIDUAL_MLP",
        {"lr": (0.001,), "dropout": (0.0, 0.1), "hidden": (64, 128)},
        maximum_trials=3,
    )

    assert first == second
    assert len(first) == 3
    assert len({trial.config_hash for trial in first}) == 3


def test_winner_uses_validation_rank_ic_with_stable_hash_tie_break() -> None:
    trials = build_pre_registered_trials("M1", {"hidden": (64, 128)}, maximum_trials=2)
    winner = select_validation_winner(
        {trials[0].config_hash: 0.02, trials[1].config_hash: 0.02},
        trials,
    )

    assert winner.config_hash == min(trial.config_hash for trial in trials)
