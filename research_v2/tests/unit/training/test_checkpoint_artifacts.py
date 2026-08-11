from __future__ import annotations

import torch
from torch import nn

from research_v2.training.artifacts import CheckpointStore


def test_checkpoint_round_trip_is_hashed_and_immutable(tmp_path: object) -> None:
    root = tmp_path / "checkpoints"  # type: ignore[operator]
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    store = CheckpointStore(root)

    artifact = store.save(
        "F01-seed-20260801-epoch-1",
        model,
        optimizer,
        metadata={"dataset_id": "data-v1", "validation_rank_ic": 0.02},
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    manifest = store.load(artifact.checkpoint_id, model, optimizer)

    assert artifact.path.is_file()
    assert store.verify(artifact.checkpoint_id)
    assert manifest["metadata"]["dataset_id"] == "data-v1"
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
    try:
        store.save(artifact.checkpoint_id, model, optimizer, metadata={})
    except FileExistsError:
        pass
    else:
        raise AssertionError("checkpoint overwrite must be rejected")
