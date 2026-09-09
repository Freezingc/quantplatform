from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_v2.datasets.preprocessing import FoldPreprocessor


def test_preprocessor_fits_only_train_and_emits_missing_masks() -> None:
    train = pd.DataFrame({"f1": [0.0, 1.0, 2.0], "f2": [1.0, None, 3.0]})
    validation = pd.DataFrame({"f1": [1000.0], "f2": [None]})
    preprocessor = FoldPreprocessor(("f1", "f2"))

    preprocessor.fit(train, role="TRAIN")
    transformed = preprocessor.transform(validation)

    assert transformed.values.loc[0, "f1"] == pytest.approx(1.0)
    assert transformed.values.loc[0, "f2"] == 0.0
    assert transformed.missing_mask.loc[0, "f2"]
    same = FoldPreprocessor(("f1", "f2"))
    same.fit(train, role="TRAIN")
    assert preprocessor.state_hash == same.state_hash


def test_preprocessor_rejects_nontrain_fit_and_schema_order_drift() -> None:
    preprocessor = FoldPreprocessor(("f1", "f2"))

    with pytest.raises(ValueError, match="TRAIN"):
        preprocessor.fit(pd.DataFrame({"f1": [1.0], "f2": [2.0]}), role="VALIDATION")
    preprocessor.fit(pd.DataFrame({"f1": [1.0], "f2": [2.0]}), role="TRAIN")
    with pytest.raises(ValueError, match="order"):
        preprocessor.transform(pd.DataFrame({"f2": [2.0], "f1": [1.0]}))


# ---------- persisting and restoring the fitted state ----------


def _fitted(order: tuple[str, ...] = ("f0", "f1", "f2")) -> tuple[FoldPreprocessor, pd.DataFrame]:
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(
        {name: rng.standard_normal(400) * (index + 1) for index, name in enumerate(order)}
    )
    preprocessor = FoldPreprocessor(factor_order=order)
    preprocessor.fit(frame, role="TRAIN")
    return preprocessor, frame


def test_state_payload_round_trips_and_transforms_identically() -> None:
    fitted, frame = _fitted()
    payload = fitted.state_payload()
    restored = FoldPreprocessor.restore(payload, expected_state_hash=fitted.state_hash)

    assert restored.state_hash == fitted.state_hash
    assert restored.state_payload() == payload
    expected = fitted.transform(frame)
    actual = restored.transform(frame)
    pd.testing.assert_frame_equal(actual.values, expected.values)
    pd.testing.assert_frame_equal(actual.missing_mask, expected.missing_mask)


def test_state_hash_is_a_digest_of_the_published_payload() -> None:
    """The bundle records both; the hash must actually bind the payload, not merely name it."""
    import hashlib
    import json

    fitted, _ = _fitted()
    canonical = json.dumps(fitted.state_payload(), separators=(",", ":"), sort_keys=True)
    assert fitted.state_hash == hashlib.sha256(canonical.encode()).hexdigest()


def test_restore_refuses_a_tampered_payload() -> None:
    fitted, _ = _fitted()
    payload = fitted.state_payload()
    state = payload["state"]
    assert isinstance(state, dict)
    entry = state["f1"]
    assert isinstance(entry, dict)
    entry["median"] = float(entry["median"]) + 1.0
    with pytest.raises(ValueError, match="does not match its recorded state hash"):
        FoldPreprocessor.restore(payload, expected_state_hash=fitted.state_hash)


def test_restore_refuses_a_foreign_hash() -> None:
    fitted, _ = _fitted()
    with pytest.raises(ValueError, match="does not match its recorded state hash"):
        FoldPreprocessor.restore(fitted.state_payload(), expected_state_hash="0" * 64)


def test_restore_refuses_an_empty_expected_hash() -> None:
    fitted, _ = _fitted()
    with pytest.raises(ValueError, match="requires the recorded state hash"):
        FoldPreprocessor.restore(fitted.state_payload(), expected_state_hash="")


def _drop_a_setting(payload: dict[str, object]) -> None:
    settings = payload["settings"]
    assert isinstance(settings, dict)
    payload["settings"] = {k: v for k, v in settings.items() if k != "minimum_iqr"}


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda p: p.pop("settings"), id="missing_settings"),
        pytest.param(lambda p: p.update(extra=1), id="extra_field"),
        pytest.param(lambda p: p.update(factor_order=[]), id="empty_factor_order"),
        pytest.param(lambda p: p.update(factor_order=["f0", "f0", "f1"]), id="duplicate_factors"),
        pytest.param(lambda p: p.update(factor_order=["f0", 1]), id="non_string_factor"),
        pytest.param(
            lambda p: p.update(settings={**p["settings"], "lower_quantile": 0.9}),
            id="quantiles_out_of_order",
        ),
        pytest.param(_drop_a_setting, id="incomplete_settings"),
        pytest.param(
            lambda p: p.update(state={k: v for k, v in p["state"].items() if k != "f2"}),
            id="state_missing_a_factor",
        ),
        pytest.param(
            lambda p: p["state"]["f0"].pop("iqr"), id="state_entry_missing_a_field"
        ),
        pytest.param(
            lambda p: p["state"]["f0"].update(constant="yes"), id="constant_not_a_bool"
        ),
        pytest.param(
            lambda p: p["state"]["f0"].update(median=float("nan")), id="non_finite_median"
        ),
    ],
)
def test_restore_refuses_a_malformed_payload(mutate) -> None:
    fitted, _ = _fitted()
    payload = fitted.state_payload()
    mutate(payload)
    with pytest.raises(ValueError):
        FoldPreprocessor.restore(payload, expected_state_hash=fitted.state_hash)


def test_state_payload_requires_a_fitted_preprocessor() -> None:
    with pytest.raises(ValueError, match="has not been fitted"):
        FoldPreprocessor(factor_order=("f0",)).state_payload()


def test_a_restored_preprocessor_cannot_be_refitted() -> None:
    """Restore yields a fitted instance, so the fit-once invariant still holds."""
    fitted, frame = _fitted()
    restored = FoldPreprocessor.restore(
        fitted.state_payload(), expected_state_hash=fitted.state_hash
    )
    with pytest.raises(ValueError, match="fitted only once"):
        restored.fit(frame, role="TRAIN")
