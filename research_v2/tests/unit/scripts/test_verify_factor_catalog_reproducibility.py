"""The reproducibility verifier has to be able to fail, or CATALOG_REPRODUCIBLE means nothing."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import verify_factor_catalog_reproducibility as verify
import yaml

from research_v2.factors.catalog import build_yeslab_catalog_document


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "factor_id": 1,
                "factor_name": "Alpha1",
                "creator": "research",
                "update_frequency": "DAILY",
                "store_factor": "factor_data1",
            },
            {
                "factor_id": 2,
                "factor_name": "Alpha2",
                "creator": "research",
                "update_frequency": "DAILY",
                "store_factor": "factor_data2",
            },
        ]
    )


def _write_catalog(path: Path, metadata: pd.DataFrame) -> dict[str, Any]:
    document = build_yeslab_catalog_document(metadata)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return document


def test_metadata_that_matches_the_catalog_is_reported_reproducible(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.yaml"
    metadata = _metadata()
    _write_catalog(catalog, metadata)

    result = verify.compare(metadata, catalog)

    assert result["hash_matches"] is True
    assert result["entries_identical"] is True
    assert result["only_in_source"] == []
    assert result["only_in_catalog"] == []
    assert result["entries_differing"] == []


def test_a_factor_added_to_the_source_and_never_registered_is_drift(tmp_path: Path) -> None:
    """The case the offline conformance tests structurally cannot see."""
    catalog = tmp_path / "catalog.yaml"
    _write_catalog(catalog, _metadata())
    grown = pd.concat(
        [
            _metadata(),
            pd.DataFrame(
                [
                    {
                        "factor_id": 3,
                        "factor_name": "Alpha3",
                        "creator": "research",
                        "update_frequency": "DAILY",
                        "store_factor": "factor_data1",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    result = verify.compare(grown, catalog)

    assert result["hash_matches"] is False
    assert result["only_in_source"] == ["yeslab_3"]
    assert result["entries_identical"] is False


def test_a_factor_missing_from_the_source_is_drift(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.yaml"
    _write_catalog(catalog, _metadata())
    shrunk = _metadata().iloc[:1].reset_index(drop=True)

    result = verify.compare(shrunk, catalog)

    assert result["only_in_catalog"] == ["yeslab_2"]
    assert result["hash_matches"] is False


def test_a_hand_edit_to_one_catalog_entry_is_drift(tmp_path: Path) -> None:
    """Catches a catalog edited away from what the generator produces, source untouched."""
    catalog = tmp_path / "catalog.yaml"
    metadata = _metadata()
    document = _write_catalog(catalog, metadata)
    document["factors"][0]["owner"] = "somebody_else"
    catalog.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    result = verify.compare(metadata, catalog)

    # The snapshot hash covers the metadata, not the entries, so it still matches -- which is
    # exactly why the entry-level comparison has to exist alongside it.
    assert result["hash_matches"] is True
    assert result["entries_identical"] is False
    assert result["entries_differing"] == [{"factor_id": "yeslab_1", "fields": ["owner"]}]
    assert result["entries_identical_ignoring_order"] is False, (
        "ignoring order must not become ignoring content: a field was changed, not moved"
    )


def _run_main(
    monkeypatch: pytest.MonkeyPatch, catalog: Path, metadata: pd.DataFrame, tmp_path: Path
) -> int:
    """Drive the verifier the way an operator does, offline, and return its exit code."""
    offline = tmp_path / "metadata.parquet"
    metadata.to_parquet(offline, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify", "--catalog", str(catalog), "--offline-metadata", str(offline)],
    )
    return verify.main()


def test_main_says_reproducible_and_exits_zero_for_an_untouched_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The gate's own contract, which nothing covered: what it prints and what it returns.

    Every other test here calls ``compare()`` and reads a dict. ``compare()`` cannot show
    whether the verdict built on top of it is right, so a gate that always printed
    CATALOG_REPRODUCIBLE and returned 0 would have passed this whole file.
    """
    catalog = tmp_path / "catalog.yaml"
    metadata = _metadata()
    _write_catalog(catalog, metadata)

    assert _run_main(monkeypatch, catalog, metadata, tmp_path) == 0
    assert "CATALOG_REPRODUCIBLE" in capsys.readouterr().out


def test_main_exits_one_and_blames_the_order_when_only_the_order_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A reshuffle is drift, but it is not the drift the generic message describes.

    ``entries_differing`` is empty here and neither set difference fires, so the only true
    statement about the file is that its bytes moved. Blaming "the frozen snapshot's
    expansion" sends whoever reads it hunting for a dropped factor that was never dropped.
    """
    catalog = tmp_path / "catalog.yaml"
    metadata = _metadata()
    document = _write_catalog(catalog, metadata)
    document["factors"] = list(reversed(document["factors"]))
    catalog.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    assert _run_main(monkeypatch, catalog, metadata, tmp_path) == 1
    printed = capsys.readouterr().out
    assert "the same entries in a different order" in printed
    assert "not the frozen snapshot's expansion" not in printed


def test_main_still_blames_the_expansion_when_a_factor_really_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Control for the message above, so it cannot become the new generic explanation."""
    catalog = tmp_path / "catalog.yaml"
    metadata = _metadata()
    document = _write_catalog(catalog, metadata)
    document["factors"] = document["factors"][:1]
    catalog.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    assert _run_main(monkeypatch, catalog, metadata, tmp_path) == 1
    printed = capsys.readouterr().out
    assert "not the frozen snapshot's expansion" in printed
    assert "different order" not in printed


def test_a_reordered_catalog_is_drift_because_the_gate_is_byte_exact(tmp_path: Path) -> None:
    """Order carries no *semantic* meaning, and it is still part of what was generated.

    This test used to be named ``..._is_still_reproducible``, which contradicted the
    verdict ``main()`` returns for the very same file: ``reproducible`` requires the
    snapshot hash to match and the hash covers the bytes. The distinction the old name was
    reaching for is now reported separately, as a diagnosis rather than as a verdict.
    """
    catalog = tmp_path / "catalog.yaml"
    metadata = _metadata()
    document = _write_catalog(catalog, metadata)
    document["factors"] = list(reversed(document["factors"]))
    catalog.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    result = verify.compare(metadata, catalog)

    assert result["entries_differing"] == []
    assert result["only_in_source"] == [] and result["only_in_catalog"] == []
    assert result["entries_identical"] is False, (
        "order still differs, so this is reported honestly rather than silently normalized"
    )
    assert result["entries_identical_ignoring_order"] is True


def test_offline_metadata_round_trips_through_parquet(tmp_path: Path) -> None:
    target = tmp_path / "metadata.parquet"
    metadata = _metadata()
    metadata.to_parquet(target, index=False)

    restored = verify._fetch_metadata(target)

    assert len(restored) == len(metadata)
    assert sorted(restored["factor_name"]) == sorted(metadata["factor_name"])


def test_the_verifier_never_constructs_a_write_session() -> None:
    tree = ast.parse(Path(verify.__file__).resolve().read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names.update(node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute))

    assert "V2WriteSession" not in names
    assert "YesLabReadOnlySession" in names


def test_the_only_query_selects_metadata_columns_and_no_factor_values() -> None:
    projection, _, remainder = verify._METADATA_QUERY.partition("from")
    selected = {name.strip() for name in projection.removeprefix("select").split(",")}

    assert selected == {
        "factor_id",
        "factor_name",
        "creator",
        "update_frequency",
        "store_factor",
    }
    assert "value" not in selected
    assert "factor_metadata" in remainder
    assert not re.search(r"factor_data(?:[1-9]|10|11)\b", verify._METADATA_QUERY), (
        "the verifier must not touch the factor observation tables"
    )
