"""C1.1: prove the shipped catalog is exactly the frozen metadata snapshot's expansion.

``configs/factors/source_catalog.yaml`` records a ``metadata_snapshot_hash``. This re-runs the
generator against the live ``factor_metadata`` table and checks three things:

* the regenerated snapshot hash equals the recorded one;
* the regenerated entries are byte-identical to the committed ones;
* no factor is present in one and absent from the other.

That is the property the offline conformance tests cannot reach. Those check the catalog describes
its source correctly; only this checks it describes *all* of it. A factor added to the source and
never registered, or a catalog entry hand-edited away from what the generator produces, are the two
drifts C1.1 exists to prevent, and both are invisible to a test that only reads the YAML.

Read-only: builds a ``YesLabReadOnlySession`` and no write session, and the single query it issues
selects five metadata columns from ``factor_metadata`` -- never factor values.

Pass ``--offline-metadata`` to check against a saved metadata frame instead of the cluster.
Exit status is 1 on any drift, so this can gate a catalog regeneration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "research_v2" / "src"))

from research_v2.factors.catalog import (  # noqa: E402
    build_yeslab_catalog_document,
)

_DEFAULT_CATALOG = REPO_ROOT / "research_v2/configs/factors/source_catalog.yaml"
_METADATA_QUERY = (
    "select factor_id, factor_name, creator, update_frequency, store_factor "
    "from loadTable('dfs://YesLab', 'factor_metadata')"
)


def _fetch_metadata(offline: Path | None) -> pd.DataFrame:
    if offline is not None:
        frame = pd.read_parquet(offline) if offline.suffix == ".parquet" else pd.read_csv(offline)
        print(f"using offline metadata from {offline} ({len(frame)} rows)", flush=True)
        return frame
    import dolphindb as ddb  # noqa: PLC0415 - only needed for a live run

    from research_v2.db.session import YesLabReadOnlySession  # noqa: PLC0415

    client = ddb.session()
    client.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    try:
        # No V2WriteSession is constructed anywhere in this script.
        result = YesLabReadOnlySession(client).query(_METADATA_QUERY)
    finally:
        client.close()
    if not isinstance(result, pd.DataFrame):
        raise TypeError("factor_metadata query did not return a DataFrame")
    return result


def _entry_sort_key(entry: dict[str, Any]) -> tuple[str, str]:
    """Order-insensitive identity of one catalog entry, for the diagnosis only.

    Both sides are compared as multisets keyed on the entry's own identity, so a reshuffled
    file compares equal here while still differing byte for byte. ``.get`` and ``str`` on
    purpose: a hand-edited catalog is exactly the input this script is asked about, and a
    verifier that crashes on the drift it exists to report has reported nothing.
    """
    return (str(entry.get("factor_id", "")), str(entry.get("factor_version", "")))


def compare(metadata: pd.DataFrame, catalog_path: Path) -> dict[str, Any]:
    stored = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    stored_entries: list[dict[str, Any]] = list(stored["factors"])
    stored_hash = str(stored["metadata_snapshot_hash"])

    rebuilt = build_yeslab_catalog_document(metadata)
    rebuilt_entries: list[dict[str, Any]] = list(rebuilt["factors"])
    rebuilt_hash = str(rebuilt["metadata_snapshot_hash"])

    stored_ids = {str(entry["factor_id"]) for entry in stored_entries}
    rebuilt_ids = {str(entry["factor_id"]) for entry in rebuilt_entries}

    differing: list[dict[str, Any]] = []
    by_id = {str(entry["factor_id"]): entry for entry in rebuilt_entries}
    for entry in stored_entries:
        counterpart = by_id.get(str(entry["factor_id"]))
        if counterpart is not None and counterpart != entry:
            fields = sorted(
                {
                    field
                    for field in set(entry) | set(counterpart)
                    if entry.get(field) != counterpart.get(field)
                }
            )
            differing.append({"factor_id": str(entry["factor_id"]), "fields": fields})

    return {
        "catalog": str(catalog_path),
        "metadata_rows": int(len(metadata)),
        "stored_snapshot_hash": stored_hash,
        "rebuilt_snapshot_hash": rebuilt_hash,
        "hash_matches": rebuilt_hash == stored_hash,
        "stored_entries": len(stored_entries),
        "rebuilt_entries": len(rebuilt_entries),
        "entries_identical": rebuilt_entries == stored_entries,
        # The gate is byte-exact on purpose and the snapshot hash is what makes it so. This
        # field exists to say *why* a byte-exact refusal happened when nothing about the
        # entries themselves changed: without it the verdict blames the expansion for what
        # is only a reshuffle, and sends whoever reads it hunting for a missing factor.
        "entries_identical_ignoring_order": sorted(rebuilt_entries, key=_entry_sort_key)
        == sorted(stored_entries, key=_entry_sort_key),
        "only_in_source": sorted(rebuilt_ids - stored_ids),
        "only_in_catalog": sorted(stored_ids - rebuilt_ids),
        "entries_differing": differing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path, default=_DEFAULT_CATALOG)
    parser.add_argument("--offline-metadata", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--save-metadata", type=Path, default=None)
    args = parser.parse_args()

    metadata = _fetch_metadata(args.offline_metadata)
    if args.save_metadata is not None:
        args.save_metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_parquet(args.save_metadata, index=False)
        print(f"saved factor_metadata to {args.save_metadata}", flush=True)

    result = compare(metadata, args.catalog)

    print(f"metadata rows        : {result['metadata_rows']}", flush=True)
    print(f"stored snapshot hash : {result['stored_snapshot_hash']}", flush=True)
    print(f"rebuilt snapshot hash: {result['rebuilt_snapshot_hash']}", flush=True)
    print(
        f"catalog entries      : {result['stored_entries']} stored / "
        f"{result['rebuilt_entries']} rebuilt",
        flush=True,
    )
    print(f"only in source       : {len(result['only_in_source'])}", flush=True)
    print(f"only in catalog      : {len(result['only_in_catalog'])}", flush=True)
    print(f"entries differing    : {len(result['entries_differing'])}", flush=True)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"report written to {args.report}", flush=True)

    reproducible = (
        result["hash_matches"]
        and result["entries_identical"]
        and not result["only_in_source"]
        and not result["only_in_catalog"]
    )
    if reproducible:
        print("CATALOG_REPRODUCIBLE", flush=True)
        return 0
    if (
        not result["entries_differing"]
        and not result["only_in_source"]
        and not result["only_in_catalog"]
        and result["entries_identical_ignoring_order"]
    ):
        print(
            "CATALOG_DRIFT: the same entries in a different order -- the file was rewritten "
            "after it was generated",
            flush=True,
        )
        return 1
    print("CATALOG_DRIFT: the shipped catalog is not the frozen snapshot's expansion", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
