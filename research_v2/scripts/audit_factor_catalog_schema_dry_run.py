"""C1.5: dry-run the factor source catalog against the live source schema.

Reads column metadata for every table the catalog names and classifies each entry as
``MAPPED``/``BLOCKED``/``RETIRED``, then reports the source columns no entry maps.

This script constructs **no** ``V2WriteSession`` and cannot write: the only session it builds is
``YesLabReadOnlySession``, whose own guard rejects mutating scripts, and the only query the dry-run
can emit is ``schema(loadTable(...)).colDefs``. No factor values are read at any point -- the YesLab
factor tables hold the full history of 487 factors, and every check C1.5 names is answerable from
column metadata alone.

Pass ``--save-schemas`` to record what the source reported and ``--schemas-from`` to replay it, so a
verdict can be re-examined later without touching the cluster again.

Exit status is 1 when any entry is BLOCKED or any table has unmapped columns, so this can gate a
catalog change; ``--report-only`` downgrades that to a printed summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "research_v2" / "src"))

from research_v2.factors.catalog_dry_run import (  # noqa: E402
    CatalogDryRunReport,
    SourceColumn,
    catalog_tables,
    collect_source_schemas,
    run_catalog_dry_run,
    schema_set_from_payload,
    schema_set_to_payload,
)

_DEFAULT_CATALOG = REPO_ROOT / "research_v2/configs/factors/source_catalog.yaml"


def _load_entries(catalog_path: Path) -> tuple[list[dict[str, Any]], str | None]:
    document = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    entries = list(document.get("factors") or [])
    if not entries:
        raise ValueError(f"catalog has no factor entries: {catalog_path}")
    snapshot_hash = document.get("metadata_snapshot_hash")
    return entries, (str(snapshot_hash) if snapshot_hash else None)


def _parse_retired(raw: str | None) -> frozenset[tuple[str, int]]:
    if not raw or not raw.strip():
        return frozenset()
    retired: set[tuple[str, int]] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        factor_id, _, version = token.partition(":")
        retired.add((factor_id.strip(), int(version or 1)))
    return frozenset(retired)


def _schemas_from_disk(path: Path) -> dict[str, tuple[SourceColumn, ...] | None]:
    """Read a schema set saved by ``--save-schemas``. Delegates to the shared format definition."""
    saved = json.loads(path.read_text(encoding="utf-8"))
    schemas, _ = schema_set_from_payload(saved)
    return schemas


def _schema_payload(
    schemas: Any, errors: dict[str, str]
) -> dict[str, Any]:
    """Serialize a schema set for ``--save-schemas``. Delegates to the shared format definition."""
    return schema_set_to_payload(schemas, errors)


def _summarize(report: CatalogDryRunReport) -> dict[str, Any]:
    return {
        "mapped": report.mapped_count,
        "blocked": report.blocked_count,
        "retired": report.retired_count,
        "total": len(report.findings),
        "unmapped_source_columns": {
            table: list(columns)
            for table, columns in sorted(report.unmapped_source_columns.items())
            if columns
        },
        "schema_fingerprints": dict(sorted(report.schema_fingerprints.items())),
        "blocked_detail": [
            {
                "factor_id": finding.factor_id,
                "factor_version": finding.factor_version,
                "source_table": finding.source_table,
                "reasons": list(finding.reasons),
            }
            for finding in report.blocked_findings
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path, default=_DEFAULT_CATALOG)
    parser.add_argument("--retired", default=None, help="comma-separated factor_id[:version]")
    parser.add_argument("--report", type=Path, default=None, help="write the JSON verdict here")
    parser.add_argument("--save-schemas", type=Path, default=None)
    parser.add_argument("--schemas-from", type=Path, default=None, help="replay a saved schema set")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="print the summary but exit 0 even when entries are blocked",
    )
    args = parser.parse_args()

    entries, snapshot_hash = _load_entries(args.catalog)
    retired = _parse_retired(args.retired)
    tables = catalog_tables(entries)
    print(
        f"catalog={args.catalog.name} entries={len(entries)} tables={len(tables)} "
        f"metadata_snapshot_hash={snapshot_hash}",
        flush=True,
    )

    errors: dict[str, str] = {}
    if args.schemas_from is not None:
        print(f"replaying source schemas from {args.schemas_from}", flush=True)
        schemas = _schemas_from_disk(args.schemas_from)
        saved = json.loads(args.schemas_from.read_text(encoding="utf-8"))
        errors = dict(saved.get("errors") or {})
        queries: list[str] = []
    else:
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
            collected = collect_source_schemas(YesLabReadOnlySession(client), tables)
        finally:
            client.close()
        schemas = dict(collected.schemas)
        errors = dict(collected.errors)
        queries = [
            f"schema(loadTable('dfs://YesLab', '{table}')).colDefs" for table in tables
        ]

    if args.save_schemas is not None:
        payload = _schema_payload(schemas, errors)
        payload["queries"] = queries
        args.save_schemas.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"saved source schemas to {args.save_schemas}", flush=True)

    for table in tables:
        columns = schemas.get(table)
        if columns is None:
            print(f"  {table}: UNREADABLE ({errors.get(table, 'not collected')})", flush=True)
        else:
            rendered = ", ".join(f"{column.name}:{column.type_string}" for column in columns)
            print(f"  {table}: {rendered}", flush=True)

    report = run_catalog_dry_run(entries, schemas, retired=retired, schema_errors=errors)
    summary = _summarize(report)

    print(
        f"mapped={summary['mapped']} blocked={summary['blocked']} retired={summary['retired']} "
        f"total={summary['total']}",
        flush=True,
    )
    if summary["unmapped_source_columns"]:
        print(
            "unmapped source columns (present in the source, registered by no entry):",
            flush=True,
        )
        for table, columns in summary["unmapped_source_columns"].items():
            print(f"  {table}: {', '.join(columns)}", flush=True)
    else:
        print("unmapped source columns: none", flush=True)
    for blocked in summary["blocked_detail"][:50]:
        print(
            f"  BLOCKED {blocked['factor_id']}:{blocked['factor_version']} "
            f"({blocked['source_table']}): {'; '.join(blocked['reasons'])}",
            flush=True,
        )
    if len(summary["blocked_detail"]) > 50:
        print(f"  ... {len(summary['blocked_detail']) - 50} more blocked entries", flush=True)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "catalog": str(args.catalog),
                    "metadata_snapshot_hash": snapshot_hash,
                    "summary": summary,
                    "findings": [
                        {
                            "factor_id": finding.factor_id,
                            "factor_version": finding.factor_version,
                            "source_table": finding.source_table,
                            "status": finding.status,
                            "reasons": list(finding.reasons),
                        }
                        for finding in report.findings
                    ],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"report written to {args.report}", flush=True)

    if args.report_only:
        return 0
    if summary["blocked"] or summary["unmapped_source_columns"]:
        print("DRY_RUN_FAILED: blocked entries or unmapped source columns present", flush=True)
        return 1
    print("DRY_RUN_PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
