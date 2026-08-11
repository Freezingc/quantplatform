"""Prepare one isolated factor/month repair batch for a later atomic publish.

The source client is intentionally opened for one factor only.  The DolphinDB
Python client can terminate unexpectedly after switching between large source
factor tables in one process; isolating preparation keeps every generated
batch write-free and lets the publisher advance one six-factor version once.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import uuid
from pathlib import Path

import dolphindb as ddb
import pandas as pd

from research_v2.db.session import YesLabReadOnlySession
from research_v2.factors.materializer import build_technical_factor_batches
from research_v2.factors.repair import patch_factor_source
from research_v2.factors.source_backfill import (
    FactorMonth,
    build_factor_definitions,
    build_factor_month_context,
)
from research_v2.runs.context import RunContext

_FACTOR_IDS = {
    "GP_008": "yeslab_377",
    "GP_011": "yeslab_380",
    "GP_018": "yeslab_387",
    "GP_019": "yeslab_388",
    "GP_022": "yeslab_391",
    "GP_064": "yeslab_433",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", required=True, help="YYYY-MM")
    parser.add_argument("--factor", required=True, choices=sorted(_FACTOR_IDS))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--repair-hash", default="primary")
    parser.add_argument(
        "--nonce",
        default="tushare-gap-20260508-gp022-gp064-v3",
    )
    args = parser.parse_args()
    factor_id = _FACTOR_IDS[args.factor]
    month = FactorMonth.from_partition(args.partition)
    if args.factor in {"GP_022", "GP_064"} and args.candidate is None:
        raise ValueError("repaired GP factors require --candidate")
    if args.factor not in {"GP_022", "GP_064"} and args.candidate is not None:
        raise ValueError("candidate is only valid for GP_022 and GP_064")

    client = ddb.session()
    client.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    source_session = YesLabReadOnlySession(client)
    metadata = source_session.query(
        "select factor_id,factor_name,creator,update_frequency,store_factor "
        "from loadTable('dfs://YesLab','factor_metadata') "
        "where factor_id in [377,380,387,388,391,433]"
    )
    if not isinstance(metadata, pd.DataFrame):
        raise RuntimeError("factor metadata query did not return a DataFrame")
    definitions = build_factor_definitions(
        metadata,
        tuple(_FACTOR_IDS.values()),
    )
    definition = next(item for item in definitions if item.factor_id == factor_id)
    context_seed = build_factor_month_context(
        month,
        definitions,
        attempt_nonce=f"{args.nonce}|{args.repair_hash}",
    )
    shared_run_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"quant-research-v2/factor-repair/{args.partition}/{args.nonce}",
        )
    )
    context = RunContext(run_id=shared_run_id, data_version=context_seed.data_version)
    query = (
        "select ts_code,trade_date,factor_name,value from "
        f"loadTable('dfs://YesLab','{definition.source_table}') "
        f"where factor_name='{definition.display_name}' "
        f"and trade_date between date('{month.start_date:%Y.%m.%d}') "
        f"and date('{month.end_date:%Y.%m.%d}')"
    )
    source = source_session.query(query)
    if not isinstance(source, pd.DataFrame):
        raise RuntimeError("factor source query did not return a DataFrame")
    if args.candidate is not None:
        candidate = pd.read_pickle(args.candidate)
        candidate["trade_date"] = pd.to_datetime(candidate["trade_date"]).dt.date
        candidate = candidate.loc[
            candidate["trade_date"].between(month.start_date, month.end_date)
        ].copy()
        source = patch_factor_source(
            source,
            candidate,
            factor_name=definition.display_name,
        )
    source_watermark_id = hashlib.sha256(
        f"{definition.source_table}|{args.partition}|{args.repair_hash}|{args.nonce}".encode()
    ).hexdigest()
    materialized = build_technical_factor_batches(
        source,
        definition,
        source_factor_name=definition.display_name,
        source_watermark_id=source_watermark_id,
        context=context,
        record_hash_strategy="fast",
        include_lineage=True,
    )
    if materialized.quarantine_batch is not None:
        reasons = set(materialized.quarantine_batch.frame["reason"].astype(str))
        if reasons.difference({"NO_FINITE_VALUE"}):
            raise RuntimeError(
                f"blocking quarantine for {args.partition}/{args.factor}: {sorted(reasons)}"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    materialized.clean_batch.frame.to_pickle(args.output)
    print(
        f"prepared {args.partition}/{args.factor} rows={len(materialized.clean_batch.frame)} "
        f"data_version={context.data_version} output={args.output}"
    )


if __name__ == "__main__":
    main()
