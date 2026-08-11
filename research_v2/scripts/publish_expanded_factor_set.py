"""Preflight and publish the frozen 16-factor V2 source/compact version.

The script is intentionally explicit about its scope and target.  It first
materializes every month in memory and rejects missing-factor or conflict
quarantines.  Only after all months pass does it append immutable source rows
and advance the V2 pointers.  Existing source versions are never deleted.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

import dolphindb as ddb
import pandas as pd

from research_v2.compact import CompactObjectSpec
from research_v2.compact.publisher import (
    list_current_source_pointers,
    materialize_and_publish_current_partition,
)
from research_v2.db.session import V2WriteSession, YesLabReadOnlySession
from research_v2.factors.source_backfill import (
    FactorBackfillError,
    preflight_factor_month,
    publish_factor_month,
)

DEFAULT_FACTOR_IDS: tuple[str, ...] = (
    "yeslab_376",
    "yeslab_377",
    "yeslab_380",
    "yeslab_381",
    "yeslab_387",
    "yeslab_388",
    "yeslab_391",
    "yeslab_395",
    "yeslab_396",
    "yeslab_400",
    "yeslab_405",
    "yeslab_425",
    "yeslab_432",
    "yeslab_433",
    "yeslab_442",
    "yeslab_443",
)


def _connect() -> tuple[object, object]:
    host = os.getenv("QRV2_DDB_HOST", "10.176.55.211")
    port = int(os.getenv("QRV2_DDB_PORT", "8848"))
    username = os.getenv("QRV2_DDB_USERNAME", "xiahongjie")
    password = os.getenv("QRV2_DDB_PASSWORD", "123456")
    source_client = ddb.session()
    source_client.connect(host, port, username, password)
    target_client = ddb.session()
    target_client.connect(host, port, username, password)
    return source_client, target_client


def _parse_date(value: str) -> date:
    return pd.Timestamp(value).date()


def _metadata(source_session: YesLabReadOnlySession, factor_ids: tuple[str, ...]) -> pd.DataFrame:
    ids = ", ".join(str(int(factor_id.removeprefix("yeslab_"))) for factor_id in factor_ids)
    result = source_session.query(
        "select factor_id,factor_name,creator,update_frequency,store_factor "
        "from loadTable('dfs://YesLab','factor_metadata') "
        f"where factor_id in [{ids}]"
    )
    if not isinstance(result, pd.DataFrame):
        raise FactorBackfillError("factor metadata query did not return a DataFrame")
    return result.copy(deep=True)


def _months(start_date: date, end_date: date) -> tuple[str, ...]:
    return tuple(str(period) for period in pd.period_range(start_date, end_date, freq="M"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--attempt-nonce", default="expanded-factor-set-16-v1")
    parser.add_argument("--factor-ids", default=",".join(DEFAULT_FACTOR_IDS))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    start_date = _parse_date(args.start)
    end_date = _parse_date(args.end)
    factor_ids = tuple(value.strip() for value in args.factor_ids.split(",") if value.strip())
    if start_date > end_date or not factor_ids:
        raise ValueError("invalid date range or factor list")
    if len(set(factor_ids)) != len(factor_ids):
        raise ValueError("factor_ids must be unique")
    if not args.attempt_nonce.strip():
        raise ValueError("attempt_nonce cannot be empty")

    source_client, target_client = _connect()
    source_session = YesLabReadOnlySession(source_client)
    target_session = V2WriteSession(target_client, target_database="dfs://QuantResearchV2Test")
    metadata = _metadata(source_session, factor_ids)
    partitions = _months(start_date, end_date)
    if args.skip_preflight:
        if not args.report.exists():
            raise FactorBackfillError("--skip-preflight requires an existing preflight report")
        prior = json.loads(args.report.read_text(encoding="utf-8"))
        if (
            prior.get("target_database") != "dfs://QuantResearchV2Test"
            or tuple(prior.get("factor_ids", ())) != factor_ids
            or prior.get("start_date") != start_date.isoformat()
            or prior.get("end_date") != end_date.isoformat()
            or prior.get("attempt_nonce") != args.attempt_nonce
            or len(prior.get("preflight", ())) != len(partitions)
        ):
            raise FactorBackfillError("preflight report does not match requested publication scope")
        preflight_rows = list(prior["preflight"])
        print("reusing validated preflight report", flush=True)
    else:
        preflight_rows = []
        for index, logical_partition in enumerate(partitions, start=1):
            audit = preflight_factor_month(
                source_session,
                metadata,
                factor_ids=factor_ids,
                logical_partition=logical_partition,
                attempt_nonce=args.attempt_nonce,
            )
            blocking = [
                (factor_id, reason, count)
                for factor_id, reason, count in audit.quarantine_reasons_by_factor
                if reason != "NO_FINITE_VALUE" and count > 0
            ]
            if blocking:
                raise FactorBackfillError(
                    f"blocking quarantine in {logical_partition}: {blocking}"
                )
            clean_counts = dict(audit.clean_rows_by_factor)
            quarantine_counts = dict(audit.quarantine_rows_by_factor)
            missing = [
                factor_id for factor_id in factor_ids if clean_counts.get(factor_id, 0) == 0
            ]
            if missing:
                raise FactorBackfillError(
                    "requested factors have no clean observations in "
                    f"{logical_partition}: {missing}"
                )
            preflight_rows.append(
                {
                    "logical_partition": logical_partition,
                    "clean_rows": audit.total_clean_rows,
                    "quarantine_groups": audit.total_quarantine_rows,
                    "max_quarantine_by_factor": max(quarantine_counts.values(), default=0),
                }
            )
            print(
                f"preflight {index}/{len(partitions)} {logical_partition} "
                f"clean_rows={audit.total_clean_rows} "
                f"quarantine_groups={audit.total_quarantine_rows}",
                flush=True,
            )

    report = {
        "target_database": "dfs://QuantResearchV2Test",
        "factor_ids": list(factor_ids),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "attempt_nonce": args.attempt_nonce,
        "preflight": preflight_rows,
        "preflight_only": bool(args.preflight_only),
        "locked_year_read": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.preflight_only:
        print(f"preflight complete; report={args.report}")
        return

    published: list[dict[str, object]] = []
    for index, logical_partition in enumerate(partitions, start=1):
        result = publish_factor_month(
            source_session,
            target_session,
            metadata,
            factor_ids=factor_ids,
            logical_partition=logical_partition,
            fail_on_quarantine=True,
            publish_quarantine=False,
            attempt_nonce=args.attempt_nonce,
        )
        published.append(
            {
                "logical_partition": result.logical_partition,
                "data_version": result.data_version,
                "factor_count": len(result.factor_ids),
            }
        )
        print(
            f"published source {index}/{len(partitions)} {logical_partition} "
            f"data_version={result.data_version}",
            flush=True,
        )

    report["published_source"] = published
    spec = CompactObjectSpec(
        source_object="factor_observation_clean",
        target_object="factor_observation_clean_latest",
        business_keys=("signal_date", "ts_code", "factor_id", "factor_version"),
        partition_date_column="signal_date",
    )
    compacted: list[dict[str, object]] = []
    for pointer in list_current_source_pointers(target_session, "factor_observation_clean"):
        if pointer.logical_partition not in partitions:
            continue
        result = materialize_and_publish_current_partition(target_session, spec, pointer)
        compacted.append(
            {
                "logical_partition": result.logical_partition,
                "batch_id": result.batch_id,
                "row_count": result.row_count,
                "status": result.status,
            }
        )
        print(
            f"published compact {result.logical_partition} rows={result.row_count} "
            f"status={result.status}",
            flush=True,
        )
    report["compacted_factor"] = compacted
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"expanded factor publication complete; report={args.report}")


if __name__ == "__main__":
    main()
