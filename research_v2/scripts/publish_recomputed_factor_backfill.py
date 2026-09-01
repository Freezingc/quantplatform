"""Preflight and publish recomputed 2014-2019 TRAIN-year factor observations.

The stored YesLab factor tables begin at 2020-01-02, so the TRAIN years of the
F01-F06 walk-forward cannot be backfilled from storage.  This script recomputes
the replay-verified admitted factors from raw market inputs (the registered GP
calculators) and publishes them through the same audited month-scoped backfill
path used for stored factors.  Only factors proven to replay stored values at
machine precision are routed (see
docs/reports/2026-09-01-admitted16-formula-replay-verification.md).

The script is explicit about scope and target: it first materializes every
month in memory and rejects missing-factor or blocking quarantines; only after
all months pass does it append immutable source rows and advance the V2
pointers.  Existing source versions are never deleted.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
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
from research_v2.factors.recomputed_source import (
    RecomputedFactorSourceSession,
    RecomputeRoute,
)
from research_v2.factors.source_backfill import (
    FactorBackfillError,
    preflight_factor_month,
    publish_factor_month,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CALCULATOR_ROOT = REPO_ROOT / "数据库"

# Replay-verified admitted factors: yeslab_id -> (GP number, verified lookback).
VERIFIED_FACTOR_LOOKBACKS: dict[str, tuple[int, int]] = {
    "yeslab_376": (7, 10),
    "yeslab_377": (8, 1),
    "yeslab_380": (11, 1),
    "yeslab_381": (12, 5),
    "yeslab_387": (18, 2),
    "yeslab_388": (19, 10),
    "yeslab_391": (22, 60),
    "yeslab_396": (27, 5),
    "yeslab_400": (31, 60),
    "yeslab_405": (36, 5),
    "yeslab_425": (56, 30),
    "yeslab_432": (63, 5),
    "yeslab_433": (64, 70),
    "yeslab_442": (73, 61),
}


def _connect() -> tuple[object, object]:
    host = os.getenv("QRV2_DDB_HOST", "10.176.55.211")
    port = int(os.getenv("QRV2_DDB_PORT", "8848"))
    username = os.getenv("QRV2_DDB_USERNAME", "xiahongjie")
    password = os.getenv("QRV2_DDB_PASSWORD", "123456")
    market_client = ddb.session()
    market_client.connect(host, port, username, password)
    target_client = ddb.session()
    target_client.connect(host, port, username, password)
    return market_client, target_client


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


def _build_routes(
    metadata: pd.DataFrame, factor_ids: tuple[str, ...]
) -> dict[str, RecomputeRoute]:
    if str(CALCULATOR_ROOT) not in sys.path:
        sys.path.insert(0, str(CALCULATOR_ROOT))
    display_by_id = {
        str(int(row.factor_id)): str(row.factor_name)
        for row in metadata.itertuples(index=False)
    }
    routes: dict[str, RecomputeRoute] = {}
    for factor_id in factor_ids:
        gp_number, lookback = VERIFIED_FACTOR_LOOKBACKS[factor_id]
        display_name = display_by_id.get(factor_id.removeprefix("yeslab_"))
        if display_name is None:
            raise FactorBackfillError(f"metadata is missing factor {factor_id}")
        module_name = f"GP_Factor_{gp_number:03d}_Calculator"
        module = importlib.import_module(f"factor_calculators.{module_name}")
        calculator_class = getattr(module, module_name)
        routes[display_name] = RecomputeRoute(
            source_table=f"factor_data{_source_table_suffix(metadata, factor_id)}",
            lookback_days=lookback,
            factory=lambda client, lookback_days, cls=calculator_class: cls(
                client, lookback_days
            ),
        )
    return routes


def _source_table_suffix(metadata: pd.DataFrame, factor_id: str) -> str:
    numeric = factor_id.removeprefix("yeslab_")
    rows = metadata.loc[metadata["factor_id"].astype(str).str.lstrip("0") == numeric.lstrip("0")]
    if rows.empty:
        raise FactorBackfillError(f"metadata is missing factor {factor_id}")
    store_factor = str(rows.iloc[0]["store_factor"])
    suffix = store_factor.removeprefix("factor_data")
    if suffix not in {"8", "9", "10"}:
        raise FactorBackfillError(f"unexpected store_factor {store_factor} for {factor_id}")
    return suffix


def _months(start_date: date, end_date: date) -> tuple[str, ...]:
    period = pd.period_range(start_date, end_date, freq="M")
    return tuple(month.strftime("%Y-%m") for month in period)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2019-12-31")
    parser.add_argument("--attempt-nonce", default="recomputed-train-years-14-v1")
    parser.add_argument("--factor-ids", default=",".join(VERIFIED_FACTOR_LOOKBACKS))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    start_date = _parse_date(args.start)
    end_date = _parse_date(args.end)
    factor_ids = tuple(value.strip() for value in args.factor_ids.split(",") if value.strip())
    unsupported = sorted(set(factor_ids).difference(VERIFIED_FACTOR_LOOKBACKS))
    if unsupported:
        raise FactorBackfillError(
            f"factors without replay verification cannot be recomputed: {unsupported}"
        )
    if start_date > end_date or not factor_ids:
        raise ValueError("invalid date range or factor list")
    if len(set(factor_ids)) != len(factor_ids):
        raise ValueError("factor_ids must be unique")
    if not args.attempt_nonce.strip():
        raise ValueError("attempt_nonce cannot be empty")

    market_client, target_client = _connect()
    metadata_session = YesLabReadOnlySession(market_client)
    target_session = V2WriteSession(target_client, target_database="dfs://QuantResearchV2Test")
    metadata = _metadata(metadata_session, factor_ids)
    routes = _build_routes(metadata, factor_ids)
    source_session = RecomputedFactorSourceSession(market_client, routes)
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
            raise FactorBackfillError("preflight report does not match requested scope")
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
        "source_mode": "recomputed-from-raw-market",
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
                "data_version": result.data_version,
            }
        )
    report["compacted_latest"] = compacted
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"recomputed backfill complete; report={args.report}", flush=True)


if __name__ == "__main__":
    main()
