"""Check deduplicated finite source keys against the train reference cross-section."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import defaultdict
from datetime import date

import dolphindb as ddb
import numpy as np
import pandas as pd
from record_factor_quality_snapshot import publish_snapshot

from research_v2.db.session import V2WriteSession, YesLabReadOnlySession

SCOPE_ID = "SCREEN_TRAIN2020_2024_PIT_KEY_COVERAGE_V1"
CONFLICT_SCOPE_ID = "SCREEN_TRAIN2020_2024_RAW_CONFLICT_AGG_V1"
REFERENCE_FACTOR_ID = "yeslab_0"
_YEARS = tuple(range(2020, 2025))
_GROUP_SIZE = 7


def summarize_finite_key_coverage(
    finite_keys: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    expected_factor_names: tuple[str, ...],
    minimum_day_ratio: float = 0.80,
    minimum_median_coverage_ratio: float = 0.80,
    minimum_p05_coverage_ratio: float = 0.70,
) -> pd.DataFrame:
    """Compute PIT coverage from server-side distinct finite key counts."""
    required_keys = {"factor_name", "trade_date", "finite_keys"}
    required_reference = {"signal_date", "sample_count"}
    if missing := sorted(required_keys.difference(finite_keys.columns)):
        raise ValueError(f"finite key aggregates are missing columns: {missing}")
    if missing := sorted(required_reference.difference(reference.columns)):
        raise ValueError(f"reference aggregates are missing columns: {missing}")
    expected = pd.to_datetime(reference["signal_date"], errors="raise").dt.date.drop_duplicates()
    if expected.empty:
        raise ValueError("reference has no signal dates")
    expected_set = set(expected)
    ref = reference.loc[:, ["signal_date", "sample_count"]].copy()
    ref["signal_date"] = pd.to_datetime(ref["signal_date"], errors="raise").dt.date
    ref["sample_count"] = pd.to_numeric(ref["sample_count"], errors="coerce")
    ref = ref.drop_duplicates("signal_date").set_index("signal_date")["sample_count"]
    working = finite_keys.loc[:, ["factor_name", "trade_date", "finite_keys"]].copy()
    working["factor_name"] = working["factor_name"].astype(str)
    working["trade_date"] = pd.to_datetime(working["trade_date"], errors="raise").dt.date
    working["finite_keys"] = pd.to_numeric(working["finite_keys"], errors="coerce").fillna(0)
    working = working.groupby(["factor_name", "trade_date"], as_index=False).agg(
        finite_keys=("finite_keys", "sum")
    )
    expected_months = pd.Series(sorted(expected_set)).map(lambda value: value.strftime("%Y-%m"))
    expected_month_counts = expected_months.value_counts()
    output: list[dict[str, object]] = []
    for factor_name in expected_factor_names:
        factor = working.loc[
            (working["factor_name"] == factor_name) & working["trade_date"].isin(expected_set)
        ].copy()
        factor = factor.loc[factor["finite_keys"] > 0]
        observed_dates = set(factor["trade_date"])
        ratios = (
            factor["finite_keys"].to_numpy(dtype=float)
            / factor["trade_date"].map(ref).to_numpy(dtype=float)
            if not factor.empty
            else np.array([], dtype=float)
        )
        ratios = ratios[np.isfinite(ratios)]
        median_ratio = float(np.median(ratios)) if len(ratios) else 0.0
        p05_ratio = float(np.quantile(ratios, 0.05)) if len(ratios) else 0.0
        observed_month_counts = pd.Series(sorted(observed_dates)).map(
            lambda value: value.strftime("%Y-%m")
        ).value_counts()
        monthly_ratios = [
            float(observed_month_counts.get(month, 0) / count)
            for month, count in expected_month_counts.items()
        ]
        day_ratio = len(observed_dates) / len(expected_set)
        gate = (
            day_ratio >= minimum_day_ratio
            and median_ratio >= minimum_median_coverage_ratio
            and p05_ratio >= minimum_p05_coverage_ratio
        )
        if gate:
            status = "PIT_KEY_DQ_PASS_PENDING_COMPACT_UNIVERSE"
            reasons = ["PIT_FINITE_KEY_COVERAGE_PASS", "COMPACT_UNIVERSE_CHECK_PENDING"]
        else:
            status = "PIT_KEY_DQ_FAIL"
            reasons = ["PIT_FINITE_KEY_COVERAGE_FAIL"]
            if day_ratio < minimum_day_ratio:
                reasons.append("LOW_PIT_DAY_COVERAGE")
            if median_ratio < minimum_median_coverage_ratio:
                reasons.append("LOW_MEDIAN_PIT_COVERAGE")
            if p05_ratio < minimum_p05_coverage_ratio:
                reasons.append("LOW_P05_PIT_COVERAGE")
        output.append(
            {
                "factor_name": factor_name,
                "expected_day_count": len(expected_set),
                "finite_key_day_count": len(observed_dates),
                "pit_day_ratio": day_ratio,
                "median_pit_coverage_ratio": median_ratio,
                "p05_pit_coverage_ratio": p05_ratio,
                "minimum_monthly_day_ratio": min(monthly_ratios) if monthly_ratios else 0.0,
                "finite_key_rows": int(factor["finite_keys"].sum()),
                "pit_coverage_gate_pass": gate,
                "decision_status": status,
                "reason_codes": reasons,
            }
        )
    return pd.DataFrame(output)


def _query_finite_keys(
    source: YesLabReadOnlySession,
    *,
    table: str,
    factor_names: tuple[str, ...],
    year: int,
) -> pd.DataFrame:
    names = ",".join(f"'{name.replace(chr(39), chr(39) + chr(39))}'" for name in factor_names)
    script = (
        "select factor_name,trade_date,count(*) as finite_keys from ("
        "select distinct factor_name,trade_date,ts_code from "
        f"loadTable('dfs://YesLab','{table}') where factor_name in [{names}] "
        f"and trade_date between date('{year}.01.01') and date('{year}.12.31') "
        "and isValid(value)) group by factor_name,trade_date"
    )
    result = source.query(script)
    if not isinstance(result, pd.DataFrame):
        raise RuntimeError(f"finite key query did not return a DataFrame: {table}/{year}")
    return result


def _query_resilient(
    source: YesLabReadOnlySession,
    *,
    table: str,
    factor_names: tuple[str, ...],
    year: int,
) -> pd.DataFrame:
    try:
        return _query_finite_keys(source, table=table, factor_names=factor_names, year=year)
    except RuntimeError:
        if len(factor_names) == 1:
            raise
        midpoint = len(factor_names) // 2
        return pd.concat(
            [
                _query_resilient(
                    source, table=table, factor_names=factor_names[:midpoint], year=year
                ),
                _query_resilient(
                    source, table=table, factor_names=factor_names[midpoint:], year=year
                ),
            ],
            ignore_index=True,
        )


def _record_hash(record: dict[str, object]) -> str:
    payload = {
        key: value.isoformat() if isinstance(value, (date, pd.Timestamp)) else value
        for key, value in record.items()
        if key != "record_hash"
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True).encode()
    ).hexdigest()


def _json_value(value: object) -> object:
    if isinstance(value, (date, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    if pd.isna(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", type=date.fromisoformat, default=date(2026, 8, 4))
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    client = ddb.session()
    client.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    source = YesLabReadOnlySession(client)
    target = V2WriteSession(client, target_database="dfs://QuantResearchV2Test")
    metadata = source.query(
        "select factor_id,factor_name,store_factor,category from loadTable('dfs://YesLab','factor_metadata')"
    )
    conflicts = target.query(
        "select factor_id,metrics_json from "
        "loadTable('dfs://QuantResearchV2Test','factor_quality_snapshot') "
        f"where scope_id='{CONFLICT_SCOPE_ID}'"
    )
    reference = target.query(
        "select signal_date,sample_count from "
        "loadTable('dfs://QuantResearchV2Test','factor_rank_ic_daily') "
        "where factor_id='yeslab_0' and signal_date between date('2020.01.01') "
        "and date('2024.12.31')"
    )
    if not all(isinstance(frame, pd.DataFrame) for frame in (metadata, conflicts, reference)):
        raise RuntimeError("PIT key audit inputs did not return DataFrames")
    candidate_ids = tuple(str(row.factor_id) for row in conflicts.itertuples(index=False))
    by_id = metadata.assign(
        factor_id=metadata["factor_id"].map(lambda value: f"yeslab_{int(value)}")
    ).set_index("factor_id")
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for factor_id in candidate_ids:
        meta = by_id.loc[factor_id]
        grouped[str(meta["store_factor"])].append((factor_id, str(meta["factor_name"])))
    finite_parts: list[pd.DataFrame] = []
    for table in sorted(grouped):
        entries = sorted(grouped[table])
        for offset in range(0, len(entries), _GROUP_SIZE):
            group = entries[offset : offset + _GROUP_SIZE]
            names = tuple(name for _, name in group)
            for year in _YEARS:
                print(f"query table={table} factors={len(group)} year={year}", flush=True)
                finite_parts.append(
                    _query_resilient(source, table=table, factor_names=names, year=year)
                )
    all_names = tuple(str(by_id.loc[factor_id]["factor_name"]) for factor_id in candidate_ids)
    metrics = summarize_finite_key_coverage(
        pd.concat(finite_parts, ignore_index=True),
        reference,
        expected_factor_names=all_names,
    )
    name_to_id = {
        str(by_id.loc[factor_id]["factor_name"]): factor_id for factor_id in candidate_ids
    }
    metrics["factor_id"] = metrics["factor_name"].map(name_to_id)
    watermark = hashlib.sha256(
        json.dumps(
            {"candidate_ids": candidate_ids, "years": _YEARS, "scope": SCOPE_ID},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    data_version = hashlib.sha256((watermark + str(args.as_of)).encode()).hexdigest()
    rows: list[dict[str, object]] = []
    for metric in metrics.to_dict("records"):
        factor_id = str(metric["factor_id"])
        meta = by_id.loc[factor_id]
        body = {
            "screen_version": SCOPE_ID,
            "factor_name": str(meta["factor_name"]),
            "source_table": str(meta["store_factor"]),
            "category": str(meta["category"]),
            "pit_key_coverage": {
                key: _json_value(value)
                for key, value in metric.items()
                if key not in {"factor_id", "factor_name", "decision_status", "reason_codes"}
            },
            "decision_status": metric["decision_status"],
            "reason_codes": metric["reason_codes"],
            "pit_policy": "TECHNICAL_SAME_DAY_CLOSE",
            "compact_universe_checked": False,
        }
        record = {
            "as_of_date": args.as_of,
            "factor_id": factor_id,
            "factor_version": 1,
            "scope_id": SCOPE_ID,
            "eligibility_status": (
                "PASS"
                if str(metric["decision_status"]).startswith("PIT_KEY_DQ_PASS")
                else "ELIGIBILITY_FAIL"
            ),
            "metrics_json": json.dumps(body, ensure_ascii=False, sort_keys=True),
            "rules_version": 1,
            "run_id": str(uuid.uuid4()),
            "data_version": data_version,
            "created_at": pd.Timestamp.utcnow().tz_localize(None),
            "source_watermark_id": watermark,
        }
        record["record_hash"] = _record_hash(record)
        rows.append(record)
    frame = pd.DataFrame(rows)
    print(metrics["decision_status"].value_counts().sort_index().to_string())
    if args.publish:
        print(publish_snapshot(target, frame, scope_id=SCOPE_ID))
    else:
        print("dry-run: no database write")


if __name__ == "__main__":
    main()
