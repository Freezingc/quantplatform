"""Full TRAIN-day cross-sectional similarity for 2025-validated candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import defaultdict
from datetime import date
from pathlib import Path

import dolphindb as ddb
import numpy as np
import pandas as pd
from record_factor_quality_snapshot import publish_snapshot
from record_train_similarity_sample import _query_resilient, classify_similarity_pairs

from research_v2.db.session import V2WriteSession, YesLabReadOnlySession
from research_v2.factors.source_backfill import build_factor_definitions

SCOPE_ID = "SCREEN_TRAIN2020_2024_SIMILARITY_FULL_DAY26_V2"
VALIDATION_SCOPE_ID = "SCREEN_TRAIN2020_2024_VALIDATION2025_UNIQUE21_V1"
# Query each source table in one batch where possible.  ``_query_resilient``
# still halves a group after a DolphinDB failure, so this lowers round trips
# without weakening the OOM fallback.
_FACTOR_GROUP_SIZE = 13
_MIN_COMMON_MEMBERS = 30
_THRESHOLD = 0.90
_MIN_VALID_DAYS = 60


def apply_candidate_overlay(
    observations: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Add candidate values only for keys absent from finite observations."""

    required = {"signal_date", "ts_code", "factor_id", "factor_version", "value"}
    for label, frame in (("observations", observations), ("candidates", candidates)):
        if missing := sorted(required.difference(frame.columns)):
            raise ValueError(f"{label} are missing columns: {missing}")
    source = observations.loc[:, sorted(required)].copy()
    candidate = candidates.loc[:, sorted(required)].copy()
    for frame in (source, candidate):
        frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.date
        frame["ts_code"] = frame["ts_code"].astype(str)
        frame["factor_id"] = frame["factor_id"].astype(str)
        frame["factor_version"] = pd.to_numeric(frame["factor_version"], errors="raise").astype(int)
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    key_columns = ["signal_date", "ts_code", "factor_id", "factor_version"]
    if source.duplicated(key_columns).any():
        raise ValueError("observations contain duplicate factor keys")
    if candidate.duplicated(
        ["signal_date", "ts_code", "factor_id", "factor_version"]
    ).any():
        raise ValueError("candidate overlay contains duplicate factor keys")
    if not np.isfinite(candidate["value"].to_numpy(dtype=float)).all():
        raise ValueError("candidate overlay contains non-finite values")
    existing = source.loc[:, key_columns].drop_duplicates().assign(_source_present=True)
    candidate = candidate.merge(existing, on=key_columns, how="left")
    candidate = candidate.loc[candidate["_source_present"].isna(), source.columns]
    result = pd.concat([source, candidate], ignore_index=True)
    if result.duplicated(key_columns).any():
        raise ValueError("candidate overlay produced duplicate factor keys")
    return result.sort_values(key_columns, kind="mergesort").reset_index(drop=True)


def load_candidate_overlay(
    candidate_dir: Path,
    factor_ids: tuple[str, ...],
) -> pd.DataFrame:
    """Load candidate CSVs into the same panel schema as cleaned observations."""

    frames: list[pd.DataFrame] = []
    required = {
        "observation_date",
        "ts_code",
        "factor_id",
        "factor_version",
        "candidate_value",
        "source_value",
    }
    for factor_id in factor_ids:
        path = candidate_dir / f"{factor_id}_turnover_gap_candidate.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if missing := sorted(required.difference(frame.columns)):
            raise ValueError(f"candidate overlay is missing columns: {missing}")
        if set(frame["factor_id"].astype(str)) != {factor_id}:
            raise ValueError(f"candidate overlay factor ID mismatch: {factor_id}")
        if pd.to_numeric(frame["source_value"], errors="coerce").notna().any():
            raise ValueError(f"candidate overlay overwrites finite source values: {factor_id}")
        frame = frame.rename(
            columns={"observation_date": "signal_date", "candidate_value": "value"}
        )
        frame = frame.loc[:, ["signal_date", "ts_code", "factor_id", "factor_version", "value"]]
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=["signal_date", "ts_code", "factor_id", "factor_version", "value"]
        )
    result = pd.concat(frames, ignore_index=True)
    result["signal_date"] = pd.to_datetime(result["signal_date"], errors="raise").dt.date
    if any(value >= date(2026, 1, 1) for value in result["signal_date"]):
        raise ValueError("candidate overlay contains locked-test dates")
    result["factor_version"] = pd.to_numeric(result["factor_version"], errors="raise").astype(int)
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    if not np.isfinite(result["value"].to_numpy(dtype=float)).all():
        raise ValueError("candidate overlay contains non-finite values")
    return result


def query_clean_factor_month(
    session: object,
    *,
    factor_ids: tuple[str, ...],
    start: date,
    end: date,
) -> pd.DataFrame:
    """Read one month from the already deduplicated V2 factor panel."""

    if start > end:
        raise ValueError("clean factor month start must not be after end")
    if not factor_ids:
        raise ValueError("factor_ids cannot be empty")
    if any(not factor_id.startswith("yeslab_") for factor_id in factor_ids):
        raise ValueError("factor IDs must use yeslab_ symbols")
    logical_partition = start.strftime("%Y-%m")
    pointer_script = (
        "select top 1 data_version from "
        "loadTable('dfs://QuantResearchV2Test','published_version') "
        "where object_name=`factor_observation_clean "
        f"and logical_partition='{logical_partition}' order by published_at desc"
    )
    if not hasattr(session, "query"):
        raise TypeError("session must provide query")
    pointer = session.query(pointer_script)
    if not isinstance(pointer, pd.DataFrame) or pointer.empty:
        raise RuntimeError(f"missing V2 clean pointer for {logical_partition}")
    if "data_version" not in pointer.columns:
        raise RuntimeError("V2 clean pointer is missing data_version")
    data_version = str(pointer.iloc[0]["data_version"])
    if not data_version or data_version == "nan":
        raise RuntimeError(f"V2 clean pointer has empty data_version for {logical_partition}")
    literals = ",".join(f"`{factor_id}" for factor_id in factor_ids)
    script = (
        "select signal_date,ts_code,factor_id,factor_version,value from "
        "loadTable('dfs://QuantResearchV2Test','factor_observation_clean') "
        f"where signal_date between date('{start:%Y.%m.%d}') "
        f"and date('{end:%Y.%m.%d}') and factor_id in [{literals}] "
        f"and factor_version=1 and data_version='{data_version}'"
    )
    result = session.query(script)
    if not isinstance(result, pd.DataFrame):
        raise RuntimeError("V2 clean factor query did not return a DataFrame")
    required = {"signal_date", "ts_code", "factor_id", "factor_version", "value"}
    if missing := sorted(required.difference(result.columns)):
        raise RuntimeError(f"V2 clean factor query is missing columns: {missing}")
    frame = result.loc[:, sorted(required)].copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.date
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["factor_id"] = frame["factor_id"].astype(str)
    frame["factor_version"] = pd.to_numeric(frame["factor_version"], errors="raise").astype(int)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.loc[np.isfinite(frame["value"].to_numpy(dtype=float))].copy()
    key_columns = ["signal_date", "ts_code", "factor_id", "factor_version"]
    if frame.duplicated(key_columns).any():
        raise RuntimeError("V2 clean factor query contains duplicate business keys")
    return frame.sort_values(key_columns, kind="mergesort").reset_index(drop=True)


def write_pair_artifact(pairs: pd.DataFrame, path: Path) -> None:
    """Persist the complete pairwise similarity matrix for audit/replay."""

    required = {
        "factor_id_left",
        "factor_id_right",
        "median_daily_abs_spearman",
        "valid_days",
    }
    if missing := sorted(required.difference(pairs.columns)):
        raise ValueError(f"similarity pairs are missing columns: {missing}")
    if pairs.duplicated(["factor_id_left", "factor_id_right"]).any():
        raise ValueError("similarity pairs contain duplicate ordered keys")
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs.loc[:, [
        "factor_id_left",
        "factor_id_right",
        "median_daily_abs_spearman",
        "valid_days",
    ]].to_csv(path, index=False)


def daily_similarity_values(
    observations: pd.DataFrame, factor_ids: tuple[str, ...]
) -> dict[tuple[str, str], list[float]]:
    """Return absolute daily Spearman values for every factor pair."""
    required = {"signal_date", "ts_code", "factor_id", "factor_version", "value"}
    if missing := sorted(required.difference(observations.columns)):
        raise ValueError(f"similarity observations are missing columns: {missing}")
    pivot = observations.pivot_table(
        index=["signal_date", "ts_code"],
        columns="factor_id",
        values="value",
        aggfunc="first",
    ).reindex(columns=list(factor_ids))
    values_by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for _, day in pivot.groupby(level="signal_date", sort=True):
        values = day.droplevel("signal_date")
        finite = values.notna().to_numpy(dtype=np.int64)
        common = finite.T @ finite
        ranks = values.rank(method="average", axis=0)
        correlations = ranks.corr(min_periods=_MIN_COMMON_MEMBERS)
        for left_index, left_id in enumerate(factor_ids):
            for right_index in range(left_index + 1, len(factor_ids)):
                right_id = factor_ids[right_index]
                if common[left_index, right_index] < _MIN_COMMON_MEMBERS:
                    continue
                correlation = correlations.iloc[left_index, right_index]
                if pd.notna(correlation) and np.isfinite(float(correlation)):
                    values_by_pair[(left_id, right_id)].append(abs(float(correlation)))
    return values_by_pair


def vectorized_clean_technical_sample(
    raw: pd.DataFrame, definitions: tuple[object, ...]
) -> pd.DataFrame:
    """Deduplicate finite technical rows once for this read-only diagnostic.

    The upstream full-range conflict scope proves that a candidate has no
    multiple distinct values for a security/date key.  Filtering finite values
    first therefore reproduces the clean technical-key panel without invoking
    the lineage-heavy publication materializer once per factor.
    """
    required = {"ts_code", "trade_date", "factor_name", "value"}
    if missing := sorted(required.difference(raw.columns)):
        raise ValueError(f"raw similarity sample is missing columns: {missing}")
    name_to_id = {
        str(definition.display_name): str(definition.factor_id)
        for definition in definitions
    }
    frame = raw.loc[raw["factor_name"].astype(str).isin(name_to_id)].copy()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.loc[np.isfinite(frame["value"].to_numpy(dtype=float))].copy()
    frame["signal_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.date
    frame["factor_id"] = frame["factor_name"].astype(str).map(name_to_id)
    frame = frame.drop_duplicates(
        ["signal_date", "ts_code", "factor_id"], keep="first"
    )
    return frame.loc[:, ["signal_date", "ts_code", "factor_id"]].assign(
        factor_version=1, value=frame["value"].to_numpy()
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", type=date.fromisoformat, default=date(2026, 8, 4))
    parser.add_argument(
        "--factor-ids",
        help="comma-separated factor IDs; defaults to validation-passing candidates",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        help="optional turnover-repair candidate CSV directory",
    )
    parser.add_argument(
        "--clean-source",
        action="store_true",
        help="read the deduplicated V2 factor_observation_clean panel",
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        help="optionally persist the complete factor-pair similarity matrix",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research_v2/artifacts/diagnostics/train_similarity_full_day_2020_2024.csv"),
    )
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
        "select factor_id,factor_name,creator,update_frequency,store_factor,category "
        "from loadTable('dfs://YesLab','factor_metadata')"
    )
    validation = target.query(
        "select factor_id,metrics_json from "
        "loadTable('dfs://QuantResearchV2Test','factor_quality_snapshot') "
        f"where scope_id='{VALIDATION_SCOPE_ID}'"
    )
    reference = target.query(
        "select signal_date from loadTable('dfs://QuantResearchV2Test','factor_rank_ic_daily') "
        "where factor_id='yeslab_0' and signal_date between date('2020.01.01') "
        "and date('2024.12.31')"
    )
    if not all(isinstance(frame, pd.DataFrame) for frame in (metadata, validation, reference)):
        raise RuntimeError("full-day similarity inputs did not return DataFrames")
    if args.factor_ids:
        factor_ids = tuple(sorted(set(value.strip() for value in args.factor_ids.split(","))))
    else:
        factor_ids = tuple(
            sorted(
                str(row.factor_id)
                for row in validation.itertuples(index=False)
                if json.loads(str(row.metrics_json)).get("decision_status")
                == "VALIDATION_2025_PASS_PENDING_FINAL_SIMILARITY"
            )
        )
    if not factor_ids:
        raise RuntimeError("no validation-passing candidates found")
    metadata_ids = {
        f"yeslab_{int(value)}" for value in pd.to_numeric(metadata["factor_id"], errors="raise")
    }
    missing_factor_ids = sorted(set(factor_ids).difference(metadata_ids))
    if missing_factor_ids:
        raise ValueError(f"factor metadata is missing requested IDs: {missing_factor_ids}")
    candidate_overlay = (
        load_candidate_overlay(args.candidate_dir, factor_ids)
        if args.candidate_dir is not None
        else pd.DataFrame(
            columns=["signal_date", "ts_code", "factor_id", "factor_version", "value"]
        )
    )
    dates = pd.to_datetime(reference["signal_date"], errors="raise").dt.date
    month_dates: list[tuple[str, tuple[date, ...]]] = []
    date_frame = pd.DataFrame({"signal_date": dates})
    month_keys = date_frame["signal_date"].map(lambda value: value.strftime("%Y-%m"))
    for month, group in date_frame.groupby(month_keys, sort=True):
        month_dates.append((str(month), tuple(sorted(set(group["signal_date"])))) )
    definitions = build_factor_definitions(metadata, factor_ids)
    grouped: dict[str, list] = defaultdict(list)
    for definition in definitions:
        grouped[definition.source_table].append(definition)
    pair_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for month_index, (logical_month, signal_dates) in enumerate(month_dates, start=1):
        if args.clean_source:
            month_observations = query_clean_factor_month(
                target,
                factor_ids=factor_ids,
                start=signal_dates[0],
                end=signal_dates[-1],
            )
            clean_factor_ids = set(month_observations["factor_id"].astype(str))
            missing_factor_ids = tuple(
                factor_id for factor_id in factor_ids if factor_id not in clean_factor_ids
            )
            if missing_factor_ids:
                fallback_parts: list[pd.DataFrame] = []
                for table in sorted(grouped):
                    entries = [
                        definition
                        for definition in grouped[table]
                        if definition.factor_id in missing_factor_ids
                    ]
                    for offset in range(0, len(entries), _FACTOR_GROUP_SIZE):
                        group = tuple(entries[offset : offset + _FACTOR_GROUP_SIZE])
                        raw = _query_resilient(
                            source,
                            table=table,
                            factor_names=tuple(definition.display_name for definition in group),
                            dates=signal_dates,
                        )
                        fallback_parts.append(vectorized_clean_technical_sample(raw, group))
                if fallback_parts:
                    month_observations = pd.concat(
                        [month_observations, *fallback_parts], ignore_index=True
                    )
        else:
            month_parts: list[pd.DataFrame] = []
            for table in sorted(grouped):
                entries = sorted(grouped[table], key=lambda definition: definition.factor_id)
                for offset in range(0, len(entries), _FACTOR_GROUP_SIZE):
                    group = tuple(entries[offset : offset + _FACTOR_GROUP_SIZE])
                    raw = _query_resilient(
                        source,
                        table=table,
                        factor_names=tuple(definition.display_name for definition in group),
                        dates=signal_dates,
                    )
                    month_parts.append(vectorized_clean_technical_sample(raw, group))
            month_observations = pd.concat(month_parts, ignore_index=True)
        month_candidates = candidate_overlay.loc[
            candidate_overlay["signal_date"].isin(signal_dates)
            & candidate_overlay["factor_id"].isin(factor_ids)
        ].copy()
        if not month_candidates.empty:
            month_observations = apply_candidate_overlay(month_observations, month_candidates)
        for pair, values in daily_similarity_values(month_observations, factor_ids).items():
            pair_values[pair].extend(values)
        print(
            f"full-day similarity month={logical_month} "
            f"progress={month_index}/{len(month_dates)} rows={len(month_observations)}",
            flush=True,
        )
    pair_rows = []
    for left_index, left_id in enumerate(factor_ids):
        for right_id in factor_ids[left_index + 1 :]:
            values = pair_values[(left_id, right_id)]
            pair_rows.append(
                {
                    "factor_id_left": left_id,
                    "factor_id_right": right_id,
                    "median_daily_abs_spearman": float(np.median(values))
                    if values
                    else np.nan,
                    "valid_days": len(values),
                }
            )
    pairs = pd.DataFrame(pair_rows)
    if args.pairs_output is not None:
        write_pair_artifact(pairs, args.pairs_output)
    classified = classify_similarity_pairs(
        pairs,
        factor_ids=factor_ids,
        threshold=_THRESHOLD,
        minimum_valid_days=_MIN_VALID_DAYS,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    classified.to_csv(args.output, index=False)
    print(
        f"full-day similarity complete: factors={len(factor_ids)} "
        f"dates={len(dates)} output={args.output}",
        flush=True,
    )
    watermark = hashlib.sha256(
        json.dumps(
            {"factor_ids": factor_ids, "dates": len(dates), "scope": SCOPE_ID},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    data_version = hashlib.sha256((watermark + str(args.as_of)).encode()).hexdigest()
    metadata_by_id = metadata.assign(
        factor_id=metadata["factor_id"].map(lambda value: f"yeslab_{int(value)}")
    ).set_index("factor_id")
    records: list[dict[str, object]] = []
    for metric in classified.to_dict("records"):
        factor_id = str(metric["factor_id"])
        meta = metadata_by_id.loc[factor_id]
        body = {
            "screen_version": SCOPE_ID,
            "factor_name": str(meta["factor_name"]),
            "source_table": str(meta["store_factor"]),
            "category": str(meta["category"]),
            "evidence_scope": {
                "start": "2020-01-01",
                "end": "2024-12-31",
                "valid_day_count": len(dates),
            },
            "similarity": metric,
            "decision_status": metric["decision_status"],
            "reason_codes": metric["reason_codes"],
            "full_day_cross_section_checked": True,
            "candidate_overlay_applied": bool(not candidate_overlay.empty),
            "factor_value_source": (
                "V2_clean_pointer_plus_YesLab_fallback"
                if args.clean_source
                else "YesLab_source"
            ),
        }
        record = {
            "as_of_date": args.as_of,
            "factor_id": factor_id,
            "factor_version": 1,
            "scope_id": SCOPE_ID,
            "eligibility_status": "PASS",
            "metrics_json": json.dumps(body, ensure_ascii=False, sort_keys=True),
            "rules_version": 1,
            "run_id": str(uuid.uuid4()),
            "data_version": data_version,
            "created_at": pd.Timestamp.utcnow().tz_localize(None),
            "source_watermark_id": watermark,
        }
        record["record_hash"] = _record_hash(record)
        records.append(record)
    print(classified["decision_status"].value_counts().sort_index().to_string())
    if args.publish:
        print(publish_snapshot(target, pd.DataFrame(records), scope_id=SCOPE_ID))
    else:
        print("dry-run: no database write")


if __name__ == "__main__":
    main()
