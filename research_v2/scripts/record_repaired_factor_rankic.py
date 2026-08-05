"""Recompute TRAIN RankIC for the bounded turnover-repair impact window.

The existing ``factor_rank_ic_daily`` table is left untouched.  This command
recomputes only the dates affected by the published turnover candidates,
replaces those rows in a local copy of the 2020--2024 RankIC panel, and emits
post-repair HAC/FDR/ICIR evidence for the current 26-factor screen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date
from pathlib import Path

import dolphindb as ddb
import numpy as np
import pandas as pd
from record_train_daily_quality_snapshot import compute_train_metrics

from research_v2.db.session import V2WriteSession, YesLabReadOnlySession
from research_v2.evaluation.factor_rank_ic import compute_factor_daily_rank_ic

REPAIR_FACTOR_IDS = (
    "yeslab_377",
    "yeslab_380",
    "yeslab_382",
    "yeslab_385",
    "yeslab_387",
    "yeslab_388",
    "yeslab_400",
    "yeslab_439",
    "yeslab_456",
)
QUALITY_FACTOR_IDS = (
    "yeslab_114",
    "yeslab_115",
    "yeslab_117",
    "yeslab_118",
    "yeslab_163",
    "yeslab_357",
    "yeslab_359",
    "yeslab_362",
    "yeslab_363",
    "yeslab_365",
    "yeslab_366",
    "yeslab_376",
    "yeslab_377",
    "yeslab_380",
    "yeslab_381",
    "yeslab_382",
    "yeslab_383",
    "yeslab_385",
    "yeslab_387",
    "yeslab_388",
    "yeslab_391",
    "yeslab_393",
    "yeslab_396",
    "yeslab_95",
    "yeslab_96",
    "yeslab_98",
    "yeslab_400",
    "yeslab_439",
    "yeslab_456",
)
IMPACT_START = date(2020, 9, 18)
IMPACT_END = date(2020, 10, 9)


def merge_daily_rankic(existing: pd.DataFrame, repaired: pd.DataFrame) -> pd.DataFrame:
    """Replace only matching factor/date rows in an existing RankIC panel."""

    required = {"signal_date", "factor_id", "factor_version", "rank_ic", "sample_count"}
    for label, frame in (("existing", existing), ("repaired", repaired)):
        if missing := sorted(required.difference(frame.columns)):
            raise ValueError(f"{label} RankIC is missing columns: {missing}")
    key_columns = ["signal_date", "factor_id", "factor_version"]
    left = existing.loc[:, sorted(required)].copy()
    right = repaired.loc[:, sorted(required)].copy()
    for frame in (left, right):
        frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.date
        frame["factor_id"] = frame["factor_id"].astype(str)
        frame["factor_version"] = pd.to_numeric(frame["factor_version"], errors="raise").astype(int)
        frame["rank_ic"] = pd.to_numeric(frame["rank_ic"], errors="coerce")
        frame["sample_count"] = pd.to_numeric(frame["sample_count"], errors="raise").astype(int)
    if left.duplicated(key_columns).any() or right.duplicated(key_columns).any():
        raise ValueError("RankIC panels contain duplicate factor/date keys")
    overlap = left.merge(right[key_columns], on=key_columns, how="inner")
    remaining = left.merge(right[key_columns], on=key_columns, how="left", indicator=True)
    remaining = remaining.loc[remaining["_merge"].eq("left_only"), left.columns]
    result = pd.concat([remaining, right], ignore_index=True)
    if result.duplicated(key_columns).any():
        raise ValueError("RankIC replacement produced duplicate factor/date keys")
    if len(overlap) > len(right):
        raise ValueError("repaired RankIC replaces more rows than requested")
    return result.sort_values(key_columns, kind="mergesort").reset_index(drop=True)


def _query_source_factors(
    source: YesLabReadOnlySession,
    metadata: pd.DataFrame,
    start: date,
    end: date,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    normalized = metadata.copy()
    normalized["factor_id"] = normalized["factor_id"].map(lambda value: f"yeslab_{int(value)}")
    scoped = normalized.loc[normalized["factor_id"].isin(REPAIR_FACTOR_IDS)].copy()
    for table, group in scoped.groupby("store_factor", sort=True):
        names = ",".join("'" + str(name).replace("'", "''") + "'" for name in group.factor_name)
        result = source.query(
            "select trade_date,ts_code,factor_name,value from "
            f"loadTable('dfs://YesLab','{table}') "
            f"where factor_name in [{names}] and trade_date between date('{start:%Y.%m.%d}') "
            f"and date('{end:%Y.%m.%d}')"
        )
        if not isinstance(result, pd.DataFrame):
            raise RuntimeError(f"repair factor query did not return a DataFrame: {table}")
        frames.append(result.loc[:, ["trade_date", "ts_code", "factor_name", "value"]])
    if not frames:
        raise RuntimeError("repair factor metadata is empty")
    result = pd.concat(frames, ignore_index=True)
    result["signal_date"] = pd.to_datetime(result.pop("trade_date"), errors="raise").dt.date
    result["ts_code"] = result["ts_code"].astype(str)
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    result = result.loc[np.isfinite(result["value"].to_numpy(dtype=float))].copy()
    names_to_ids = {
        str(row.factor_name): str(row.factor_id) for row in scoped.itertuples(index=False)
    }
    result["factor_id"] = result["factor_name"].astype(str).map(names_to_ids)
    result = result.loc[result["factor_id"].isin(REPAIR_FACTOR_IDS)].copy()
    result["factor_version"] = 1
    key_columns = ["signal_date", "ts_code", "factor_id", "factor_version"]
    conflict = result.groupby(key_columns, as_index=False)["value"].nunique()
    if (conflict["value"] > 1).any():
        raise ValueError("repair source contains conflicting finite values")
    return result.drop_duplicates(key_columns, keep="last").loc[
        :, ["signal_date", "ts_code", "factor_id", "factor_version", "value"]
    ]


def _load_candidates(candidate_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for factor_id in REPAIR_FACTOR_IDS:
        path = candidate_dir / f"{factor_id}_turnover_gap_candidate.csv"
        if not path.is_file():
            raise FileNotFoundError(f"repair candidate is missing: {path}")
        frame = pd.read_csv(path)
        required = {
            "observation_date",
            "ts_code",
            "factor_id",
            "factor_version",
            "candidate_value",
            "source_value",
        }
        if missing := sorted(required.difference(frame.columns)):
            raise ValueError(f"repair candidate is missing columns: {missing}")
        if set(frame["factor_id"].astype(str)) != {factor_id}:
            raise ValueError(f"repair candidate factor ID mismatch: {factor_id}")
        if pd.to_numeric(frame["source_value"], errors="coerce").notna().any():
            raise ValueError(f"repair candidate overwrites finite source values: {factor_id}")
        frame = frame.rename(
            columns={"observation_date": "signal_date", "candidate_value": "value"}
        )
        frames.append(
            frame.loc[:, ["signal_date", "ts_code", "factor_id", "factor_version", "value"]]
        )
    result = pd.concat(frames, ignore_index=True)
    result["signal_date"] = pd.to_datetime(result["signal_date"], errors="raise").dt.date
    result["ts_code"] = result["ts_code"].astype(str)
    result["factor_id"] = result["factor_id"].astype(str)
    result["factor_version"] = pd.to_numeric(result["factor_version"], errors="raise").astype(int)
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    if not np.isfinite(result["value"].to_numpy(dtype=float)).all():
        raise ValueError("repair candidates contain non-finite values")
    return result


def _query_mature_labels(target: V2WriteSession, start: date, end: date) -> pd.DataFrame:
    universe = target.query(
        "select signal_date,ts_code,is_member from "
        "loadTable('dfs://QuantResearchV2Test','universe_daily') "
        "where universe_id=`cn_liquid_a and universe_version=1 "
        f"and signal_date between date('{start:%Y.%m.%d}') and date('{end:%Y.%m.%d}')"
    )
    labels = target.query(
        "select signal_date,ts_code,raw_return,label_valid from "
        "loadTable('dfs://QuantResearchV2Test','label_daily') "
        "where label_id=`forward_open_5d and label_version=1 "
        f"and signal_date between date('{start:%Y.%m.%d}') and date('{end:%Y.%m.%d}')"
    )
    if not isinstance(universe, pd.DataFrame) or not isinstance(labels, pd.DataFrame):
        raise RuntimeError("repair label/universe query did not return DataFrames")
    universe = universe.loc[universe["is_member"].fillna(False), ["signal_date", "ts_code"]]
    labels = labels.loc[labels["label_valid"].fillna(False)].copy()
    labels["raw_return"] = pd.to_numeric(labels["raw_return"], errors="coerce")
    labels = labels.loc[np.isfinite(labels["raw_return"].to_numpy(dtype=float))]
    mature = universe.merge(
        labels, on=["signal_date", "ts_code"], how="inner", validate="one_to_one"
    )
    mature["signal_date"] = pd.to_datetime(mature["signal_date"], errors="raise").dt.date
    mature["ts_code"] = mature["ts_code"].astype(str)
    return mature.loc[:, ["signal_date", "ts_code", "raw_return", "label_valid"]]


def _apply_candidate_fill(source: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    keys = ["signal_date", "ts_code", "factor_id", "factor_version"]
    source = source.copy()
    candidates = candidates.copy()
    if candidates.duplicated(keys).any():
        raise ValueError("repair candidates contain duplicate keys")
    existing = source.loc[:, keys].drop_duplicates().assign(_source=True)
    candidates = candidates.merge(existing, on=keys, how="left")
    candidates = candidates.loc[candidates["_source"].isna(), source.columns]
    result = pd.concat([source, candidates], ignore_index=True)
    if result.duplicated(keys).any():
        raise ValueError("repair overlay produced duplicate keys")
    return result


def _connect() -> tuple[YesLabReadOnlySession, V2WriteSession]:
    client = ddb.session()
    client.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    return YesLabReadOnlySession(client), V2WriteSession(
        client, target_database="dfs://QuantResearchV2Test"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path("research_v2/artifacts/diagnostics/turnover_gap_candidates_train_repaired_v3"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("research_v2/artifacts/diagnostics/repaired_factor_rankic_train26_v1"),
    )
    args = parser.parse_args()
    source, target = _connect()
    metadata = source.query(
        "select factor_id,factor_name,store_factor from "
        "loadTable('dfs://YesLab','factor_metadata')"
    )
    if not isinstance(metadata, pd.DataFrame):
        raise RuntimeError("factor metadata query did not return a DataFrame")
    impact_source = _query_source_factors(source, metadata, IMPACT_START, IMPACT_END)
    candidates = _load_candidates(args.candidate_dir)
    impact_source = _apply_candidate_fill(impact_source, candidates)
    labels = _query_mature_labels(target, IMPACT_START, IMPACT_END)
    factor_panel = impact_source.rename(columns={"factor_id": "factor_name"})
    repaired = compute_factor_daily_rank_ic(
        factor_panel,
        labels,
        minimum_cross_section=30,
        factor_name_column="factor_name",
        factor_value_column="value",
        label_column="raw_return",
    )
    repaired["factor_id"] = repaired["factor_name"].astype(str)
    repaired["factor_version"] = 1
    repaired = repaired.loc[
        :, ["signal_date", "factor_id", "factor_version", "rank_ic", "sample_count"]
    ]
    existing = target.query(
        "select signal_date,factor_id,factor_version,rank_ic,sample_count from "
        "loadTable('dfs://QuantResearchV2Test','factor_rank_ic_daily') "
        "where signal_date between date('2020.01.01') and date('2024.12.31') "
        f"and factor_id in [{','.join('`' + x for x in QUALITY_FACTOR_IDS)}] and factor_version=1"
    )
    if not isinstance(existing, pd.DataFrame):
        raise RuntimeError("existing RankIC query did not return a DataFrame")
    existing["signal_date"] = pd.to_datetime(existing["signal_date"], errors="raise").dt.date
    merged = merge_daily_rankic(existing, repaired)
    metrics = compute_train_metrics(merged)
    before = compute_train_metrics(existing)
    impact_compare = metrics.loc[metrics["factor_id"].isin(REPAIR_FACTOR_IDS)].merge(
        before.loc[before["factor_id"].isin(REPAIR_FACTOR_IDS)],
        on=["factor_id", "factor_version"],
        suffixes=("_after", "_before"),
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    repaired_path = args.output_prefix.with_name(args.output_prefix.name + "_impact.csv")
    metrics_path = args.output_prefix.with_name(args.output_prefix.name + "_metrics.csv")
    report_path = args.output_prefix.with_suffix(".json")
    repaired.to_csv(repaired_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    report = {
        "scope": "REPAIRED_FACTOR_RANKIC_TRAIN_EXTENDED_V1",
        "impact_start": IMPACT_START.isoformat(),
        "impact_end": IMPACT_END.isoformat(),
        "factor_ids": list(REPAIR_FACTOR_IDS),
        "quality_factor_ids": list(QUALITY_FACTOR_IDS),
        "candidate_rows": len(candidates),
        "repaired_rankic_rows": len(repaired),
        "existing_rankic_rows": len(existing),
        "merged_rankic_rows": len(merged),
        "changed_rankic_rows": int(
            len(
                repaired.merge(
                    existing,
                    on=["signal_date", "factor_id", "factor_version"],
                    how="inner",
                )
            )
        ),
        "metrics": metrics.loc[metrics["factor_id"].isin(REPAIR_FACTOR_IDS)].to_dict("records"),
        "before_after": impact_compare.to_dict("records"),
        "database_writes": False,
        "locked_test_read": False,
        "artifacts": {
            "impact_rankic": str(repaired_path),
            "train_metrics": str(metrics_path),
        },
        "source_hash": hashlib.sha256(
            json.dumps(
                {"candidate_rows": len(candidates), "existing_rows": len(existing)},
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
