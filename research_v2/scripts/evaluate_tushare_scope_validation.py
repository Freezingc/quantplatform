"""Re-materialize selected factors in the SH/SZ universe for validation.

The command is intentionally read-only.  It computes factor values from the
Tushare market snapshot using the current calculator, then evaluates them
against the V2 validation labels.  Stored YesLab factor values are not used as
the replay input, which makes the result suitable for a changed universe scope.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import date
from itertools import combinations
from pathlib import Path

import audit_formula_replay_consistency as replay
import numpy as np
import pandas as pd
from audit_formula_replay_tushare import (
    TushareMarketSession,
    _calculator_market_start,
    _connect,
    _fetch_tushare_market,
    _query_open_dates,
    required_market_fields,
    requires_daily_basic,
)
from audit_tushare_pit_industry import (
    PITIndustryProvider,
    VersionedPITIndustryProvider,
    coverage_for_keys,
)

from research_v2.db.session import V2WriteSession

LOCKED_TEST_START = date(2026, 1, 1)
DEFAULT_FACTOR_IDS = ("yeslab_94", "yeslab_98")
DEFAULT_DIRECTION_ARTIFACT = Path(
    "research_v2/artifacts/diagnostics/train45_oos_representative_quality_v3.csv"
)


def restrict_scope_inputs(universe: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    required_universe = {"signal_date", "ts_code", "is_member"}
    required_labels = {"signal_date", "ts_code", "raw_return", "label_valid"}
    if missing := sorted(required_universe.difference(universe.columns)):
        raise ValueError(f"universe is missing columns: {missing}")
    if missing := sorted(required_labels.difference(labels.columns)):
        raise ValueError(f"labels are missing columns: {missing}")
    left = universe.loc[universe["is_member"].fillna(False).astype(bool)].copy()
    right = labels.loc[labels["label_valid"].fillna(False).astype(bool)].copy()
    for frame, column in ((left, "signal_date"), (right, "signal_date")):
        frame[column] = pd.to_datetime(frame[column], errors="raise").dt.normalize()
        frame["ts_code"] = frame["ts_code"].astype(str)
    left = left.loc[left["ts_code"].str.endswith((".SH", ".SZ"))]
    right = right.loc[right["ts_code"].str.endswith((".SH", ".SZ"))].copy()
    right["raw_return"] = pd.to_numeric(right["raw_return"], errors="coerce")
    right = right.loc[np.isfinite(right["raw_return"].to_numpy(float))]
    keys = ["signal_date", "ts_code"]
    left = left.drop_duplicates(keys, keep="last")
    right = right.drop_duplicates(keys, keep="last")
    return left.loc[:, keys].merge(
        right.loc[:, keys + ["raw_return"]], on=keys, how="inner", validate="one_to_one"
    ).sort_values(keys, kind="mergesort").reset_index(drop=True)


def compute_daily_rank_ic(
    replay_values: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    minimum_cross_section: int = 30,
) -> pd.DataFrame:
    required_replay = {"trade_date", "ts_code", "replay_value"}
    required_labels = {"signal_date", "ts_code", "raw_return"}
    if missing := sorted(required_replay.difference(replay_values.columns)):
        raise ValueError(f"replay values are missing columns: {missing}")
    if missing := sorted(required_labels.difference(labels.columns)):
        raise ValueError(f"labels are missing columns: {missing}")
    left = replay_values.loc[:, ["trade_date", "ts_code", "replay_value"]].copy()
    right = labels.loc[:, ["signal_date", "ts_code", "raw_return"]].copy()
    left["signal_date"] = pd.to_datetime(left.pop("trade_date"), errors="raise").dt.normalize()
    right["signal_date"] = pd.to_datetime(right["signal_date"], errors="raise").dt.normalize()
    left["ts_code"] = left["ts_code"].astype(str)
    right["ts_code"] = right["ts_code"].astype(str)
    left["replay_value"] = pd.to_numeric(left["replay_value"], errors="coerce")
    right["raw_return"] = pd.to_numeric(right["raw_return"], errors="coerce")
    merged = left.merge(right, on=["signal_date", "ts_code"], how="inner", validate="one_to_one")
    merged = merged.loc[
        np.isfinite(merged["replay_value"].to_numpy(float))
        & np.isfinite(merged["raw_return"].to_numpy(float))
    ]
    rows: list[dict[str, object]] = []
    for signal_date, group in merged.groupby("signal_date", sort=True):
        if len(group) < minimum_cross_section:
            continue
        factor_rank = group["replay_value"].rank(method="average")
        label_rank = group["raw_return"].rank(method="average")
        rank_ic = factor_rank.corr(label_rank)
        if np.isfinite(rank_ic):
            rows.append(
                {
                    "signal_date": signal_date,
                    "rank_ic": float(rank_ic),
                    "sample_count": int(len(group)),
                }
            )
    return pd.DataFrame(rows, columns=["signal_date", "rank_ic", "sample_count"])


def summarize_rank_ic(daily: pd.DataFrame, *, direction: int) -> dict[str, object]:
    required = {"signal_date", "rank_ic"}
    if missing := sorted(required.difference(daily.columns)):
        raise ValueError(f"daily RankIC is missing columns: {missing}")
    if direction not in {-1, 1}:
        raise ValueError("direction must be -1 or 1")
    values = pd.to_numeric(daily["rank_ic"], errors="coerce")
    values = values.loc[np.isfinite(values.to_numpy(float))].to_numpy(float)
    if values.size == 0:
        raise ValueError("daily RankIC has no finite observations")
    signed = values * direction
    std = float(values.std(ddof=1)) if values.size > 1 else float("nan")
    return {
        "valid_day_count": int(values.size),
        "mean_rank_ic": float(values.mean()),
        "rank_ic_std": std,
        "icir": float(values.mean() / std) if np.isfinite(std) and std > 0 else float("nan"),
        "direction": int(direction),
        "signed_mean_rank_ic": float(signed.mean()),
        "directional_day_share": float((signed > 0).mean()),
        "start": pd.to_datetime(daily["signal_date"]).min().date().isoformat(),
        "end": pd.to_datetime(daily["signal_date"]).max().date().isoformat(),
    }


def compute_pair_daily_spearman(
    replay_frames: dict[str, pd.DataFrame],
    *,
    minimum_cross_section: int = 30,
) -> pd.DataFrame:
    """Summarize absolute daily Spearman similarity for replayed panels."""
    rows: list[dict[str, object]] = []
    for left_id, right_id in combinations(sorted(replay_frames), 2):
        left = replay_frames[left_id].loc[:, ["trade_date", "ts_code", "replay_value"]].copy()
        right = replay_frames[right_id].loc[:, ["trade_date", "ts_code", "replay_value"]].copy()
        for frame in (left, right):
            frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.normalize()
            frame["ts_code"] = frame["ts_code"].astype(str)
            frame["replay_value"] = pd.to_numeric(frame["replay_value"], errors="coerce")
        left = left.rename(columns={"replay_value": "left_value"})
        right = right.rename(columns={"replay_value": "right_value"})
        merged = left.merge(
            right,
            on=["trade_date", "ts_code"],
            how="inner",
            validate="one_to_one",
        )
        values: list[float] = []
        for _, group in merged.groupby("trade_date", sort=True):
            group = group.loc[
                np.isfinite(group["left_value"].to_numpy(float))
                & np.isfinite(group["right_value"].to_numpy(float))
            ]
            if len(group) < minimum_cross_section:
                continue
            corr = group["left_value"].rank(method="average").corr(
                group["right_value"].rank(method="average")
            )
            if np.isfinite(corr):
                values.append(float(corr))
        if values:
            absolute = np.abs(np.asarray(values, dtype=float))
            rows.append(
                {
                    "factor_id_left": left_id,
                    "factor_id_right": right_id,
                    "median_abs_spearman": float(np.median(absolute)),
                    "max_abs_spearman": float(np.max(absolute)),
                    "valid_day_count": len(values),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "factor_id_left",
            "factor_id_right",
            "median_abs_spearman",
            "max_abs_spearman",
            "valid_day_count",
        ],
    )


def _read_current_compact_range(
    session: V2WriteSession,
    *,
    object_name: str,
    selected_columns: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Read every current monthly batch in a date range with one DDB response."""
    if object_name not in {"universe_daily_latest", "label_daily_latest"}:
        raise ValueError(f"unsupported compact range object: {object_name}")
    if start > end:
        raise ValueError("range start must not be after end")
    result = session.query(
        "selectedBatches = select top 1 batch_id,status from "
        f"loadTable('{session.target_database}', 'latest_manifest') "
        f"where object_name=`{object_name} and "
        f"logical_partition between '{start:%Y-%m}':'{end:%Y-%m}' "
        "context by logical_partition csort published_at desc\n"
        "publishedBatches = select batch_id from selectedBatches where status=`PUBLISHED\n"
        f"select {selected_columns} from loadTable('{session.target_database}', '{object_name}') "
        "where batch_id in publishedBatches.batch_id and "
        f"signal_date between date('{start:%Y.%m.%d}') and date('{end:%Y.%m.%d}')"
    )
    if not isinstance(result, pd.DataFrame):
        raise RuntimeError(f"compact range query did not return a frame: {object_name}")
    return result.copy(deep=True)


def _read_current_universe_range(
    session: V2WriteSession, *, start: date, end: date
) -> pd.DataFrame:
    frame = _read_current_compact_range(
        session,
        object_name="universe_daily_latest",
        selected_columns="batch_id,signal_date,ts_code,is_member",
        start=start,
        end=end,
    )
    required = {"batch_id", "signal_date", "ts_code", "is_member"}
    if not required.issubset(frame.columns):
        raise RuntimeError("universe range query returned an invalid frame")
    return frame.loc[:, ["batch_id", "signal_date", "ts_code", "is_member"]].copy()


def _read_current_label_range(
    session: V2WriteSession, *, start: date, end: date
) -> pd.DataFrame:
    frame = _read_current_compact_range(
        session,
        object_name="label_daily_latest",
        selected_columns="batch_id,signal_date,ts_code,raw_return,int(label_valid) as label_valid",
        start=start,
        end=end,
    )
    required = {"batch_id", "signal_date", "ts_code", "raw_return", "label_valid"}
    if not required.issubset(frame.columns):
        raise RuntimeError("label range query returned an invalid frame")
    frame = frame.loc[
        :, ["batch_id", "signal_date", "ts_code", "raw_return", "label_valid"]
    ].copy()
    encoded = pd.to_numeric(frame["label_valid"], errors="coerce")
    if encoded.isna().any() or not encoded.isin((0, 1)).all():
        raise ValueError("label range has non-boolean validity values")
    frame["label_valid"] = encoded.astype(bool)
    return frame


def _load_validation_inputs(client: object, start: date, end: date) -> pd.DataFrame:
    session = V2WriteSession(client, target_database="dfs://QuantResearchV2Test")
    universe = _read_current_universe_range(session, start=start, end=end)
    label = _read_current_label_range(session, start=start, end=end)
    label_keys = ["batch_id", "signal_date", "ts_code"]
    if label.duplicated(label_keys).any():
        raise ValueError("label range has duplicate business keys")
    return restrict_scope_inputs(
        universe, label
    )


def _load_directions(path: Path, factor_ids: Sequence[str]) -> dict[str, int]:
    frame = pd.read_csv(path)
    direction_column = next(
        (
            column
            for column in ("validation_direction", "frozen_direction", "direction")
            if column in frame.columns
        ),
        None,
    )
    missing = []
    if "factor_id" not in frame.columns:
        missing.append("factor_id")
    if direction_column is None:
        missing.append("validation_direction (or frozen_direction/direction)")
    if missing:
        raise ValueError(f"direction artifact is missing columns: {missing}")
    selected = frame.loc[frame["factor_id"].astype(str).isin(set(factor_ids))]
    directions = {
        str(row.factor_id): int(getattr(row, direction_column))
        for row in selected.itertuples(index=False)
    }
    missing = sorted(set(factor_ids).difference(directions))
    if missing:
        raise ValueError(f"direction artifact is missing factors: {missing}")
    return directions


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2025, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--factor-id", action="append")
    parser.add_argument("--stored-factor-id", action="append")
    parser.add_argument("--direction-artifact", type=Path, default=DEFAULT_DIRECTION_ARTIFACT)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("research_v2/artifacts/tushare_cache"),
    )
    parser.add_argument(
        "--pit-industry-cache",
        type=Path,
        help="cached Tushare SW2021 L1 membership files for date-aware industry replay",
    )
    parser.add_argument(
        "--pit-industry-cutover",
        type=date.fromisoformat,
        help="use SW2014 before this date and SW2021 from this date onward",
    )
    parser.add_argument(
        "--skip-similarity",
        action="store_true",
        help="write RankIC quality only; defer the expensive all-pairs similarity calculation",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.factor_id = tuple(
        dict.fromkeys(map(str, args.factor_id or DEFAULT_FACTOR_IDS))
    )
    args.stored_factor_id = tuple(dict.fromkeys(map(str, args.stored_factor_id or ())))
    unsupported_stored = sorted(
        set(args.stored_factor_id).difference(replay.REPLAY_DEFINITIONS)
    )
    if unsupported_stored:
        parser.error(f"unsupported stored factor ids: {unsupported_stored}")
    if not args.factor_id:
        parser.error("at least one factor id is required")
    if args.start > args.end or args.end >= LOCKED_TEST_START:
        parser.error("evaluation range must be before 2026 and start <= end")
    if args.pit_industry_cutover is not None and args.pit_industry_cache is None:
        parser.error("--pit-industry-cutover requires --pit-industry-cache")
    if args.pit_industry_cutover is not None and args.pit_industry_cutover >= LOCKED_TEST_START:
        parser.error("industry cutover must be before 2026")
    return args


def run_evaluation(args: argparse.Namespace) -> dict[str, object]:
    client = _connect()
    inputs = _load_validation_inputs(client, args.start, args.end)
    codes = tuple(sorted(inputs["ts_code"].astype(str).unique()))
    definitions = [replay.REPLAY_DEFINITIONS[factor_id] for factor_id in args.factor_id]
    market_start = min(
        _calculator_market_start(client, definition, args.start, args.end)
        for definition in definitions
    )
    open_dates = _query_open_dates(client, market_start, args.end)
    required_fields = required_market_fields(definitions)
    include_daily_basic = requires_daily_basic(required_fields)
    include_adjusted_fields = bool(
        {"open_hfq", "high_hfq", "low_hfq", "close_hfq", "pre_close_hfq"}
        & set(required_fields)
    )
    market = _fetch_tushare_market(
        codes,
        market_start,
        args.end,
        token=os.getenv("QRV2_TUSHARE_TOKEN", ""),
        open_dates=open_dates,
        cache_dir=args.cache_dir,
        include_daily_basic=include_daily_basic,
        include_adjusted_fields=include_adjusted_fields,
    )
    directions = _load_directions(args.direction_artifact, args.factor_id)
    daily_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    replay_frames: dict[str, pd.DataFrame] = {}
    if args.pit_industry_cache is None:
        industry_provider = None
    elif args.pit_industry_cutover is None:
        industry_provider = PITIndustryProvider.from_cache(args.pit_industry_cache)
    else:
        industry_provider = VersionedPITIndustryProvider.from_cache(
            args.pit_industry_cache,
            cutover=args.pit_industry_cutover,
        )
    industry_coverage = None
    if industry_provider is not None:
        industry_coverage = coverage_for_keys(
            industry_provider,
            inputs.loc[:, ["signal_date", "ts_code"]],
        )
    for definition in definitions:
        replay_values = replay._replay_factor(
            TushareMarketSession(client, market, codes),
            definition,
            args.start,
            args.end,
            industry_provider=industry_provider,
        )
        replay_frames[definition.factor_id] = replay_values
        daily = compute_daily_rank_ic(replay_values, inputs)
        daily.insert(0, "factor_id", definition.factor_id)
        daily_parts.append(daily)
        summaries.append(
            {
                "factor_id": definition.factor_id,
                "factor_name": definition.factor_name,
                **summarize_rank_ic(daily, direction=directions[definition.factor_id]),
            }
        )
    stored_factor_ids: list[str] = []
    for factor_id in args.stored_factor_id:
        if factor_id in replay_frames:
            continue
        definition = replay.REPLAY_DEFINITIONS[factor_id]
        stored = replay._query_source(client, definition, args.start, args.end)
        stored = stored.loc[stored["ts_code"].astype(str).str.endswith((".SH", ".SZ"))].copy()
        stored = stored.rename(columns={"value": "replay_value", "trade_date": "trade_date"})
        replay_frames[factor_id] = stored.loc[:, ["trade_date", "ts_code", "replay_value"]]
        stored_factor_ids.append(factor_id)
    daily_frame = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    similarity = (
        pd.DataFrame(
            columns=[
                "factor_id_left",
                "factor_id_right",
                "median_abs_spearman",
                "max_abs_spearman",
                "valid_day_count",
            ]
        )
        if args.skip_similarity
        else compute_pair_daily_spearman(replay_frames)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    daily_frame.to_csv(args.output / "daily_rank_ic.csv", index=False)
    similarity.to_json(args.output / "similarity_summary.json", orient="records", indent=2)
    (args.output / "quality_summary.json").write_text(
        json.dumps(
            {
                "scope": "TUSHARE_SHSZ_REMATERIALIZED_VALIDATION_V1",
                "start": args.start.isoformat(),
                "end": args.end.isoformat(),
                "factor_ids": list(args.factor_id),
                "stored_factor_ids": stored_factor_ids,
                "code_count": len(codes),
                "market_start": market_start.isoformat(),
                "market_source": "Tushare.daily",
                "industry_source": (
                    "Tushare SW2014/SW2021 versioned PIT membership"
                    if args.pit_industry_cutover is not None
                    else (
                        "Tushare SW2021 L1 PIT membership"
                        if industry_provider is not None
                        else "YesLab.stock_info_table static labels"
                    )
                ),
                "industry_coverage": industry_coverage,
                "similarity_skipped": bool(args.skip_similarity),
                "excluded_exchanges": ["BJ"],
                "database_writes": False,
                "locked_test_read": False,
                "summaries": summaries,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {"summaries": summaries, "code_count": len(codes)}


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    return run_evaluation(parse_args(argv))


if __name__ == "__main__":
    main()
