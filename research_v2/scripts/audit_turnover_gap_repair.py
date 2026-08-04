"""Audit a bounded Tushare repair candidate for the 2020 turnover-rate gap.

The command is intentionally read-only.  It uses Tushare only to fill the
missing ``turnover_rate`` evidence for dates where the mature compact universe
has a source gap, reconstructs the affected GP expressions in memory, and
checks finite source overlaps before any future repair overlay is considered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from research_v2.db.session import V2WriteSession, YesLabReadOnlySession
from research_v2.factors.repair import apply_repair_overlay
from research_v2.factors.turnover_repair import (
    TURNOVER_GAP_FORMULAS,
    compute_turnover_gap_candidate,
)

LOCKED_TEST_START = date(2026, 1, 1)
DEFAULT_FACTOR_IDS = (
    "yeslab_377",
    "yeslab_380",
    "yeslab_382",
    "yeslab_385",
    "yeslab_387",
    "yeslab_388",
    "yeslab_395",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-id", action="append")
    parser.add_argument("--gap-date", type=date.fromisoformat, default=date(2020, 9, 18))
    parser.add_argument("--impact-start", type=date.fromisoformat, default=date(2020, 9, 14))
    parser.add_argument("--impact-end", type=date.fromisoformat, default=date(2020, 9, 25))
    parser.add_argument("--warmup-days", type=int, default=200)
    parser.add_argument(
        "--candidate-output-dir",
        type=Path,
        help="optionally write finite fill rows for candidate-ready factors",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.factor_id = tuple(args.factor_id or DEFAULT_FACTOR_IDS)
    if args.impact_start > args.impact_end:
        raise ValueError("impact start date must not be after impact end date")
    if args.gap_date < args.impact_start or args.gap_date > args.impact_end:
        raise ValueError("gap date must be inside the impact range")
    if args.impact_end >= LOCKED_TEST_START:
        raise ValueError("turnover-gap audit cannot read 2026 locked-test data")
    if args.warmup_days < 60:
        raise ValueError("warmup-days must cover the longest 60-row formula")
    if not args.factor_id or len(set(args.factor_id)) != len(args.factor_id):
        raise ValueError("factor IDs must be non-empty and unique")
    return args


def run_audit(
    args: argparse.Namespace,
    *,
    source_session: YesLabReadOnlySession,
    target_session: V2WriteSession,
    tushare_client: object,
) -> dict[str, object]:
    metadata = _query_metadata(source_session, tuple(args.factor_id))
    names_by_id = {
        f"yeslab_{int(str(row.factor_id))}": str(row.factor_name)
        for row in metadata.itertuples(index=False)
    }
    missing_ids = sorted(set(args.factor_id).difference(names_by_id))
    if missing_ids:
        raise ValueError(f"factor metadata is missing requested IDs: {missing_ids}")
    factor_names = tuple(names_by_id[factor_id] for factor_id in args.factor_id)
    warmup_start = args.gap_date - timedelta(days=args.warmup_days)
    market = _query_market(source_session, warmup_start, args.impact_end)
    mature = _query_mature(target_session, args.impact_start, args.impact_end)
    market, turnover_evidence = _apply_tushare_turnover_gap(
        market,
        mature,
        gap_date=args.gap_date,
        tushare_client=tushare_client,
    )
    factors = _query_factor_values(
        source_session,
        metadata,
        args.impact_start,
        args.impact_end,
        factor_names,
    )
    rows: list[dict[str, object]] = []
    candidate_artifacts: list[dict[str, object]] = []
    for factor_id in args.factor_id:
        factor_name = names_by_id[factor_id]
        candidate = compute_turnover_gap_candidate(factor_name, market)
        source = factors.loc[factors["factor_name"] == factor_name].copy()
        audit_row = _audit_factor(
            factor_id=factor_id,
            factor_name=factor_name,
            source=source,
            candidate=candidate,
            mature=mature,
            gap_date=args.gap_date,
        )
        rows.append(audit_row)
        if args.candidate_output_dir is not None and audit_row["status"] == (
            "TURNOVER_REPAIR_CANDIDATE_READY"
        ):
            fill_frame = _candidate_fill_frame(
                factor_id=factor_id,
                factor_name=factor_name,
                source=source,
                candidate=candidate,
                mature=mature,
            )
            candidate_artifacts.append(
                _write_candidate_artifact(
                    args.candidate_output_dir,
                    factor_id=factor_id,
                    frame=fill_frame,
                )
            )
    return {
        "scope": "TURNOVER_GAP_TUSHARE_REPAIR_AUDIT_V1",
        "gap_date": args.gap_date.isoformat(),
        "impact_start": args.impact_start.isoformat(),
        "impact_end": args.impact_end.isoformat(),
        "warmup_start": warmup_start.isoformat(),
        "factor_ids": list(args.factor_id),
        "turnover_evidence": turnover_evidence,
        "candidate_artifacts": candidate_artifacts,
        "database_writes": False,
        "rows": rows,
    }


def _query_metadata(source: YesLabReadOnlySession, factor_ids: tuple[str, ...]) -> pd.DataFrame:
    numeric = ",".join(str(int(item.removeprefix("yeslab_"))) for item in factor_ids)
    result = source.query(
        "select factor_id,factor_name,store_factor from "
        "loadTable('dfs://YesLab','factor_metadata') "
        f"where factor_id in [{numeric}]"
    )
    if not isinstance(result, pd.DataFrame):
        raise RuntimeError("factor metadata query did not return a DataFrame")
    return result


def _query_market(source: YesLabReadOnlySession, start: date, end: date) -> pd.DataFrame:
    result = source.query(
        "select trade_date,ts_code,open,close,vol,amount,pct_chg,turnover_rate "
        "from loadTable('dfs://YesLab','stock_factor_daily') "
        f"where trade_date between date('{start:%Y.%m.%d}') "
        f"and date('{end:%Y.%m.%d}')"
    )
    if not isinstance(result, pd.DataFrame):
        raise RuntimeError("market query did not return a DataFrame")
    required = {
        "trade_date",
        "ts_code",
        "open",
        "close",
        "vol",
        "amount",
        "pct_chg",
        "turnover_rate",
    }
    if not required.issubset(result.columns):
        raise RuntimeError(f"market query is missing columns: {sorted(required)}")
    frame = result.loc[:, sorted(required)].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.date
    frame["ts_code"] = frame["ts_code"].astype(str)
    for column in sorted(required.difference({"trade_date", "ts_code"})):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.drop_duplicates().reset_index(drop=True)
    if frame.duplicated(["trade_date", "ts_code"], keep=False).any():
        raise RuntimeError("market query contains conflicting business keys")
    return frame


def _query_mature(target: V2WriteSession, start: date, end: date) -> pd.DataFrame:
    universe = target.query(
        "select signal_date,ts_code,is_member from "
        "loadTable('dfs://QuantResearchV2Test','universe_daily') "
        "where universe_id=`cn_liquid_a` and universe_version=1 "
        f"and signal_date between date('{start:%Y.%m.%d}') "
        f"and date('{end:%Y.%m.%d}')"
    )
    labels = target.query(
        "select signal_date,ts_code,label_valid from "
        "loadTable('dfs://QuantResearchV2Test','label_daily') "
        "where label_id=`forward_open_5d` and label_version=1 "
        f"and signal_date between date('{start:%Y.%m.%d}') "
        f"and date('{end:%Y.%m.%d}')"
    )
    if not isinstance(universe, pd.DataFrame) or not isinstance(labels, pd.DataFrame):
        raise RuntimeError("compact input query did not return DataFrames")
    universe_keys = universe.loc[
        universe["is_member"].fillna(False), ["signal_date", "ts_code"]
    ].copy()
    label_keys = labels.loc[labels["label_valid"].fillna(False), ["signal_date", "ts_code"]].copy()
    mature = universe_keys.merge(label_keys, on=["signal_date", "ts_code"], how="inner")
    mature["signal_date"] = pd.to_datetime(mature["signal_date"], errors="raise").dt.date
    mature["ts_code"] = mature["ts_code"].astype(str)
    return mature.drop_duplicates(["signal_date", "ts_code"])


def _apply_tushare_turnover_gap(
    market: pd.DataFrame,
    mature: pd.DataFrame,
    *,
    gap_date: date,
    tushare_client: object,
) -> tuple[pd.DataFrame, dict[str, object]]:
    keys = mature.loc[mature["signal_date"] == gap_date, ["signal_date", "ts_code"]]
    observed = keys.merge(
        market,
        left_on=["signal_date", "ts_code"],
        right_on=["trade_date", "ts_code"],
        how="left",
    )
    missing_mask = ~np.isfinite(pd.to_numeric(observed["turnover_rate"], errors="coerce"))
    missing_symbols = tuple(sorted(observed.loc[missing_mask, "ts_code"].astype(str)))
    if not missing_symbols:
        return market, {
            "gap_date": gap_date.isoformat(),
            "source_missing_mature_rows": 0,
            "tushare_rows": 0,
            "tushare_finite_rows": 0,
            "replacement_rows": 0,
            "replacement_symbol_count": 0,
            "replacement_symbol_sample": [],
            "replacement_symbol_hash": "",
        }
    if not hasattr(tushare_client, "daily_basic"):
        raise TypeError("tushare client must provide daily_basic")
    basic = tushare_client.daily_basic(
        trade_date=gap_date.strftime("%Y%m%d"),
        fields="ts_code,turnover_rate",
    )
    if not isinstance(basic, pd.DataFrame) or "ts_code" not in basic.columns:
        raise RuntimeError("Tushare daily_basic did not return the expected columns")
    basic = basic.loc[:, ["ts_code", "turnover_rate"]].copy()
    basic["ts_code"] = basic["ts_code"].astype(str)
    basic["turnover_rate"] = pd.to_numeric(basic["turnover_rate"], errors="coerce")
    lookup = basic.drop_duplicates("ts_code").set_index("ts_code")["turnover_rate"]
    replacement_mask = (
        (market["trade_date"] == gap_date)
        & ~np.isfinite(market["turnover_rate"])
        & market["ts_code"].isin(missing_symbols)
    )
    replacements = market.loc[replacement_mask, "ts_code"].map(lookup)
    finite_replacements = np.isfinite(replacements.to_numpy(dtype=float))
    replacement_index = replacements.index[finite_replacements]
    market = market.copy(deep=True)
    market.loc[replacement_index, "turnover_rate"] = replacements.loc[replacement_index]
    replacement_symbols = sorted(set(market.loc[replacement_index, "ts_code"].astype(str)))
    symbol_hash = hashlib.sha256("|".join(replacement_symbols).encode()).hexdigest()
    return market, {
        "gap_date": gap_date.isoformat(),
        "source_missing_mature_rows": len(missing_symbols),
        "tushare_rows": len(basic),
        "tushare_finite_rows": int(np.isfinite(basic["turnover_rate"]).sum()),
        "replacement_rows": len(replacement_index),
        "replacement_symbol_count": len(replacement_symbols),
        "replacement_symbol_sample": replacement_symbols[:10],
        "replacement_symbol_hash": symbol_hash,
    }


def _query_factor_values(
    source: YesLabReadOnlySession,
    metadata: pd.DataFrame,
    start: date,
    end: date,
    factor_names: tuple[str, ...],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for table, group in metadata.groupby("store_factor", sort=True):
        names = ",".join("'" + str(name).replace("'", "''") + "'" for name in group.factor_name)
        result = source.query(
            "select trade_date,ts_code,factor_name,value from "
            f"loadTable('dfs://YesLab','{table}') "
            f"where factor_name in [{names}] "
            f"and trade_date between date('{start:%Y.%m.%d}') "
            f"and date('{end:%Y.%m.%d}')"
        )
        if not isinstance(result, pd.DataFrame):
            raise RuntimeError(f"factor query did not return a DataFrame: {table}")
        frames.append(result.loc[:, ["trade_date", "ts_code", "factor_name", "value"]])
    if not frames:
        return pd.DataFrame(columns=["trade_date", "ts_code", "factor_name", "value"])
    result = pd.concat(frames, ignore_index=True)
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="raise").dt.date
    result["ts_code"] = result["ts_code"].astype(str)
    result["factor_name"] = result["factor_name"].astype(str)
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    return result.loc[result["factor_name"].isin(factor_names)].copy()


def _audit_factor(
    *,
    factor_id: str,
    factor_name: str,
    source: pd.DataFrame,
    candidate: pd.DataFrame,
    mature: pd.DataFrame,
    gap_date: date,
) -> dict[str, object]:
    source = source.drop_duplicates(["trade_date", "ts_code"], keep="last")
    candidate = candidate.loc[candidate["trade_date"].isin(set(source["trade_date"]))].copy()
    merged = source.merge(candidate, on=["trade_date", "ts_code"], how="outer")
    source_values = pd.to_numeric(merged["value"], errors="coerce").to_numpy(float)
    candidate_values = pd.to_numeric(merged["candidate_value"], errors="coerce").to_numpy(float)
    overlap = np.isfinite(source_values) & np.isfinite(candidate_values)
    abs_diff = np.abs(source_values[overlap] - candidate_values[overlap])
    rel_diff = abs_diff / np.maximum(np.abs(source_values[overlap]), 1e-12)
    mismatch = (abs_diff > 1e-8) & (rel_diff > 1e-6)
    mature_window = mature.merge(
        source[["trade_date", "ts_code", "value"]],
        left_on=["signal_date", "ts_code"],
        right_on=["trade_date", "ts_code"],
        how="left",
    ).merge(candidate, on=["trade_date", "ts_code"], how="left")
    mature_source = pd.to_numeric(mature_window["value"], errors="coerce").to_numpy(float)
    mature_candidate = pd.to_numeric(
        mature_window["candidate_value"], errors="coerce"
    ).to_numpy(float)
    fillable = ~np.isfinite(mature_source) & np.isfinite(mature_candidate)
    gap_mask = mature_window["signal_date"].eq(gap_date).to_numpy()
    finite_candidate = np.isfinite(
        pd.to_numeric(candidate["candidate_value"], errors="coerce")
    )
    overlay = apply_repair_overlay(
        source,
        candidate.loc[finite_candidate].copy(),
        factor_name=factor_name,
    )
    quarantine_reasons = Counter(str(value) for value in overlay.quarantine["reason"])
    status = (
        "TURNOVER_REPAIR_CANDIDATE_READY"
        if not mismatch.any()
        else "TURNOVER_REPAIR_BLOCKED_OVERLAP"
    )
    return {
        "factor_id": factor_id,
        "factor_name": factor_name,
        "formula": TURNOVER_GAP_FORMULAS[factor_name],
        "source_rows": len(source),
        "source_finite_rows": int(np.isfinite(source_values).sum()),
        "candidate_finite_rows": int(np.isfinite(candidate_values).sum()),
        "overlap_finite_rows": int(overlap.sum()),
        "overlap_mismatch_rows": int(mismatch.sum()),
        "overlap_max_abs_diff": float(abs_diff[mismatch].max()) if mismatch.any() else 0.0,
        "overlap_max_rel_diff": float(rel_diff[mismatch].max()) if mismatch.any() else 0.0,
        "compact_rows": len(mature_window),
        "compact_source_finite_rows": int(np.isfinite(mature_source).sum()),
        "compact_candidate_finite_rows": int(np.isfinite(mature_candidate).sum()),
        "compact_fillable_rows": int(fillable.sum()),
        "compact_fillable_gap_date_rows": int((fillable & gap_mask).sum()),
        "overlay_status": overlay.status.value,
        "overlay_clean_rows": len(overlay.clean),
        "overlay_quarantine_rows": len(overlay.quarantine),
        "overlay_filled_rows": overlay.filled_rows,
        "overlay_unresolved_rows": overlay.unresolved_rows,
        "overlay_quarantine_reasons": dict(sorted(quarantine_reasons.items())),
        "status": status,
    }


def _candidate_fill_frame(
    *,
    factor_id: str,
    factor_name: str,
    source: pd.DataFrame,
    candidate: pd.DataFrame,
    mature: pd.DataFrame,
) -> pd.DataFrame:
    source_values = source.loc[:, ["trade_date", "ts_code", "value"]].copy()
    source_values = source_values.drop_duplicates(["trade_date", "ts_code"], keep="last")
    candidate_values = candidate.loc[:, ["trade_date", "ts_code", "candidate_value"]].copy()
    candidate_values = candidate_values.loc[
        np.isfinite(pd.to_numeric(candidate_values["candidate_value"], errors="coerce"))
    ]
    merged = mature.merge(
        source_values,
        left_on=["signal_date", "ts_code"],
        right_on=["trade_date", "ts_code"],
        how="left",
    ).merge(candidate_values, on=["trade_date", "ts_code"], how="left")
    source_numeric = pd.to_numeric(merged["value"], errors="coerce")
    candidate_numeric = pd.to_numeric(merged["candidate_value"], errors="coerce")
    fillable = ~np.isfinite(source_numeric.to_numpy(float)) & np.isfinite(
        candidate_numeric.to_numpy(float)
    )
    frame = pd.DataFrame(
        {
            "observation_date": merged.loc[fillable, "signal_date"].to_numpy(),
            "ts_code": merged.loc[fillable, "ts_code"].astype(str).to_numpy(),
            "factor_id": factor_id,
            "factor_version": 1,
            "factor_name": factor_name,
            "candidate_value": candidate_numeric.loc[fillable].to_numpy(dtype=float),
            "source_value": source_numeric.loc[fillable].to_numpy(dtype=float),
            "repair_method": "TUSHARE_TURNOVER_GAP",
            "reason": "TURNOVER_RATE_GAP",
        }
    )
    return frame.sort_values(
        ["observation_date", "ts_code"], kind="mergesort"
    ).reset_index(drop=True)


def _write_candidate_artifact(
    output_dir: Path,
    *,
    factor_id: str,
    frame: pd.DataFrame,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{factor_id}_turnover_gap_candidate.csv"
    frame.to_csv(path, index=False, float_format="%.17g")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "factor_id": factor_id,
        "path": str(path),
        "rows": len(frame),
        "sha256": digest,
    }


def _connect() -> tuple[YesLabReadOnlySession, V2WriteSession, object]:
    import dolphindb as ddb  # type: ignore[import-untyped]
    import tushare as ts  # type: ignore[import-untyped]

    client = ddb.session()
    client.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    token = os.getenv("QRV2_TUSHARE_TOKEN") or os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("Tushare token is not configured")
    return YesLabReadOnlySession(client), V2WriteSession(
        client, target_database="dfs://QuantResearchV2Test"
    ), ts.pro_api(token)


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    source, target, tushare_client = _connect()
    payload = run_audit(
        args,
        source_session=source,
        target_session=target,
        tushare_client=tushare_client,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


if __name__ == "__main__":
    main()
