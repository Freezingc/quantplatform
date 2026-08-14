"""Audit Tushare SW2021 level-1 point-in-time industry membership.

The command is read-only.  It compares historical Tushare membership intervals
with a YesLab security universe and the current static ``stock_info_table``
labels; it does not use static labels as a fallback for missing PIT members.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

import pandas as pd

LOCKED_TEST_START = date(2026, 1, 1)
SHSZ_SUFFIXES = (".SH", ".SZ")
CLASSIFICATION_COLUMNS = ("index_code", "industry_name", "level")
MEMBER_COLUMNS = ("index_code", "con_code", "in_date", "out_date")


def parse_date(value: str) -> date:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("date must use YYYY-MM-DD") from exc
    if parsed >= LOCKED_TEST_START:
        raise ValueError("2026 locked-test data is not allowed")
    return parsed


def _require_columns(frame: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def normalize_classifications(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, CLASSIFICATION_COLUMNS, "classification frame")
    result = frame.loc[:, list(CLASSIFICATION_COLUMNS)].copy()
    result["index_code"] = result["index_code"].astype(str)
    result["industry_name"] = result["industry_name"].astype(str)
    result["level"] = result["level"].astype(str)
    result = result.loc[result["level"].eq("L1")]
    result = result.loc[result["industry_name"].str.len().gt(0)]
    return result.drop_duplicates("index_code", keep="last").reset_index(drop=True)


def normalize_members(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, MEMBER_COLUMNS, "membership frame")
    result = frame.loc[:, list(MEMBER_COLUMNS)].copy()
    result["index_code"] = result["index_code"].astype(str)
    result["con_code"] = result["con_code"].astype(str)
    for column in ("in_date", "out_date"):
        values = result[column]
        if pd.api.types.is_datetime64_any_dtype(values):
            result[column] = pd.to_datetime(values, errors="coerce")
        else:
            result[column] = pd.to_datetime(
                values.astype("string"), format="%Y%m%d", errors="coerce"
            )
    result = result.dropna(subset=["in_date", "con_code", "index_code"])
    result = result.loc[
        result["con_code"].str.endswith(SHSZ_SUFFIXES)
    ].copy()
    return result.reset_index(drop=True)


def build_pit_snapshot(
    members: pd.DataFrame,
    classifications: pd.DataFrame,
    *,
    as_of: date,
    codes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build one PIT snapshot; ``out_date`` is an exclusive boundary."""

    if as_of >= LOCKED_TEST_START:
        raise ValueError("2026 locked-test data is not allowed")
    classification_frame = normalize_classifications(classifications)
    member_frame = normalize_members(members)
    active = member_frame.loc[
        (member_frame["in_date"] <= pd.Timestamp(as_of))
        & (
            member_frame["out_date"].isna()
            | (pd.Timestamp(as_of) < member_frame["out_date"])
        )
    ].merge(classification_frame, on="index_code", how="inner")
    if codes is not None:
        code_set = set(map(str, codes))
        active = active.loc[active["con_code"].isin(code_set)]

    conflict_counts = active.groupby("con_code")["index_code"].nunique()
    conflict_codes = int((conflict_counts > 1).sum())
    active = active.sort_values(
        ["con_code", "in_date", "index_code"], kind="stable"
    ).drop_duplicates("con_code", keep="last")
    snapshot = active.loc[
        :, ["con_code", "index_code", "industry_name", "in_date", "out_date"]
    ].rename(columns={"con_code": "ts_code", "index_code": "industry_code"})
    snapshot = snapshot.sort_values("ts_code").reset_index(drop=True)
    snapshot.attrs["overlap_conflict_code_count"] = conflict_codes
    snapshot.attrs["as_of"] = pd.Timestamp(as_of).date().isoformat()
    return snapshot


def summarize_snapshot(
    snapshot: pd.DataFrame,
    universe: pd.DataFrame,
    static_labels: pd.DataFrame | None = None,
) -> dict[str, object]:
    _require_columns(snapshot, ("ts_code", "industry_name"), "PIT snapshot")
    _require_columns(universe, ("ts_code",), "universe frame")
    universe_codes = {
        str(code)
        for code in universe["ts_code"].dropna()
        if str(code).endswith(SHSZ_SUFFIXES)
    }
    pit_codes = {
        str(code)
        for code in snapshot["ts_code"].dropna()
        if str(code).endswith(SHSZ_SUFFIXES)
    }
    covered_codes = universe_codes.intersection(pit_codes)
    summary: dict[str, object] = {
        "universe_shsz_count": len(universe_codes),
        "pit_shsz_count": len(pit_codes),
        "pit_shsz_coverage_count": len(covered_codes),
        "pit_shsz_coverage_ratio": (
            len(covered_codes) / len(universe_codes) if universe_codes else 0.0
        ),
        "pit_unmapped_universe_count": len(universe_codes.difference(pit_codes)),
        "overlap_conflict_code_count": int(
            snapshot.attrs.get("overlap_conflict_code_count", 0)
        ),
    }
    if static_labels is None:
        return summary
    _require_columns(static_labels, ("ts_code", "industry"), "static label frame")
    static = static_labels.loc[:, ["ts_code", "industry"]].copy()
    static["ts_code"] = static["ts_code"].astype(str)
    static = static.drop_duplicates("ts_code", keep="last")
    comparable = snapshot.merge(static, on="ts_code", how="inner")
    comparable = comparable.loc[comparable["ts_code"].isin(universe_codes)]
    summary["static_label_overlap_count"] = int(len(comparable))
    summary["static_label_exact_name_count"] = int(
        comparable["industry"].eq(comparable["industry_name"]).sum()
    )
    summary["static_label_exact_name_share"] = (
        float(comparable["industry"].eq(comparable["industry_name"]).mean())
        if len(comparable)
        else 0.0
    )
    return summary


def _read_or_fetch(
    path: Path | None,
    fetch: object,
) -> pd.DataFrame:
    if path is not None and path.is_file():
        return pd.read_json(path, orient="table")
    result = fetch()
    if result is None:
        result = pd.DataFrame()
    if not isinstance(result, pd.DataFrame):
        raise TypeError("Tushare response is not a DataFrame")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.to_json(orient="table", date_format="iso"), encoding="utf-8")
    return result


def fetch_sw2021_l1_members(
    pro: object, *, cache_dir: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    classification_path = (
        cache_dir / "pit_industry_sw2021_l1_classification.json"
        if cache_dir is not None
        else None
    )
    classifications = _read_or_fetch(
        classification_path,
        lambda: pro.index_classify(
            src="SW2021",
            level="L1",
            fields="index_code,industry_name,level,parent_code,src",
        ),
    )
    normalized_classifications = normalize_classifications(classifications)
    member_frames: list[pd.DataFrame] = []
    for index_code in normalized_classifications["index_code"]:
        member_path = (
            cache_dir / f"pit_industry_members_{index_code.replace('.', '_')}.json"
            if cache_dir is not None
            else None
        )
        member_frames.append(
            _read_or_fetch(
                member_path,
                lambda index_code=index_code: pro.index_member(
                    index_code=index_code,
                    fields="index_code,con_code,in_date,out_date,is_new",
                ),
            )
        )
    members = (
        pd.concat(member_frames, ignore_index=True)
        if member_frames
        else pd.DataFrame(columns=list(MEMBER_COLUMNS))
    )
    return normalized_classifications, normalize_members(members)


def run_audit(
    *,
    pro: object,
    universe: pd.DataFrame,
    static_labels: pd.DataFrame,
    as_of_dates: Sequence[date],
    cache_dir: Path | None = None,
) -> dict[str, object]:
    classifications, members = fetch_sw2021_l1_members(pro, cache_dir=cache_dir)
    universe_codes = universe["ts_code"].astype(str).dropna().unique()
    snapshots: dict[str, object] = {}
    for as_of in as_of_dates:
        snapshot = build_pit_snapshot(
            members, classifications, as_of=as_of, codes=universe_codes
        )
        snapshots[as_of.isoformat()] = summarize_snapshot(
            snapshot, universe, static_labels
        )
    return {
        "scope": "TUSHARE_SW2021_L1_PIT_INDUSTRY_AUDIT_V1",
        "industry_source": "Tushare.index_classify(SW2021,L1)+index_member",
        "market_scope": "SH/SZ",
        "excluded_exchanges": ["BJ"],
        "classification_count": int(len(classifications)),
        "membership_row_count": int(len(members)),
        "as_of_dates": [value.isoformat() for value in as_of_dates],
        "snapshots": snapshots,
        "database_writes": False,
        "locked_test_read": False,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", action="append", type=parse_date, required=True)
    parser.add_argument("--universe-start", type=parse_date, required=True)
    parser.add_argument("--universe-end", type=parse_date, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.universe_start > args.universe_end:
        parser.error("--universe-start must be on or before --universe-end")
    args.as_of = tuple(sorted(set(args.as_of)))
    return args


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parse_args(argv)
    import dolphindb as ddb  # type: ignore[import-untyped]
    import tushare as ts  # type: ignore[import-untyped]

    token = os.getenv("QRV2_TUSHARE_TOKEN") or os.getenv("TUSHARE_TOKEN", "")
    if not token:
        raise RuntimeError("Tushare token is required")
    session = ddb.session()
    session.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    universe = session.run(
        "select distinct ts_code from loadTable('dfs://YesLab','stock_factor_daily') "
        f"where trade_date between date('{args.universe_start:%Y.%m.%d}') "
        f"and date('{args.universe_end:%Y.%m.%d}')"
    )
    static_labels = session.run(
        "select ts_code,industry from loadTable('dfs://YesLab','stock_info_table')"
    )
    payload = run_audit(
        pro=ts.pro_api(token),
        universe=universe,
        static_labels=static_labels,
        as_of_dates=args.as_of,
        cache_dir=args.cache_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


if __name__ == "__main__":
    main()
