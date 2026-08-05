"""Infer auditable historical BJ code aliases from matching market sequences.

The alias is metadata only: factor values are never sourced from YesLab.  The
source panel is used to identify a historical key, while the replay itself must
use the matched Tushare rows under the historical key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

LOCKED_TEST_START = date(2026, 1, 1)
FINGERPRINT_FIELDS = ("open", "high", "low", "close", "pct_chg", "vol")


@dataclass(frozen=True)
class AliasInference:
    mapping: dict[str, str]
    unmatched_source: tuple[str, ...]
    ambiguous_source: tuple[str, ...]
    unmatched_tushare: tuple[str, ...]


def market_fingerprint(frame: pd.DataFrame) -> str:
    required = {"trade_date", "ts_code", *FINGERPRINT_FIELDS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"market frame is missing columns: {missing}")
    data = frame.loc[:, ["trade_date", *FINGERPRINT_FIELDS]].copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="raise").dt.strftime(
        "%Y-%m-%d"
    )
    for field in FINGERPRINT_FIELDS:
        data[field] = pd.to_numeric(data[field], errors="coerce").round(3)
    return hashlib.sha256(
        data.sort_values("trade_date", kind="stable")
        .to_csv(index=False, lineterminator="\n")
        .encode("utf-8")
    ).hexdigest()


def _subset_matches(source: pd.DataFrame, tushare: pd.DataFrame) -> bool:
    left = source.loc[:, ["trade_date", *FINGERPRINT_FIELDS]].copy()
    right = tushare.loc[:, ["trade_date", *FINGERPRINT_FIELDS]].copy()
    for frame in (left, right):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise")
        for field in FINGERPRINT_FIELDS:
            frame[field] = pd.to_numeric(frame[field], errors="coerce").round(
                4 if field == "pct_chg" else 3
            )
    merged = left.merge(right, on="trade_date", how="left", suffixes=("_source", "_tushare"))
    if len(merged) != len(left) or merged["trade_date"].duplicated().any():
        return False
    for field in FINGERPRINT_FIELDS:
        source_values = merged[f"{field}_source"].to_numpy()
        tushare_values = merged[f"{field}_tushare"].to_numpy()
        if not np.array_equal(source_values, tushare_values, equal_nan=True):
            return False
    return True


def infer_aliases(source: pd.DataFrame, tushare: pd.DataFrame) -> AliasInference:
    source = source.loc[source["ts_code"].astype(str).str.endswith(".BJ")].copy()
    tushare = tushare.loc[tushare["ts_code"].astype(str).str.endswith(".BJ")].copy()
    source["ts_code"] = source["ts_code"].astype(str)
    tushare["ts_code"] = tushare["ts_code"].astype(str)
    source = source.drop_duplicates(["trade_date", "ts_code"], keep="last")
    tushare = tushare.drop_duplicates(["trade_date", "ts_code"], keep="last")
    source_fingerprints = {
        code: market_fingerprint(group)
        for code, group in source.groupby("ts_code", sort=True)
    }
    tushare_fingerprints = {
        code: market_fingerprint(group)
        for code, group in tushare.groupby("ts_code", sort=True)
    }
    by_fingerprint: dict[str, list[str]] = {}
    for code, fingerprint in tushare_fingerprints.items():
        by_fingerprint.setdefault(fingerprint, []).append(code)
    mapping: dict[str, str] = {}
    ambiguous: list[str] = []
    for source_code, fingerprint in source_fingerprints.items():
        candidates = by_fingerprint.get(fingerprint, [])
        if len(candidates) == 1:
            mapping[source_code] = candidates[0]
        elif len(candidates) > 1:
            ambiguous.append(source_code)
    remaining_source = sorted(set(source_fingerprints).difference(mapping, ambiguous))
    used_targets = set(mapping.values())
    source_groups = {code: group for code, group in source.groupby("ts_code", sort=True)}
    tushare_groups = {code: group for code, group in tushare.groupby("ts_code", sort=True)}
    for source_code in remaining_source:
        candidates = [
            target_code
            for target_code, target_group in tushare_groups.items()
            if target_code not in used_targets
            and _subset_matches(source_groups[source_code], target_group)
        ]
        if len(candidates) == 1:
            mapping[source_code] = candidates[0]
            used_targets.add(candidates[0])
        elif len(candidates) > 1:
            ambiguous.append(source_code)
    unmatched_source = tuple(sorted(set(source_fingerprints).difference(mapping, ambiguous)))
    unmatched_tushare = tuple(sorted(set(tushare_fingerprints).difference(mapping.values())))
    return AliasInference(
        mapping=dict(sorted(mapping.items())),
        unmatched_source=unmatched_source,
        ambiguous_source=tuple(sorted(ambiguous)),
        unmatched_tushare=unmatched_tushare,
    )


def load_tushare_cache(cache_dir: Path, start: date, end: date) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(cache_dir.glob("daily_*.json")):
        stem_date = path.stem.removeprefix("daily_")
        try:
            trade_date = date.fromisoformat(
                f"{stem_date[:4]}-{stem_date[4:6]}-{stem_date[6:8]}"
            )
        except ValueError:
            continue
        if start <= trade_date <= end:
            frames.append(pd.read_json(path, orient="table"))
    if not frames:
        raise RuntimeError("Tushare daily cache has no rows in the requested window")
    return pd.concat(frames, ignore_index=True)


def _connect() -> object:
    import dolphindb as ddb  # type: ignore[import-untyped]

    client = ddb.session()
    client.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    return client


def run_audit(
    *,
    client: object,
    cache_dir: Path,
    fingerprint_start: date,
    fingerprint_end: date,
) -> dict[str, object]:
    source = client.run(
        "select trade_date,ts_code,open,high,low,close,pct_chg,vol "
        "from loadTable('dfs://YesLab','stock_factor_daily') "
        f"where trade_date between date('{fingerprint_start:%Y.%m.%d}') "
        f"and date('{fingerprint_end:%Y.%m.%d}')"
    )
    if not isinstance(source, pd.DataFrame):
        raise RuntimeError("YesLab stock panel query did not return a DataFrame")
    tushare = load_tushare_cache(cache_dir, fingerprint_start, fingerprint_end)
    inference = infer_aliases(source, tushare)
    return {
        "scope": "BJ_CODE_MIGRATION_AUDIT_V1",
        "fingerprint_start": fingerprint_start.isoformat(),
        "fingerprint_end": fingerprint_end.isoformat(),
        "source": "YesLab.stock_factor_daily raw market fields for key inference only",
        "tushare": "cached Tushare.daily raw market fields",
        "mapping_is_factor_data": False,
        "source_bj_code_count": int(source["ts_code"].astype(str).loc[source["ts_code"].astype(str).str.endswith(".BJ")].nunique()),
        "tushare_bj_code_count": int(tushare["ts_code"].astype(str).loc[tushare["ts_code"].astype(str).str.endswith(".BJ")].nunique()),
        "mapping_count": len(inference.mapping),
        "ambiguous_source_count": len(inference.ambiguous_source),
        "unmatched_source_count": len(inference.unmatched_source),
        "mapping": inference.mapping,
        "ambiguous_source": list(inference.ambiguous_source),
        "unmatched_source": list(inference.unmatched_source),
        "unmatched_tushare": list(inference.unmatched_tushare),
        "database_writes": False,
        "locked_test_read": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fingerprint-start", type=date.fromisoformat, default=date(2025, 1, 2))
    parser.add_argument("--fingerprint-end", type=date.fromisoformat, default=date(2025, 1, 31))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.fingerprint_start > args.fingerprint_end:
        raise ValueError("fingerprint start must not be after end")
    if args.fingerprint_end >= LOCKED_TEST_START:
        raise ValueError("BJ migration audit cannot read 2026 locked-test data")
    return args


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    payload = run_audit(
        client=_connect(),
        cache_dir=args.cache_dir,
        fingerprint_start=args.fingerprint_start,
        fingerprint_end=args.fingerprint_end,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


if __name__ == "__main__":
    main()
