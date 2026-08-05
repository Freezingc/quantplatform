"""Audit daily RankIC cross-section sample coverage for SH/SZ factors."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

DEFAULT_FACTOR_IDS = (
    "yeslab_380",
    "yeslab_431",
    "yeslab_437",
    "yeslab_155",
    "yeslab_98",
    "yeslab_94",
    "yeslab_376",
    "yeslab_396",
    "yeslab_442",
)


def _validate_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    required = {"factor_id", "signal_date", "sample_count"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    result = frame.loc[:, ["factor_id", "signal_date", "sample_count"]].copy()
    result["factor_id"] = result["factor_id"].astype(str)
    result["signal_date"] = pd.to_datetime(result["signal_date"], errors="raise").dt.normalize()
    result["sample_count"] = pd.to_numeric(result["sample_count"], errors="coerce")
    if result["sample_count"].isna().any() or (result["sample_count"] < 0).any():
        raise ValueError(f"{label} contains invalid sample_count")
    if result.duplicated(["factor_id", "signal_date"]).any():
        raise ValueError(f"{label} contains duplicate factor/date rows")
    return result


def _period_stats(frame: pd.DataFrame, factor_id: str, prefix: str) -> dict[str, object]:
    values = frame.loc[frame["factor_id"] == factor_id, "sample_count"].to_numpy(float)
    if values.size == 0:
        return {
            f"{prefix}_valid_day_count": 0,
            f"{prefix}_min_sample_count": 0,
            f"{prefix}_median_sample_count": 0.0,
            f"{prefix}_p05_sample_count": 0.0,
            f"{prefix}_max_sample_count": 0,
            f"{prefix}_min_to_median": 0.0,
        }
    median = float(pd.Series(values).median())
    return {
        f"{prefix}_valid_day_count": int(values.size),
        f"{prefix}_min_sample_count": int(values.min()),
        f"{prefix}_median_sample_count": median,
        f"{prefix}_p05_sample_count": float(pd.Series(values).quantile(0.05)),
        f"{prefix}_max_sample_count": int(values.max()),
        f"{prefix}_min_to_median": float(values.min() / median) if median > 0 else 0.0,
    }


def audit_sample_coverage(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    factor_ids: Sequence[str],
    *,
    minimum_valid_days: int = 200,
    minimum_min_to_median: float = 0.70,
) -> dict[str, object]:
    factor_ids = tuple(str(value) for value in factor_ids)
    if not factor_ids or len(set(factor_ids)) != len(factor_ids):
        raise ValueError("factor_ids must be non-empty and unique")
    if minimum_valid_days < 1 or not math.isfinite(float(minimum_min_to_median)):
        raise ValueError("invalid coverage thresholds")
    train = _validate_frame(train, "train")
    validation = _validate_frame(validation, "validation")
    rows: list[dict[str, object]] = []
    for factor_id in sorted(factor_ids):
        row = {"factor_id": factor_id}
        row.update(_period_stats(train, factor_id, "train"))
        row.update(_period_stats(validation, factor_id, "validation"))
        row["status"] = (
            "PASS"
            if row["train_valid_day_count"] >= minimum_valid_days
            and row["validation_valid_day_count"] >= minimum_valid_days
            and row["train_min_to_median"] >= minimum_min_to_median
            and row["validation_min_to_median"] >= minimum_min_to_median
            else "FAIL"
        )
        rows.append(row)
    failed = [str(row["factor_id"]) for row in rows if row["status"] == "FAIL"]
    return {
        "status": "FAIL" if failed else "PASS",
        "factor_count": len(rows),
        "failed_factor_ids": failed,
        "minimum_valid_days": minimum_valid_days,
        "minimum_min_to_median": minimum_min_to_median,
        "rows": rows,
        "database_writes": False,
        "locked_test_read": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, action="append", required=True)
    parser.add_argument("--validation-input", type=Path, action="append", required=True)
    parser.add_argument("--factor-id", action="append")
    parser.add_argument("--minimum-valid-days", type=int, default=200)
    parser.add_argument("--minimum-min-to-median", type=float, default=0.70)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.factor_id = tuple(dict.fromkeys(args.factor_id or DEFAULT_FACTOR_IDS))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    train = pd.concat([pd.read_csv(path) for path in args.train_input], ignore_index=True)
    validation = pd.concat([pd.read_csv(path) for path in args.validation_input], ignore_index=True)
    result = audit_sample_coverage(
        train,
        validation,
        args.factor_id,
        minimum_valid_days=args.minimum_valid_days,
        minimum_min_to_median=args.minimum_min_to_median,
    )
    result.update(
        {
            "scope": "TUSHARE_SHSZ_SAMPLE_COVERAGE_AUDIT_V1",
            "market_scope": ["SH", "SZ"],
            "excluded_exchanges": ["BJ"],
            "train_inputs": [str(path) for path in args.train_input],
            "validation_inputs": [str(path) for path in args.validation_input],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
