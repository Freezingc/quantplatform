"""Audit complete 2025 OOS similarity evidence for an SH/SZ factor set."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from audit_shsz_admission_preflight import pair_key, required_pairs

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
DEFAULT_THRESHOLD = 0.9
DEFAULT_MINIMUM_VALID_DAYS = 200


def _metric_column(frame: pd.DataFrame) -> str:
    for column in ("median_daily_abs_spearman", "median_abs_spearman"):
        if column in frame.columns:
            return column
    raise ValueError(
        "validation similarity input must contain median_daily_abs_spearman or median_abs_spearman"
    )


def audit_validation_similarity(
    frame: pd.DataFrame,
    factor_ids: Sequence[str],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    minimum_valid_days: int = DEFAULT_MINIMUM_VALID_DAYS,
) -> dict[str, object]:
    """Check pair completeness, valid-day coverage and redundancy threshold."""

    factor_ids = tuple(str(value) for value in factor_ids)
    if not factor_ids or len(set(factor_ids)) != len(factor_ids):
        raise ValueError("factor_ids must be non-empty and unique")
    if not math.isfinite(float(threshold)) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    if minimum_valid_days < 1:
        raise ValueError("minimum_valid_days must be positive")
    required = required_pairs(list(factor_ids))
    required_set = set(required)
    metric_column = _metric_column(frame)
    required_columns = {"factor_id_left", "factor_id_right", "valid_days", metric_column}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"validation similarity input is missing columns: {missing_columns}")

    evidence: dict[tuple[str, str], dict[str, object]] = {}
    for row in frame.to_dict(orient="records"):
        key = pair_key(str(row["factor_id_left"]), str(row["factor_id_right"]))
        if key not in required_set:
            continue
        metric = float(row[metric_column])
        valid_days = int(row["valid_days"])
        if not math.isfinite(metric):
            raise ValueError(f"similarity metric is not finite for {key}")
        if valid_days < 0:
            raise ValueError(f"valid_days must be non-negative for {key}")
        previous = evidence.get(key)
        if previous is not None and (
            not math.isclose(float(previous["metric"]), metric, rel_tol=0.0, abs_tol=1e-12)
            or int(previous["valid_days"]) != valid_days
        ):
            raise ValueError(f"conflicting validation similarity evidence for {key}")
        evidence[key] = {"metric": metric, "valid_days": valid_days}

    missing = [list(key) for key in required if key not in evidence]
    pair_failures: list[list[str]] = []
    pair_results: list[dict[str, object]] = []
    for key in required:
        row = evidence.get(key)
        if row is None:
            pair_results.append(
                {
                    "factor_id_left": key[0],
                    "factor_id_right": key[1],
                    "status": "MISSING",
                }
            )
            continue
        metric = float(row["metric"])
        valid_days = int(row["valid_days"])
        status = "PASS" if metric < threshold and valid_days >= minimum_valid_days else "FAIL"
        if status == "FAIL":
            pair_failures.append(list(key))
        pair_results.append(
            {
                "factor_id_left": key[0],
                "factor_id_right": key[1],
                "metric": metric,
                "valid_days": valid_days,
                "threshold": threshold,
                "minimum_valid_days": minimum_valid_days,
                "status": status,
            }
        )
    observed = [row for row in evidence.values()]
    failures = bool(missing or pair_failures)
    return {
        "factor_ids": list(factor_ids),
        "required_pair_count": len(required),
        "observed_pair_count": len(evidence),
        "missing_pairs": missing,
        "pair_failures": pair_failures,
        "pair_results": pair_results,
        "threshold": float(threshold),
        "minimum_valid_days": int(minimum_valid_days),
        "minimum_valid_days_observed": min((int(row["valid_days"]) for row in observed), default=0),
        "maximum_valid_days_observed": max((int(row["valid_days"]) for row in observed), default=0),
        "max_median_abs_spearman": max((float(row["metric"]) for row in observed), default=float("nan")),
        "status": "FAIL" if failures else "PASS",
        "database_writes": False,
        "locked_test_read": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--factor-id", action="append")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--minimum-valid-days", type=int, default=DEFAULT_MINIMUM_VALID_DAYS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.factor_id = tuple(dict.fromkeys(args.factor_id or DEFAULT_FACTOR_IDS))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    frame = pd.read_csv(args.input)
    result = audit_validation_similarity(
        frame,
        args.factor_id,
        threshold=args.threshold,
        minimum_valid_days=args.minimum_valid_days,
    )
    result.update(
        {
            "scope": "TUSHARE_SHSZ_VALIDATION_SIMILARITY_ADMISSION_V1",
            "validation_window": "2025-01-02/2025-12-23",
            "source_artifact": str(args.input),
            "market_scope": ["SH", "SZ"],
            "excluded_exchanges": ["BJ"],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
