"""Replay source-file calculators against stored YesLab factor observations.

This command is read-only.  It is intentionally limited to the two factors
whose repair candidates are currently blocked.  A finite overlap mismatch is
evidence that the stored factor cannot be safely reconstructed from the
current calculator and market snapshot, even when the missing input field is
available from a secondary source.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

LOCKED_TEST_START = date(2026, 1, 1)
REPO_ROOT = Path(__file__).resolve().parents[2]
CALCULATOR_ROOT = REPO_ROOT / "数据库"


@dataclass(frozen=True)
class ReplayDefinition:
    factor_id: str
    factor_name: str
    store_factor: str
    calculator_module: str
    calculator_class: str
    lookback_days: int


REPLAY_DEFINITIONS = {
    "yeslab_395": ReplayDefinition(
        factor_id="yeslab_395",
        factor_name="GP_026",
        store_factor="factor_data9",
        calculator_module="GP_Factor_026_Calculator",
        calculator_class="GP_Factor_026_Calculator",
        lookback_days=60,
    ),
    "yeslab_471": ReplayDefinition(
        factor_id="yeslab_471",
        factor_name="AttentionSpillover",
        store_factor="factor_data10",
        calculator_module="AttentionSpilloverCalculator",
        calculator_class="AttentionSpilloverCalculator",
        lookback_days=21,
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-id", action="append")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 9, 14))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2020, 9, 25))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.factor_id = tuple(args.factor_id or REPLAY_DEFINITIONS)
    unsupported = sorted(set(args.factor_id).difference(REPLAY_DEFINITIONS))
    if unsupported:
        raise ValueError(f"unsupported formula replay factor IDs: {unsupported}")
    if not args.factor_id or len(set(args.factor_id)) != len(args.factor_id):
        raise ValueError("factor IDs must be non-empty and unique")
    if args.start > args.end:
        raise ValueError("start date must not be after end date")
    if args.end >= LOCKED_TEST_START:
        raise ValueError("formula replay cannot read 2026 locked-test data")
    return args


def compare_replay_frames(source: pd.DataFrame, replay: pd.DataFrame) -> dict[str, object]:
    """Compare finite source observations with a calculator replay."""

    required_source = {"trade_date", "ts_code", "value"}
    required_replay = {"trade_date", "ts_code", "replay_value"}
    missing_source = sorted(required_source.difference(source.columns))
    missing_replay = sorted(required_replay.difference(replay.columns))
    if missing_source or missing_replay:
        raise ValueError(
            f"replay frames are missing columns: source={missing_source}, replay={missing_replay}"
        )
    left = source.loc[:, ["trade_date", "ts_code", "value"]].copy()
    right = replay.loc[:, ["trade_date", "ts_code", "replay_value"]].copy()
    for frame in (left, right):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.normalize()
        frame["ts_code"] = frame["ts_code"].astype(str)
    key_columns = ["trade_date", "ts_code"]
    if left.duplicated(key_columns).any() or right.duplicated(key_columns).any():
        raise ValueError("formula replay frames contain duplicate business keys")
    merged = left.merge(right, on=key_columns, how="inner")
    source_values = pd.to_numeric(merged["value"], errors="coerce").to_numpy(float)
    replay_values = pd.to_numeric(merged["replay_value"], errors="coerce").to_numpy(float)
    finite = np.isfinite(source_values) & np.isfinite(replay_values)
    abs_diff = np.abs(source_values - replay_values)
    rel_diff = abs_diff / np.maximum(np.abs(source_values), 1e-12)
    mismatch = finite & (abs_diff > 1e-8) & (rel_diff > 1e-6)
    mismatch_dates = merged.loc[mismatch, "trade_date"]
    reason_codes = []
    if not finite.any():
        reason_codes.append("NO_FINITE_OVERLAP")
    if mismatch.any():
        reason_codes.append("FINITE_OVERLAP_MISMATCH")
    return {
        "source_rows": len(left),
        "replay_rows": len(right),
        "overlap_rows": len(merged),
        "overlap_finite_rows": int(finite.sum()),
        "mismatch_rows": int(mismatch.sum()),
        "mismatch_date_count": int(mismatch_dates.nunique()),
        "mismatch_start": (
            mismatch_dates.min().date().isoformat() if not mismatch_dates.empty else None
        ),
        "mismatch_end": (
            mismatch_dates.max().date().isoformat() if not mismatch_dates.empty else None
        ),
        "max_abs_diff": float(abs_diff[finite].max()) if finite.any() else 0.0,
        "max_rel_diff": float(rel_diff[finite].max()) if finite.any() else 0.0,
        "decision_status": (
            "FORMULA_REPLAY_PASS"
            if finite.any() and not mismatch.any()
            else "FORMULA_REPLAY_BLOCKED"
        ),
        "reason_codes": reason_codes,
    }


def _query_source(
    client: object, definition: ReplayDefinition, start: date, end: date
) -> pd.DataFrame:
    result = client.run(
        "select trade_date,ts_code,value from "
        f"loadTable('dfs://YesLab','{definition.store_factor}') "
        f"where factor_name == '{definition.factor_name}' "
        f"and trade_date between date('{start:%Y.%m.%d}') "
        f"and date('{end:%Y.%m.%d}')"
    )
    if not isinstance(result, pd.DataFrame):
        raise RuntimeError("stored factor replay query did not return a DataFrame")
    frame = result.loc[:, ["trade_date", "ts_code", "value"]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.normalize()
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.drop_duplicates(["trade_date", "ts_code"], keep="last").reset_index(drop=True)


def _replay_factor(
    client: object, definition: ReplayDefinition, start: date, end: date
) -> pd.DataFrame:
    if str(CALCULATOR_ROOT) not in sys.path:
        sys.path.insert(0, str(CALCULATOR_ROOT))
    module = importlib.import_module(f"factor_calculators.{definition.calculator_module}")
    calculator_class = getattr(module, definition.calculator_class)
    wide = calculator_class(client, definition.lookback_days).calculate(
        start.strftime("%Y.%m.%d"), end.strftime("%Y.%m.%d")
    )
    return (
        wide.rename_axis("trade_date")
        .reset_index()
        .melt(id_vars=["trade_date"], var_name="ts_code", value_name="replay_value")
    )


def run_audit(args: argparse.Namespace, *, client: object) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for factor_id in args.factor_id:
        definition = REPLAY_DEFINITIONS[factor_id]
        comparison = compare_replay_frames(
            _query_source(client, definition, args.start, args.end),
            _replay_factor(client, definition, args.start, args.end),
        )
        rows.append(
            {
                "factor_id": factor_id,
                "factor_name": definition.factor_name,
                "calculator": f"{definition.calculator_module}.{definition.calculator_class}",
                "start": args.start.isoformat(),
                "end": args.end.isoformat(),
                "lookback_days": definition.lookback_days,
                **comparison,
            }
        )
    return {
        "scope": "FORMULA_REPLAY_CONSISTENCY_AUDIT_V1",
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "factor_ids": list(args.factor_id),
        "database_writes": False,
        "locked_test_read": False,
        "rows": rows,
    }


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


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    payload = run_audit(args, client=_connect())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


if __name__ == "__main__":
    main()
