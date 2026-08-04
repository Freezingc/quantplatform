"""Build a deterministic before/after quality report for repaired factors.

Inputs are read-only audit and quality-summary frames.  The command defaults to
writing a local CSV artifact and never opens a database or performs a publish;
unit tests and callers may inject frames directly through :func:`main`.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

_DEFAULT_OUTPUT = Path(
    "research_v2/artifacts/diagnostics/repaired_factor_quality_comparison.csv"
)
_AUDIT_REQUIRED = {"factor_id"}
_AUDIT_COUNTERS = (
    "filled_rows",
    "added_rows",
    "remaining_nonfinite_rows",
    "unresolved_rows",
    "conflict_groups",
    "pit_violation_rows",
)


def build_comparison_report(
    audit_frame: pd.DataFrame,
    before_frame: pd.DataFrame,
    after_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return a sorted factor-level before/after quality comparison.

    ``audit_frame`` may contain one row per factor/month.  Counters are summed
    by factor, while quality metrics are read from the before/after summary
    frames and compared only when both sides provide a numeric value.
    ``metrics_json`` snapshots are accepted in addition to flat metric columns.
    """

    audit = _normalise_audit(audit_frame)
    before = _quality_by_factor(before_frame, "before")
    after = _quality_by_factor(after_frame, "after")
    factor_ids = tuple(sorted(set(audit).union(before).union(after)))
    if not factor_ids:
        raise ValueError("quality comparison has no factor IDs")

    rows: list[dict[str, object]] = []
    metric_names = sorted(
        {
            *(
                metric
                for metrics in before.values()
                for metric in metrics
            ),
            *(
                metric
                for metrics in after.values()
                for metric in metrics
            ),
        }
    )
    for factor_id in factor_ids:
        audit_row = audit.get(factor_id, {})
        before_metrics = before.get(factor_id, {})
        after_metrics = after.get(factor_id, {})
        filled = _as_int(audit_row.get("filled_rows", 0))
        added = _as_int(audit_row.get("added_rows", 0))
        remaining = _as_int(
            audit_row.get(
                "remaining_nonfinite_rows",
                audit_row.get("unresolved_rows", 0),
            )
        )
        raw_statuses = audit_row.get("_statuses", ())
        statuses = raw_statuses if isinstance(raw_statuses, set) else set()
        if "REPAIR_BLOCKED" in statuses:
            repair_status = "BLOCKED"
        elif remaining > 0 or "REPAIR_AUDIT_PASS_WITH_UNRESOLVED_QUARANTINE" in statuses:
            repair_status = "UNRESOLVED"
        elif filled + added > 0:
            repair_status = "REPAIRED"
        else:
            repair_status = "UNCHANGED"
        row: dict[str, object] = {
            "factor_id": factor_id,
            "before": _json_dump(before_metrics),
            "after": _json_dump(after_metrics),
            "filled_rows": filled,
            "added_rows": added,
            "remaining_nonfinite_rows": remaining,
            "repair_status": repair_status,
            "decision_status": _latest_status(statuses),
            "repair_data_version": str(audit_row.get("data_version", "")),
        }
        for counter in _AUDIT_COUNTERS:
            if counter not in row:
                row[counter] = _as_int(audit_row.get(counter, 0))
        for metric in metric_names:
            before_value = before_metrics.get(metric)
            after_value = after_metrics.get(metric)
            row[f"before_{metric}"] = before_value
            row[f"after_{metric}"] = after_value
            row[f"{metric}_delta"] = _delta(after_value, before_value)
        rows.append(row)

    fixed = [
        "factor_id",
        "before",
        "after",
        "filled_rows",
        "added_rows",
        "remaining_nonfinite_rows",
        "repair_status",
        "decision_status",
        "repair_data_version",
        *_AUDIT_COUNTERS,
    ]
    metric_columns = [
        column
        for metric in metric_names
        for column in (f"before_{metric}", f"after_{metric}", f"{metric}_delta")
    ]
    # ``fixed`` contains aliases such as filled_rows; preserve one deterministic
    # copy of each column when constructing the final frame.
    columns = list(dict.fromkeys([*fixed, *metric_columns]))
    return pd.DataFrame(rows, columns=columns).sort_values(
        "factor_id", kind="mergesort"
    ).reset_index(drop=True)


def compare_repaired_factor_quality(
    audit_frame: pd.DataFrame,
    before_frame: pd.DataFrame,
    after_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Compatibility alias for callers using a verb-oriented function name."""

    return build_comparison_report(audit_frame, before_frame, after_frame)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse file-only comparison options; no database target is accepted."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", "--audit-csv", dest="audit_path", type=Path)
    parser.add_argument("--before", "--before-csv", dest="before_path", type=Path)
    parser.add_argument("--after", "--after-csv", dest="after_path", type=Path)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--format", choices=("csv", "markdown"), default=None)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    audit_frame: pd.DataFrame | None = None,
    before_frame: pd.DataFrame | None = None,
    after_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build and write a comparison report from injected frames or CSV paths."""

    args = parse_args(argv)
    audit = _resolve_frame(audit_frame, args.audit_path, "audit")
    before = _resolve_frame(before_frame, args.before_path, "before")
    after = _resolve_frame(after_frame, args.after_path, "after")
    report = build_comparison_report(audit, before, after)
    output_format = args.format or (
        "markdown" if args.output.suffix.lower() in {".md", ".markdown"} else "csv"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "markdown":
        args.output.write_text(_markdown_table(report), encoding="utf-8")
    else:
        report.to_csv(args.output, index=False)
    print(
        f"quality comparison complete: factors={len(report)} "
        f"repaired={(report['repair_status'] == 'REPAIRED').sum()} "
        f"blocked={(report['repair_status'] == 'BLOCKED').sum()} "
        f"output={args.output}",
        flush=True,
    )
    return report


def _resolve_frame(
    injected: pd.DataFrame | None,
    path: Path | None,
    label: str,
) -> pd.DataFrame:
    if injected is not None:
        return injected.copy(deep=True)
    if path is None:
        raise ValueError(f"{label} input is required (inject a frame or pass --{label})")
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    return pd.read_csv(path)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Serialize a DataFrame without requiring the optional tabulate package."""

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(_markdown_cell(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        lines.append(
            "| " + " | ".join(_markdown_cell(value) for value in values) + " |"
        )
    return "\n".join(lines) + "\n"


def _markdown_cell(value: object) -> str:
    if _is_null(value):
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _normalise_audit(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    _require_columns(frame, _AUDIT_REQUIRED, "repair audit")
    grouped: dict[str, dict[str, object]] = {}
    for row in frame.to_dict(orient="records"):
        factor_id = _normalise_factor_id(row.get("factor_id"))
        if factor_id is None:
            continue
        target = grouped.setdefault(
            factor_id,
            {counter: 0 for counter in _AUDIT_COUNTERS} | {"_statuses": set()},
        )
        for counter in _AUDIT_COUNTERS:
            target[counter] = _as_int(target.get(counter, 0)) + _as_int(row.get(counter, 0))
        status = str(row.get("decision_status", ""))
        cast_statuses = target["_statuses"]
        if isinstance(cast_statuses, set) and status:
            cast_statuses.add(status)
        data_version = str(row.get("data_version", ""))
        if data_version:
            versions = set(str(target.get("data_version", "")).split(",")) - {""}
            versions.add(data_version)
            target["data_version"] = ",".join(sorted(versions))
    return grouped


def _quality_by_factor(frame: pd.DataFrame, label: str) -> dict[str, dict[str, object]]:
    _require_columns(frame, {"factor_id"}, f"{label} quality summary")
    grouped: dict[str, dict[str, object]] = {}
    for row in frame.to_dict(orient="records"):
        factor_id = _normalise_factor_id(row.get("factor_id"))
        if factor_id is None:
            continue
        if factor_id in grouped:
            raise ValueError(f"{label} quality summary contains duplicate factor_id: {factor_id}")
        grouped[factor_id] = _extract_metrics(cast(Mapping[str, object], row))
    return grouped


def _extract_metrics(row: Mapping[str, object]) -> dict[str, object]:
    metrics: dict[str, object] = {}
    raw = row.get("metrics_json")
    if raw is not None and not _is_null(raw):
        try:
            parsed = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = {}
        if isinstance(parsed, dict):
            flattened = _flatten_mapping(parsed)
            metrics.update(flattened)
            metrics.update(_leaf_aliases(flattened))
    for key, value in row.items():
        if key in {"factor_id", "metrics_json"} or _is_null(value):
            continue
        metrics[str(key)] = _jsonable(value)
    return metrics


def _flatten_mapping(
    value: Mapping[str, object],
    prefix: str = "",
) -> dict[str, object]:
    """Flatten nested snapshot sections into stable dotted metric names."""

    flattened: dict[str, object] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(_flatten_mapping(item, name))
        else:
            flattened[name] = _jsonable(item)
    return flattened


def _leaf_aliases(flattened: Mapping[str, object]) -> dict[str, object]:
    """Add stable underscore and unqualified aliases for nested metrics."""

    aliases = {
        path.replace(".", "_"): value for path, value in flattened.items() if "." in path
    }
    paths_by_leaf: dict[str, list[str]] = {}
    for path in flattened:
        leaf = path.rsplit(".", 1)[-1]
        paths_by_leaf.setdefault(leaf, []).append(path)
    for leaf, paths in paths_by_leaf.items():
        selected = min(paths, key=_metric_path_sort_key)
        aliases[leaf] = flattened[selected]
    return aliases


def _metric_path_sort_key(path: str) -> tuple[int, str]:
    section = path.split(".", 1)[0]
    preferred = {"predictive": 0, "train": 1, "validation": 2, "structural": 3}
    return preferred.get(section, 4), path


def _normalise_factor_id(value: object) -> str | None:
    if value is None or _is_null(value):
        return None
    text = str(value)
    if text.isdigit():
        return f"yeslab_{text}"
    return text if text.startswith("yeslab_") else text


def _latest_status(statuses: object) -> str:
    if not isinstance(statuses, set) or not statuses:
        return ""
    precedence = {
        "REPAIR_BLOCKED": 4,
        "REPAIR_AUDIT_PASS_WITH_UNRESOLVED_QUARANTINE": 3,
        "REPAIRED_PENDING_QUALITY_RECHECK": 2,
        "REPAIR_AUDIT_PASS": 1,
    }
    return max(statuses, key=lambda status: (precedence.get(status, 0), status))


def _delta(after: object, before: object) -> float | None:
    if not _is_numeric(after) or not _is_numeric(before):
        return None
    return round(float(str(after)) - float(str(before)), 12)


def _as_int(value: object) -> int:
    try:
        return int(str(value)) if not _is_null(value) else 0
    except (TypeError, ValueError):
        return 0


def _is_numeric(value: object) -> bool:
    if _is_null(value) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(str(value)))
    except (TypeError, ValueError):
        return False


def _is_null(value: object) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(cast(Any, value))
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, bool) else False


def _jsonable(value: object) -> object:
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass
    return value


def _json_dump(value: Mapping[str, object]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


if __name__ == "__main__":
    main()
