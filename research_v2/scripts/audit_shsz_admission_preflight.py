"""Run a read-only admission preflight for the SH/SZ candidate factor set.

The preflight deliberately stops before publication.  It verifies that every
candidate pair has an observed TRAIN similarity edge and that the proposal's
predictive gate is already satisfied.  Missing evidence is a hard failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any


def pair_key(left: str, right: str) -> tuple[str, str]:
    values = tuple(sorted((str(left), str(right))))
    if values[0] == values[1]:
        raise ValueError("a similarity pair must contain two distinct factors")
    return values


def required_pairs(factor_ids: list[str] | tuple[str, ...]) -> list[tuple[str, str]]:
    return [pair_key(left, right) for left, right in combinations(sorted(set(factor_ids)), 2)]


def _finite_metric(value: Any) -> float:
    metric = float(value)
    if not math.isfinite(metric):
        raise ValueError(f"similarity metric is not finite: {value!r}")
    return metric


def _add_evidence(
    evidence: dict[tuple[str, str], dict[str, Any]],
    left: str,
    right: str,
    metric: Any,
    source: str,
) -> None:
    key = pair_key(left, right)
    row = {"metric": _finite_metric(metric), "source": source}
    previous = evidence.get(key)
    if previous is not None and not math.isclose(
        float(previous["metric"]), float(row["metric"]), rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            f"conflicting similarity evidence for {key}: "
            f"{previous['metric']} from {previous['source']} vs {row['metric']} from {source}"
        )
    evidence[key] = previous or row


def _json_pairs(data: dict[str, Any], source: str) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    for row in data.get("pairs", []):
        rows.append(
            (
                str(row["factor_id_left"]),
                str(row["factor_id_right"]),
                _finite_metric(row["median_abs_spearman"]),
            )
        )
    if "median_abs_spearman" in data and data.get("rematerialized_factor_ids"):
        rematerialized = [str(value) for value in data["rematerialized_factor_ids"]]
        stored = [str(value) for value in data.get("stored_tushare_canonical_factor_ids", [])]
        for left in rematerialized:
            for right in stored:
                rows.append((left, right, _finite_metric(data["median_abs_spearman"])))
    return rows


def collect_similarity_evidence(
    paths: list[Path] | tuple[Path, ...],
    *,
    csv_focus_factor: str | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Read JSON pair artifacts and optional CSV pair matrices.

    ``csv_focus_factor`` is used for a legacy full matrix when a newer JSON
    artifact is authoritative for all other pairs.  It keeps duplicate,
    differently materialized edges from being silently mixed.
    """

    evidence: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        source = str(path)
        if path.suffix.lower() == ".csv":
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    left = str(row["factor_id_left"])
                    right = str(row["factor_id_right"])
                    if csv_focus_factor is not None and csv_focus_factor not in {left, right}:
                        continue
                    metric_column = (
                        "median_abs_spearman"
                        if row.get("median_abs_spearman") not in (None, "")
                        else "median_daily_abs_spearman"
                    )
                    _add_evidence(evidence, left, right, row[metric_column], source)
            continue
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"similarity artifact must be a JSON object: {path}")
        for left, right, metric in _json_pairs(data, source):
            _add_evidence(evidence, left, right, metric, source)
    return evidence


def load_quality_summaries(paths: list[Path] | tuple[Path, ...]) -> dict[str, dict[str, Any]]:
    """Normalize the three quality artifact layouts used by this project."""

    summaries: dict[str, dict[str, Any]] = {}
    for path in paths:
        data = json.loads(path.read_text())
        for summary in data.get("summaries", []):
            summaries[str(summary["factor_id"])] = summary
        if data.get("summary"):
            summary = dict(data["summary"])
            summaries[str(summary["factor_id"])] = summary
        metrics = data.get("train", {}).get("metrics", {})
        annual = data.get("annual_icir_abs_pass_years", {})
        for factor_id, train_metrics in metrics.items():
            summaries[str(factor_id)] = {
                "factor_id": str(factor_id),
                "train": train_metrics,
                "annual_abs_icir_pass_years": annual.get(factor_id),
            }
    return summaries


def _quality_status(summary: dict[str, Any]) -> str:
    explicit = summary.get("dual_gate_status")
    if explicit is not None:
        return str(explicit)
    train = summary.get("train", {})
    if isinstance(train, dict):
        try:
            if math.isfinite(float(train.get("icir", "nan"))):
                return "PASS"
        except (TypeError, ValueError):
            pass
    return "FAIL"


def build_preflight(
    proposal: dict[str, Any],
    quality_summaries: dict[str, dict[str, Any]],
    similarity_evidence: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    candidates = proposal.get("candidates", [])
    factor_ids = [str(row["factor_id"]) for row in candidates]
    threshold = float(proposal.get("similarity_threshold", 0.9))
    expected = required_pairs(factor_ids)
    missing = [key for key in expected if key not in similarity_evidence]
    pair_results: list[dict[str, Any]] = []
    pair_failures: list[list[str]] = []
    for key in expected:
        row = similarity_evidence.get(key)
        if row is None:
            pair_results.append({"factor_id_left": key[0], "factor_id_right": key[1], "status": "MISSING"})
            continue
        metric = float(row["metric"])
        status = "PASS" if metric < threshold else "FAIL"
        if status == "FAIL":
            pair_failures.append(list(key))
        pair_results.append(
            {
                "factor_id_left": key[0],
                "factor_id_right": key[1],
                "metric": metric,
                "threshold": threshold,
                "status": status,
                "source": row["source"],
            }
        )

    candidate_results: list[dict[str, Any]] = []
    candidate_failures: list[str] = []
    monitoring: list[str] = []
    for candidate in candidates:
        factor_id = str(candidate["factor_id"])
        predictive = str(candidate.get("predictive_gate", "FAIL"))
        annual = str(candidate.get("annual_stability_gate", "FAIL"))
        quality = quality_summaries.get(factor_id)
        quality_status = "SKIPPED" if not quality_summaries else (
            _quality_status(quality) if quality is not None else "MISSING"
        )
        status = "PASS" if predictive == "PASS" and annual in {"PASS", "MONITOR"} and quality_status in {"PASS", "SKIPPED"} else "FAIL"
        if status == "FAIL":
            candidate_failures.append(factor_id)
        if annual == "MONITOR":
            monitoring.append(factor_id)
        candidate_results.append(
            {
                "factor_id": factor_id,
                "predictive_gate": predictive,
                "annual_stability_gate": annual,
                "quality_status": quality_status,
                "status": status,
            }
        )

    scope_status = proposal.get("market_scope") == ["SH", "SZ"] and "BJ" not in proposal.get("market_scope", [])
    failures = []
    if not scope_status:
        failures.append("market_scope")
    if proposal.get("label") != "forward_open_5d":
        failures.append("label")
    if missing:
        failures.append("missing_similarity_pairs")
    if pair_failures:
        failures.append("similarity_threshold")
    failures.extend(f"candidate:{factor_id}" for factor_id in candidate_failures)
    if failures:
        status = "FAIL"
    elif monitoring:
        status = "PASS_WITH_MONITORING"
    else:
        status = "PASS"
    return {
        "scope": "TUSHARE_SHSZ_ADMISSION_PREFLIGHT_V1",
        "market_scope": proposal.get("market_scope"),
        "label": proposal.get("label"),
        "candidate_count": len(factor_ids),
        "candidate_ids": factor_ids,
        "required_pair_count": len(expected),
        "observed_pair_count": len([key for key in expected if key in similarity_evidence]),
        "missing_pairs": [list(key) for key in missing],
        "pair_failures": pair_failures,
        "pair_results": pair_results,
        "candidate_results": candidate_results,
        "monitoring_factor_ids": monitoring,
        "failed_checks": failures,
        "preflight_status": status,
        "publication_status": "NOT_PUBLISHED",
        "model_training_started": False,
        "database_writes": False,
        "locked_test_read": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--quality-artifact", type=Path, action="append", default=[])
    parser.add_argument("--similarity-artifact", type=Path, action="append", default=[])
    parser.add_argument("--similarity-csv", type=Path, action="append", default=[])
    parser.add_argument("--csv-focus-factor")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    proposal = json.loads(args.proposal.read_text())
    quality = load_quality_summaries(args.quality_artifact)
    evidence = collect_similarity_evidence(args.similarity_artifact)
    if args.similarity_csv:
        csv_evidence = collect_similarity_evidence(
            args.similarity_csv, csv_focus_factor=args.csv_focus_factor
        )
        for key, row in csv_evidence.items():
            _add_evidence(evidence, key[0], key[1], row["metric"], row["source"])
    result = build_preflight(proposal, quality, evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps({"preflight_status": result["preflight_status"], "output": str(args.output)}))
    return 0 if result["preflight_status"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
