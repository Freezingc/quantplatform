"""Build a single read-only quality scoreboard for the nine SH/SZ candidates."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

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
DEFAULT_PREFLIGHT = Path(
    "research_v2/artifacts/diagnostics/tushare_shsz_admission_preflight_v3.json"
)
DEFAULT_DRIFT = Path(
    "research_v2/artifacts/diagnostics/tushare_shsz_performance_drift_audit_v1.json"
)
DEFAULT_VALIDATION_SIMILARITY = Path(
    "research_v2/artifacts/diagnostics/tushare_shsz_validation_similarity_admission_v1.json"
)
DEFAULT_SCOPE = Path(
    "research_v2/artifacts/diagnostics/tushare_shsz_scope_transition_audit_v1.json"
)
DEFAULT_QUALITY_ARTIFACTS = (
    Path("research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_quality_16_v1.json"),
    Path("research_v2/artifacts/diagnostics/tushare_canonical_core_quality_v1.json"),
    Path("research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_quality_155_v1.json"),
    Path("research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_quality_94_98_v1.json"),
)


def _by_factor(rows: Sequence[Mapping[str, object]], label: str) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for row in rows:
        factor_id = str(row.get("factor_id", ""))
        if not factor_id:
            raise ValueError(f"{label} contains a row without factor_id")
        if factor_id in result:
            raise ValueError(f"{label} contains duplicate factor_id: {factor_id}")
        result[factor_id] = row
    return result


def build_scoreboard(
    *,
    factor_ids: Sequence[str],
    preflight: Mapping[str, object],
    drift: Mapping[str, object],
    annual_pass_years: Mapping[str, int],
    validation_similarity: Mapping[str, object],
    scope: Mapping[str, object] | None = None,
) -> dict[str, object]:
    factor_ids = tuple(str(value) for value in factor_ids)
    if not factor_ids or len(set(factor_ids)) != len(factor_ids):
        raise ValueError("factor_ids must be non-empty and unique")
    candidates = _by_factor(preflight.get("candidate_results", []), "preflight")
    drift_rows = _by_factor(drift.get("rows", []), "performance drift")
    missing = sorted(
        set(factor_ids).difference(candidates)
        | set(factor_ids).difference(drift_rows)
        | set(factor_ids).difference(annual_pass_years)
    )
    if missing:
        raise ValueError(f"scoreboard evidence is missing factors: {missing}")
    validation_pair_results = validation_similarity.get("pair_results", [])
    pair_by_factor: dict[str, list[Mapping[str, object]]] = {factor_id: [] for factor_id in factor_ids}
    for pair in validation_pair_results:
        if not isinstance(pair, Mapping):
            continue
        left = str(pair.get("factor_id_left", ""))
        right = str(pair.get("factor_id_right", ""))
        if left in pair_by_factor:
            pair_by_factor[left].append(pair)
        if right in pair_by_factor:
            pair_by_factor[right].append(pair)
    global_validation_status = str(validation_similarity.get("status", "FAIL"))
    monitoring = sorted(str(value) for value in preflight.get("monitoring_factor_ids", []))
    rows: list[dict[str, object]] = []
    for factor_id in sorted(factor_ids):
        candidate = candidates[factor_id]
        drift_row = drift_rows[factor_id]
        factor_pairs = pair_by_factor[factor_id]
        validation_status = (
            "PASS"
            if global_validation_status == "PASS"
            and (not validation_pair_results or all(pair.get("status") == "PASS" for pair in factor_pairs))
            else "FAIL"
        )
        row_status = (
            "PASS"
            if candidate.get("status") == "PASS"
            and drift_row.get("validation_predictive_status") == "PASS"
            and drift_row.get("drift_status") == "PASS"
            and validation_status == "PASS"
            else "FAIL"
        )
        rows.append(
            {
                "factor_id": factor_id,
                "predictive_gate": candidate.get("predictive_gate"),
                "annual_stability_gate": candidate.get("annual_stability_gate"),
                "annual_abs_icir_pass_years": int(annual_pass_years[factor_id]),
                "formula_scope_gate": candidate.get("formula_scope_gate"),
                "quality_status": candidate.get("quality_status"),
                "direction_consistent": drift_row.get("direction_consistent"),
                "validation_predictive_status": drift_row.get("validation_predictive_status"),
                "drift_status": drift_row.get("drift_status"),
                "validation_similarity_status": validation_status,
                "train_mean_rank_ic": drift_row.get("train_mean_rank_ic"),
                "validation_mean_rank_ic": drift_row.get("validation_mean_rank_ic"),
                "train_icir": drift_row.get("train_icir"),
                "validation_icir": drift_row.get("validation_icir"),
                "admission_status": "PASS_WITH_MONITORING" if factor_id in monitoring else row_status,
                "status": row_status,
            }
        )
    failures = [str(row["factor_id"]) for row in rows if row["status"] != "PASS"]
    status = "FAIL" if failures else "PASS_WITH_MONITORING" if monitoring else "PASS"
    result = {
        "scope": "TUSHARE_SHSZ_FACTOR_QUALITY_SCOREBOARD_V1",
        "market_scope": ["SH", "SZ"],
        "excluded_exchanges": ["BJ"],
        "factor_count": len(rows),
        "factor_ids": list(factor_ids),
        "status": status,
        "monitoring_factor_ids": monitoring,
        "failed_factor_ids": failures,
        "preflight_status": preflight.get("preflight_status"),
        "validation_similarity_status": validation_similarity.get("status"),
        "scope_audit_status": scope.get("formula_static_audit") if scope else None,
        "rows": rows,
        "publication_status": "NOT_PUBLISHED",
        "database_writes": False,
        "model_training_started": False,
        "locked_test_read": False,
    }
    return result


def _load_json(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return data


def _load_annual_pass_years(paths: Sequence[Path], factor_ids: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for path in paths:
        payload = _load_json(path)
        for annual_key in ("annual_abs_icir_pass_years", "annual_icir_abs_pass_years"):
            annual = payload.get(annual_key, {})
            if isinstance(annual, dict):
                result.update({str(key): int(value) for key, value in annual.items()})
        for summary in payload.get("summaries", []):
            if not isinstance(summary, Mapping) or "factor_id" not in summary:
                continue
            if "annual_abs_icir_pass_years" in summary:
                result[str(summary["factor_id"])] = int(summary["annual_abs_icir_pass_years"])
        summary = payload.get("summary")
        if isinstance(summary, Mapping) and "annual_abs_icir_pass_years" in summary:
            result[str(summary["factor_id"])] = int(summary["annual_abs_icir_pass_years"])
    missing = sorted(set(map(str, factor_ids)).difference(result))
    if missing:
        raise ValueError(f"annual quality artifacts are missing factors: {missing}")
    return {factor_id: result[factor_id] for factor_id in factor_ids}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-id", action="append")
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--drift", type=Path, default=DEFAULT_DRIFT)
    parser.add_argument("--validation-similarity", type=Path, default=DEFAULT_VALIDATION_SIMILARITY)
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE)
    parser.add_argument("--quality-artifact", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.factor_id = tuple(dict.fromkeys(args.factor_id or DEFAULT_FACTOR_IDS))
    args.quality_artifact = tuple(args.quality_artifact or DEFAULT_QUALITY_ARTIFACTS)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preflight = _load_json(args.preflight)
    drift = _load_json(args.drift)
    validation_similarity = _load_json(args.validation_similarity)
    scope = _load_json(args.scope)
    annual = _load_annual_pass_years(args.quality_artifact, args.factor_id)
    result = build_scoreboard(
        factor_ids=args.factor_id,
        preflight=preflight,
        drift=drift,
        annual_pass_years=annual,
        validation_similarity=validation_similarity,
        scope=scope,
    )
    result["source_artifacts"] = {
        "preflight": str(args.preflight),
        "drift": str(args.drift),
        "validation_similarity": str(args.validation_similarity),
        "scope": str(args.scope),
        "quality": [str(path) for path in args.quality_artifact],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0 if result["status"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
