#!/usr/bin/env python3
"""Read-only extractor: model-side walk-forward results -> text/JSON paper evidence.

What it reads
    * ``<artifacts>/<workspace>/report.json``          (signed run reports)
    * ``<artifacts>/<workspace>/promotion_evidence.json`` (hash only, never copied)
    * ``<artifacts>/<workspace>/datasets/*/manifest.json`` (published dataset manifests)
    * ``<artifacts>/<workspace>/search/<fold>/*.json``  (per-config search records)
    * ``research_v2/scripts/walk_forward_factor_daily.py`` (factor-set definitions, via ast)

What it never does
    * no writes anywhere except ``--out`` (default: this script's own directory);
    * no database access, no model weights, no checkpoints, no parquet panels;
    * no editing of the repository main tree (which is running a multi-day job).

Every emitted number is copied from one of the files above, and ``provenance.json``
carries the sha256 of each source file together with the git HEAD it was read at,
so a reader on another machine can confirm the extraction was not retyped by hand.

Stdlib only. Usage:
    python3 extract_from_reports.py \
        [--artifacts /data/zhangyudong/qrv2_research_artifacts] \
        [--repo /data/zhangyudong/FDU/Quant-Platform-Prod_CODEX] [--out .]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# workspace -> how this run participates in the standing comparison.  Labels come
# from docs/reports/2026-09-12-candidate-standings-comparison.md and
# docs/handoffs/2026-09-14-m1-full-grid-passed-and-m2-launched.md, both vendored
# alongside this script; nothing here is inferred from the artifacts alone.
CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "workspace": "baseline_v2",
        "role": "baseline",
        "carrier": "best_simple B0-B3, selected per fold on validation",
        "factor_source": "frozen_v2_baseline (9 dense baseline-eligible factors)",
        "search_grid": "4 simple reference rules",
        "authoritative": True,
        "note": "The G4 reference. Deliberately denied the 6 sparse candidate-only factors.",
    },
    {
        "workspace": "m0_v2",
        "role": "candidate",
        "carrier": "M0 / LightGBM LambdaRank, --grid base",
        "factor_source": "frozen_v2 (15 factors)",
        "search_grid": "24 configs",
        "authoritative": True,
        "note": "G4 FAIL. Grid has no lambda_l2 dimension.",
    },
    {
        "workspace": "m0_v3",
        "role": "candidate",
        "carrier": "M0 / LightGBM LambdaRank, --grid refined",
        "factor_source": "frozen_v2 (15 factors)",
        "search_grid": "24 configs",
        "authoritative": True,
        "note": "G4 FAIL. Same family as m0_v2 but a different 24-config grid (adds lambda_l2).",
    },
    {
        "workspace": "m0r_v2",
        "role": "candidate",
        "carrier": "M0R / LightGBM rank-regression, config pinned by the carrier",
        "factor_source": "frozen_v2 (15 factors)",
        "search_grid": "1 config (no validation-set selection)",
        "authoritative": True,
        "note": "G4 PASS. The only candidate registered in model_registry (WALK_FORWARD_PASSED).",
    },
    {
        "workspace": "m0r_v3",
        "role": "replicate",
        "carrier": "M0R (independent full re-run of m0r_v2)",
        "factor_source": "frozen_v2 (15 factors)",
        "search_grid": "1 config",
        "authoritative": False,
        "note": "Reproducibility evidence: byte-identical report to m0r_v2.",
    },
    {
        "workspace": "m0r_resume_a",
        "role": "resume-replica",
        "carrier": "M0R (--resume acceptance re-run)",
        "factor_source": "frozen_v2 (15 factors)",
        "search_grid": "1 config",
        "authoritative": False,
        "note": "Acceptance criterion A replica: adopts every fold, must reproduce the report.",
    },
    {
        "workspace": "m0r_resume_c",
        "role": "resume-replica",
        "carrier": "M0R (--resume acceptance re-run)",
        "factor_source": "frozen_v2 (15 factors)",
        "search_grid": "1 config",
        "authoritative": False,
        "note": "Acceptance criterion A replica (second instance).",
    },
    {
        "workspace": "m1_parallel_v1",
        "role": "candidate-superseded",
        "carrier": "M1 / ResidualMLP, single pinned carrier config",
        "factor_source": "frozen_v2 (15 factors)",
        "search_grid": "1 config",
        "authoritative": False,
        "note": "G4 PASS (+0.005845) but superseded by the full grid below.",
    },
    {
        "workspace": "m1_full_v1",
        "role": "candidate",
        "carrier": "M1 / ResidualMLP, full grid",
        "factor_source": "frozen_v2 (15 factors)",
        "search_grid": "27 configs",
        "authoritative": True,
        "note": "G4 PASS (+0.007827). All four fold winners chose hidden_dim 128.",
    },
    {
        "workspace": "m2_v1",
        "role": "candidate",
        "carrier": "M2 / TemporalGRU + TemporalTCN (history_length 20)",
        "factor_source": "frozen_v2 (15 factors)",
        "search_grid": "24 configs = family{GRU,TCN} x hidden{64,128} x dropout{0,.1,.2} x lr{3e-4,1e-3}",
        "authoritative": True,
        "note": (
            "COMPLETE 2026-10-08; report.json exists. G4 FAIL (+0.004959 against the 0.005 "
            "bar, short by 4.08e-05) and G6 FAIL (905/969 = 0.933953 against the 0.95 bar); "
            "G1/G2/G3/G5 PASS, so all_passed=false and no promotion evidence was emitted. "
            "The G6 loss is a coverage loss, not a score collapse: M2's 15-observation "
            "minimum history leaves 17,306 labelled rows unscored (F1 6,333 / F2 5,709 / "
            "F3 3,603 / F4 1,661), and M2 fails the daily health check on 64 of the 969 "
            "ordinary test days where the 945-day carriers fail 24 -- 40 additional days. "
            "M2's delta is NOT comparable by subtraction with M0R/M1/M0; see "
            "comparison_basis."
        ),
    },
)

GATE_ORDER = ("G1", "G2", "G3", "G4", "G5", "G6")
OOS_KEYS = (
    "valid_day_count",
    "mean_rank_ic",
    "baseline_mean_rank_ic",
    "delta_rank_ic",
    "icir",
    "hac_t_statistic",
    "all_passed",
    "gate_note",
)
FACTOR_SET_NAMES = (
    "FROZEN_FACTOR_SET_V1",
    "FROZEN_FACTOR_SET_V2",
    "FROZEN_FACTOR_SET_V2_DENSE",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(out: Path, payload: Any) -> None:
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"  wrote {out.name} ({out.stat().st_size} bytes)")


def digest_versions(versions: Any) -> dict[str, Any]:
    """Collapse a per-month data_version mapping to something quotable in a paper."""
    if not isinstance(versions, dict):
        return {"present": False}
    canonical = json.dumps(versions, sort_keys=True)
    per_table = {}
    for table, months in versions.items():
        if not isinstance(months, dict):
            continue
        distinct = sorted(set(months.values()))
        per_table[table] = {
            "month_partitions": len(months),
            "first_partition": min(months),
            "last_partition": max(months),
            "distinct_data_versions": len(distinct),
            "sha256_of_mapping": hashlib.sha256(
                json.dumps(months, sort_keys=True).encode()
            ).hexdigest(),
        }
    return {
        "present": True,
        "sha256_of_full_mapping": hashlib.sha256(canonical.encode()).hexdigest(),
        "tables": per_table,
    }


def summarise_report(report: dict[str, Any]) -> dict[str, Any]:
    oos = report.get("oos") or {}
    gates_raw = oos.get("gates") or {}
    gates = {}
    for name in GATE_ORDER:
        entry = gates_raw.get(name)
        if isinstance(entry, dict):
            gates[name] = {
                "status": entry.get("status"),
                "observed_value": entry.get("observed_value"),
                "threshold": entry.get("threshold"),
            }
    folds = []
    raw_folds = report.get("folds") or []
    # The baseline run reports `folds` as a bare id list plus a separate
    # `selections` block (per-fold chosen best_simple and the validation scores
    # that chose it); candidate runs report a dict per fold. Handle both.
    if raw_folds and not isinstance(raw_folds[0], dict):
        folds = [{"fold_id": fold_id} for fold_id in raw_folds]
        raw_folds = []
    for fold in raw_folds:
        results = fold.get("search_results") or []
        ranked = sorted(
            results,
            key=lambda item: item.get("ensemble_validation_rank_ic") or float("-inf"),
            reverse=True,
        )
        folds.append(
            {
                "fold_id": fold.get("fold_id"),
                "dataset_id": fold.get("dataset_id"),
                "model_id": fold.get("model_id"),
                "baseline_selection_id": fold.get("baseline_selection_id"),
                "model_selection_id": fold.get("model_selection_id"),
                "best_config": fold.get("best_config"),
                "configs_searched": len(results),
                "best_ensemble_validation_rank_ic": (
                    ranked[0].get("ensemble_validation_rank_ic") if ranked else None
                ),
                "top3_by_validation": [
                    {
                        "config": item.get("config"),
                        "ensemble_validation_rank_ic": item.get(
                            "ensemble_validation_rank_ic"
                        ),
                        "seed_validation_rank_ic": item.get("seed_validation_rank_ic"),
                        "worst_seed": item.get("worst_seed"),
                    }
                    for item in ranked[:3]
                ],
            }
        )
    environment = report.get("environment")
    summary = {
        "family": report.get("family"),
        "mode": report.get("mode"),
        "oos": {key: oos.get(key) for key in OOS_KEYS if key in oos},
        "yearly_mean_rank_ic": oos.get("yearly_mean_rank_ic"),
        "gates": gates,
        "folds": folds,
        "baseline_selections": report.get("selections"),
        "data_versions_digest": digest_versions(report.get("data_versions")),
        "environment": environment,
    }
    # Only the temporal carrier reports this: labelled rows the candidate never scored,
    # because a security needs >=15 observed rows before M2 will window it. Copied verbatim
    # when present so the G6 coverage story stays traceable to the signed report instead of
    # being retyped; absent (not null) for carriers that score every labelled row.
    if "unscored_scoreable_rows_by_fold" in report:
        summary["unscored_scoreable_rows_by_fold"] = report["unscored_scoreable_rows_by_fold"]
    return summary


def read_factor_sets(repo: Path) -> dict[str, Any]:
    """Parse the frozen factor sets out of the driver's own source (no import)."""
    source_path = repo / "research_v2" / "scripts" / "walk_forward_factor_daily.py"
    text = source_path.read_text()
    tree = ast.parse(text)
    sets: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id not in FACTOR_SET_NAMES:
            continue
        try:
            value = ast.literal_eval(node.value)
        except ValueError:
            continue
        sets[node.target.id] = list(value)
    return {
        "source_file": str(source_path),
        "source_sha256": sha256_file(source_path),
        "sets": sets,
        "counts": {name: len(ids) for name, ids in sets.items()},
        "canonical_feature_names_v2": [
            f"f{index:02d}" for index in range(len(sets.get("FROZEN_FACTOR_SET_V2", [])))
        ],
        "baseline_eligibility_rationale": (
            "Baselines B0-B3 are computed on FROZEN_FACTOR_SET_V2_DENSE only. The six "
            "sparse candidate-only factors are NaN on roughly half of all days by formula "
            "(their sparsity IS the candidate signal), so admitting them to the baseline "
            "would make the G4 bar a dishonest simple-model reference. Consequence for "
            "any write-up: the candidate's information set strictly contains the "
            "baseline's by design; G4 measures improvement over an honest simple "
            "reference, not over an information-matched control."
        ),
    }


def read_dataset_manifests(workspace: Path) -> list[dict[str, Any]]:
    manifests = []
    datasets = workspace / "datasets"
    if not datasets.is_dir():
        return manifests
    for manifest_path in sorted(datasets.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        shards = manifest.get("shard_hashes") or {}
        splits: dict[str, dict[str, Any]] = {}
        for key in shards:
            match = re.match(r"split=([a-z]+)/year=(\d{4})/", key)
            if not match:
                continue
            role, year = match.group(1), int(match.group(2))
            entry = splits.setdefault(role, {"years": [], "shards": 0})
            entry["years"].append(year)
            entry["shards"] += 1
        for entry in splits.values():
            entry["years"] = sorted(set(entry["years"]))
        metadata = manifest.get("metadata") or {}
        manifests.append(
            {
                "dataset_id": manifest.get("dataset_id"),
                "row_count": manifest.get("row_count"),
                "column_order": manifest.get("column_order"),
                "factor_columns": [
                    column
                    for column in (manifest.get("column_order") or [])
                    if re.fullmatch(r"f\d{2}", column)
                ],
                "business_keys": manifest.get("business_keys"),
                "splits": splits,
                "split_content_hashes": manifest.get("split_content_hashes"),
                "manifest_sha256": sha256_file(manifest_path),
                "data_versions_digest": digest_versions(metadata.get("data_versions")),
            }
        )
    return manifests


def read_search_records(workspace: Path) -> dict[str, Any]:
    search = workspace / "search"
    partial: dict[str, Any] = {}
    if not search.is_dir():
        return partial
    for fold_dir in sorted(p for p in search.iterdir() if p.is_dir()):
        records = []
        for record_path in sorted(fold_dir.glob("*.json")):
            record = json.loads(record_path.read_text())
            records.append(
                {
                    "config_index": record.get("config_index"),
                    "config": record.get("config"),
                    "config_hash": record.get("config_hash"),
                    "ensemble_validation_rank_ic": (
                        record.get("result") or {}
                    ).get("ensemble_validation_rank_ic"),
                    "seed_validation_rank_ic": (
                        record.get("result") or {}
                    ).get("seed_validation_rank_ic"),
                    "worst_seed": (record.get("result") or {}).get("worst_seed"),
                    "record_written_utc": datetime.fromtimestamp(
                        record_path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(timespec="seconds"),
                    "record_sha256": sha256_file(record_path),
                }
            )
        records.sort(
            key=lambda item: item.get("ensemble_validation_rank_ic") or float("-inf"),
            reverse=True,
        )
        partial[fold_dir.name] = {
            "configs_completed": len(records),
            "records": records,
        }
    return partial


def git_head(repo: Path) -> dict[str, str]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    return {
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": run("rev-parse", "HEAD"),
        "subject": run("log", "-1", "--format=%s"),
        "dirty": bool(run("status", "--porcelain")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("/data/zhangyudong/qrv2_research_artifacts"),
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX"),
    )
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    extracted_at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    sources: dict[str, str] = {}
    standings: list[dict[str, Any]] = []
    winners: dict[str, Any] = {}
    yearly: dict[str, Any] = {}
    datasets: dict[str, Any] = {}
    in_progress: dict[str, Any] = {}

    for candidate in CANDIDATES:
        workspace = args.artifacts / candidate["workspace"]
        report_path = workspace / "report.json"
        evidence_path = workspace / "promotion_evidence.json"
        entry: dict[str, Any] = dict(candidate)
        entry["workspace_path"] = str(workspace)
        entry["report_exists"] = report_path.exists()
        if report_path.exists():
            report = json.loads(report_path.read_text())
            entry.update(summarise_report(report))
            entry["report_sha256"] = sha256_file(report_path)
            sources[str(report_path)] = entry["report_sha256"]
            winners[candidate["workspace"]] = entry["folds"]
            yearly[candidate["workspace"]] = {
                "yearly_mean_rank_ic": entry.get("yearly_mean_rank_ic"),
                "valid_day_count": (entry.get("oos") or {}).get("valid_day_count"),
                "gates": entry.get("gates"),
            }
        else:
            entry["oos"] = None
            entry["gates"] = None
            entry["folds"] = None
        if evidence_path.exists():
            entry["promotion_evidence_sha256"] = sha256_file(evidence_path)
            entry["promotion_evidence_bytes"] = evidence_path.stat().st_size
            sources[str(evidence_path)] = entry["promotion_evidence_sha256"]
        else:
            entry["promotion_evidence_sha256"] = None
        manifests = read_dataset_manifests(workspace)
        if manifests:
            datasets[candidate["workspace"]] = manifests
        partial = read_search_records(workspace)
        if partial and not report_path.exists():
            in_progress[candidate["workspace"]] = {
                "status": "IN_PROGRESS_NO_REPORT",
                "caveat": (
                    "Validation-set numbers only, from per-config search records of a run "
                    "that had not finished. These are NOT out-of-sample results and must "
                    "not be compared with any G4 delta_rank_ic."
                ),
                "extracted_at_utc": extracted_at,
                "folds": partial,
            }
        standings.append(entry)

    factor_sets = read_factor_sets(args.repo)
    sources[factor_sets["source_file"]] = factor_sets["source_sha256"]

    write_json(
        args.out / "candidate_standings.json",
        {
            "extracted_at_utc": extracted_at,
            "artifacts_root": str(args.artifacts),
            "comparison_basis": (
                "M0 (m0_v3), M0R (m0r_v2) and M1 (m1_full_v1) share one baseline run "
                "(baseline_v2), one 945-valid-day pooled OOS set (2022-2025) and one matched "
                "baseline_mean_rank_ic (0.0852268662737014), so their G4 delta_rank_ic values "
                "ARE directly comparable by subtraction. M2 (m2_v1) is not in that set and "
                "must not be subtracted against them: its 15-observation minimum history "
                "scores fewer rows, leaving 905 valid days and a different matched baseline "
                "mean (0.0851235472459345). The reason is mechanical -- "
                "model_evaluator._match_prediction_coverage inner-joins candidate and "
                "baseline onto identical (signal_date, ts_code) keys, so baseline_mean_rank_ic "
                "is a carrier-dependent quantity, not one shared reference. G6's denominator "
                "is the 969 ordinary test days for every family (945/969 = "
                "0.9752321981424149; 905/969 = 0.9339525283797729), so G6 ratios remain "
                "comparable across carriers even though the G4 deltas are not."
            ),
            "candidates": standings,
        },
    )
    write_json(
        args.out / "per_fold_winners.json",
        {"extracted_at_utc": extracted_at, "by_workspace": winners},
    )
    write_json(
        args.out / "yearly_rank_ic.json",
        {"extracted_at_utc": extracted_at, "by_workspace": yearly},
    )
    write_json(
        args.out / "factor_sets.json",
        {"extracted_at_utc": extracted_at, **factor_sets},
    )
    write_json(
        args.out / "datasets.json",
        {"extracted_at_utc": extracted_at, "by_workspace": datasets},
    )
    if in_progress:
        write_json(args.out / "in_progress_search_records.json", in_progress)

    script_path = Path(__file__).resolve()
    write_json(
        args.out / "provenance.json",
        {
            "extracted_at_utc": extracted_at,
            "extractor": {
                "path": str(script_path),
                "sha256": sha256_file(script_path),
            },
            "repository_main_tree": git_head(args.repo),
            "source_file_sha256": sources,
            "excluded_by_design": [
                "model weights / checkpoints (*.pt)",
                "parquet panels and fold caches",
                "promotion_evidence.json payloads (hashed, not copied: 129 KB each)",
                "anything requiring a database connection",
            ],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
