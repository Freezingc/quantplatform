"""Build candidate review artifacts for the GP coverage-repair four (436/446/464/468).

Reads the training and (when present) validation evaluation outputs produced by
evaluate_tushare_scope_validation.py and writes one candidate review JSON per
factor in the style of tushare_alpha443_canonical_candidate_review_v1.json.

Read-only: no database writes, no locked-test reads, no publication.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

NEW_CANDIDATES = {
    "yeslab_436": ("GP_067", "REJECT_STRUCTURAL_DEGENERACY_TURNOVER_SCALAR"),
    "yeslab_446": ("GP_077", "REJECT_STRUCTURAL_DEGENERACY_TURNOVER_SCALAR"),
    "yeslab_468": ("GP_099", "REJECT_STRUCTURAL_DEGENERACY_TURNOVER_SCALAR"),
    "yeslab_464": ("GP_095", None),
}
FORMAL_NINE = [
    "yeslab_380", "yeslab_431", "yeslab_437", "yeslab_155",
    "yeslab_98", "yeslab_94", "yeslab_376", "yeslab_396", "yeslab_442",
]
THRESHOLDS = {"rank_ic": 0.02, "icir": 0.5, "directional_day_share": 0.55, "annual_pass_years": 4}


def _gate(row: dict, direction: int) -> dict:
    signed_ic = direction * row["mean_rank_ic"]
    signed_icir = direction * row["icir"]
    return {
        "signed_mean_rank_ic": signed_ic,
        "signed_icir": signed_icir,
        "directional_day_share": row.get("directional_day_share"),
        "valid_day_count": row.get("valid_day_count"),
        "rank_ic_gate": signed_ic > THRESHOLDS["rank_ic"],
        "icir_gate": signed_icir >= THRESHOLDS["icir"],
        "day_share_gate": (row.get("directional_day_share") or 0) >= THRESHOLDS["directional_day_share"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, default=None)
    parser.add_argument("--direction-artifact", type=Path, required=True)
    parser.add_argument("--output-prefix", type=str, default="research_v2/artifacts/diagnostics/tushare_alpha")
    args = parser.parse_args()

    directions = {
        str(r.factor_id): int(r.direction)
        for r in pd.read_csv(args.direction_artifact).itertuples(index=False)
    }
    train_quality = json.loads((args.train_dir / "quality_summary.json").read_text(encoding="utf-8"))
    train_similarity = json.loads((args.train_dir / "similarity_summary.json").read_text(encoding="utf-8"))
    train_rows = {row["factor_id"]: row for row in train_quality.get("summaries", train_quality.get("factors", []))}
    validation_rows: dict[str, dict] = {}
    validation_similarity: list[dict] = []
    if args.validation_dir is not None and (args.validation_dir / "quality_summary.json").exists():
        validation_quality = json.loads((args.validation_dir / "quality_summary.json").read_text(encoding="utf-8"))
        validation_rows = {row["factor_id"]: row for row in validation_quality.get("summaries", validation_quality.get("factors", []))}
        validation_similarity = json.loads((args.validation_dir / "similarity_summary.json").read_text(encoding="utf-8"))

    def nine_edges(similarity: list[dict], factor_id: str) -> dict:
        edges = {}
        for row in similarity:
            left, right = row["factor_id_left"], row["factor_id_right"]
            if factor_id in (left, right):
                other = right if left == factor_id else left
                if other in FORMAL_NINE:
                    edges[other] = row["median_abs_spearman"]
        if len(edges) != 9:
            raise ValueError(f"{factor_id}: expected 9 formal-nine edges, found {len(edges)}")
        nearest = max(edges, key=edges.get)
        return {
            "edges": "9/9",
            "maximum_median_abs_spearman": edges[nearest],
            "nearest_factor_id": nearest,
            "threshold": 0.9,
            "status": "PASS" if edges[nearest] < 0.9 else "FAIL_REDUNDANT",
        }

    for factor_id, (name, structural_reject) in NEW_CANDIDATES.items():
        direction = directions[factor_id]
        train_gate = _gate(train_rows[factor_id], direction)
        review = {
            "scope": "TUSHARE_SHSZ_GP_COVERAGE_REPAIR4_CANDIDATE_REVIEW_V1",
            "candidate_id": f"{factor_id}_tushare_shsz_v1",
            "source_factor_id": factor_id,
            "factor_name": name,
            "market_scope": ["SH", "SZ"],
            "excluded_exchanges": ["BJ"],
            "market_source": "Tushare.daily",
            "label": "forward_open_5d:v1",
            "direction": direction,
            "training_2020_2024": {
                "start": "2020-01-02",
                "end": "2024-12-31",
                **train_gate,
                "mean_rank_ic": train_rows[factor_id]["mean_rank_ic"],
                "icir": train_rows[factor_id]["icir"],
            },
            "similarity_to_formal_nine_train": nine_edges(train_similarity, factor_id),
            "structural_degeneracy": structural_reject,
            "database_writes": False,
            "locked_test_read": False,
            "formal_factor_set_changed": False,
            "model_training_started": False,
            "publication_status": "NOT_PUBLISHED",
        }
        if structural_reject:
            review["decision"] = structural_reject
            review["decision_reason"] = (
                "Factor is structurally turnover_rate times a per-date cross-sectional scalar; "
                "see gp_turnover_scalar_degeneracy_diagnostic_v1.json. Admission excluded "
                "regardless of gate outcomes."
            )
        else:
            passes = train_gate["rank_ic_gate"] and train_gate["icir_gate"] and train_gate["day_share_gate"]
            review["training_gate_pass"] = bool(passes)
            if args.validation_dir is not None and factor_id in validation_rows:
                validation_gate = _gate(validation_rows[factor_id], direction)
                review["validation_2025"] = {
                    "start": "2025-01-02",
                    "end": "2025-12-31",
                    **validation_gate,
                    "mean_rank_ic": validation_rows[factor_id]["mean_rank_ic"],
                    "icir": validation_rows[factor_id]["icir"],
                }
                review["similarity_to_formal_nine_validation"] = nine_edges(validation_similarity, factor_id)
            if not passes:
                review["decision"] = "REJECT_TRAINING_GATE"
            elif args.validation_dir is not None and factor_id in validation_rows:
                vg = review["validation_2025"]
                vpass = vg["rank_ic_gate"] and vg["icir_gate"] and vg["day_share_gate"]
                review["decision"] = (
                    "READY_FOR_EXPLICIT_ADMISSION_REVIEW_WITH_SCOPE_APPROVAL"
                    if vpass else "REJECT_VALIDATION_GATE"
                )
            else:
                review["decision"] = "TRAINING_PASS_VALIDATION_PENDING"
        out = Path(f"{args.output_prefix}{factor_id.replace('yeslab_', '')}_coverage_repair4_review_v1.json")
        out.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out} -> {review['decision']}")


if __name__ == "__main__":
    main()
