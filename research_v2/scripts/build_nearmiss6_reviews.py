"""Build candidate review artifacts for the near-miss six (370/375/389/390/415/213).

Reads training and validation evaluation outputs plus (optionally) the
candidate-vs-READY-pool pair similarity artifacts, and writes one candidate
review JSON per factor in the style of the coverage-repair four reviews.

Read-only: no database writes, no locked-test reads, no publication.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CANDIDATES = {
    "yeslab_370": ("GP_001", None),
    "yeslab_375": ("GP_006", None),
    "yeslab_389": ("GP_020", None),
    "yeslab_390": ("GP_021", None),
    "yeslab_415": ("GP_046", "REJECT_STRICT_ANNUAL_STABILITY_3_OF_5"),
    "yeslab_213": ("量价齐飞加换手率", "REJECT_AGGREGATE_ICIR_AND_ANNUAL_0_OF_5"),
}
FORMAL_NINE = [
    "yeslab_380", "yeslab_431", "yeslab_437", "yeslab_155",
    "yeslab_98", "yeslab_94", "yeslab_376", "yeslab_396", "yeslab_442",
]
ANNUAL_PASS = {
    "yeslab_370": "4/5", "yeslab_375": "5/5", "yeslab_389": "4/5",
    "yeslab_390": "4/5", "yeslab_415": "3/5", "yeslab_213": "0/5",
}
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--direction-artifact", type=Path, required=True)
    parser.add_argument("--pair-artifact", type=Path, action="append", default=[],
                        help="candidateNNN_pair_similarity_v1.json files (optional)")
    args = parser.parse_args()

    directions = {
        str(r.factor_id): int(r.direction)
        for r in pd.read_csv(args.direction_artifact).itertuples(index=False)
    }
    train_quality = json.loads((args.train_dir / "quality_summary.json").read_text(encoding="utf-8"))
    train_similarity = json.loads((args.train_dir / "similarity_summary.json").read_text(encoding="utf-8"))
    validation_quality = json.loads((args.validation_dir / "quality_summary.json").read_text(encoding="utf-8"))
    validation_similarity = json.loads((args.validation_dir / "similarity_summary.json").read_text(encoding="utf-8"))
    train_rows = {row["factor_id"]: row for row in train_quality.get("summaries", [])}
    validation_rows = {row["factor_id"]: row for row in validation_quality.get("summaries", [])}

    pair_info: dict[str, dict] = {}
    for path in args.pair_artifact:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        candidate_id = payload.get("new_candidate_id")
        source = str(payload.get("source_factor_id"))
        pairs = []
        for window in ("train_pairs", "validation_pairs"):
            for row in payload.get(window, []):
                pairs.append((window, row["factor_id_left"], row["factor_id_right"], row["median_abs_spearman"]))
        pair_info[source] = {"candidate_id": candidate_id, "pairs": pairs, "status": payload.get("status")}

    for factor_id, (name, forced_reject) in CANDIDATES.items():
        direction = directions[factor_id]
        train_gate = _gate(train_rows[factor_id], direction)
        validation_gate = _gate(validation_rows[factor_id], direction)
        review = {
            "scope": "TUSHARE_SHSZ_NEARMISS6_CANDIDATE_REVIEW_V1",
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
                "strict_annual_pass_years": ANNUAL_PASS[factor_id],
            },
            "validation_2025": {
                "start": "2025-01-02",
                "end": "2025-12-31",
                **validation_gate,
                "mean_rank_ic": validation_rows[factor_id]["mean_rank_ic"],
                "icir": validation_rows[factor_id]["icir"],
            },
            "similarity_to_formal_nine_train": nine_edges(train_similarity, factor_id),
            "similarity_to_formal_nine_validation": nine_edges(validation_similarity, factor_id),
            "database_writes": False,
            "locked_test_read": False,
            "formal_factor_set_changed": False,
            "model_training_started": False,
            "publication_status": "NOT_PUBLISHED",
        }
        if factor_id in pair_info:
            info = pair_info[factor_id]
            maxima = {}
            for window, left, right, value in info["pairs"]:
                maxima[window] = max(maxima.get(window, 0.0), value)
            review["candidate_pair_similarity"] = {
                "reference_pool": "eleven READY candidates",
                "training_maximum_median_abs_spearman": maxima.get("train_pairs"),
                "validation_maximum_median_abs_spearman": maxima.get("validation_pairs"),
                "status": info["status"],
            }
        if forced_reject:
            review["decision"] = forced_reject
        else:
            train_ok = train_gate["rank_ic_gate"] and train_gate["icir_gate"] and train_gate["day_share_gate"]
            val_ok = validation_gate["rank_ic_gate"] and validation_gate["icir_gate"] and validation_gate["day_share_gate"]
            annual_ok = int(ANNUAL_PASS[factor_id][0]) >= THRESHOLDS["annual_pass_years"]
            if not (train_ok and val_ok and annual_ok):
                review["decision"] = "REJECT_GATE"
            elif factor_id not in pair_info:
                review["decision"] = "GATES_PASS_PAIR_EVIDENCE_PENDING"
            else:
                review["decision"] = "READY_FOR_EXPLICIT_ADMISSION_REVIEW_WITH_SCOPE_APPROVAL"
        out = Path(f"research_v2/artifacts/diagnostics/tushare_{factor_id}_nearmiss6_review_v1.json")
        out.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out} -> {review['decision']}")


if __name__ == "__main__":
    main()
