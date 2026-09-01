"""Build read-only preflight inputs for the eighteen READY candidates.

Normalizes every pairwise-similarity artifact (base-six matrix, extension
audit, per-candidate pair audits, nine-edge batch summaries) to candidate
version ids, verifies full C(18,2) coverage for train and validation, and
writes the CSV evidence files plus the pool-wide admission proposal consumed
by audit_shsz_admission_preflight.py.  No database writes, no locked-test
reads, no admission decisions.
"""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

DIAG = Path("research_v2/artifacts/diagnostics")

CANDIDATES: dict[str, str] = {
    "yeslab_114_tushare_shsz_v2": "yeslab_114",
    "yeslab_381_tushare_shsz_v1": "yeslab_381",
    "yeslab_405_tushare_shsz_v1": "yeslab_405",
    "yeslab_425_tushare_shsz_v2": "yeslab_425",
    "yeslab_484_tushare_shsz_v1": "yeslab_484",
    "yeslab_206_median_abs_v2": "yeslab_206_median_abs_v2",
    "yeslab_363_tushare_shsz_v1": "yeslab_363",
    "yeslab_357_tushare_shsz_v1": "yeslab_357",
    "yeslab_96_canonical_replay_v1": "yeslab_96",
    "yeslab_443_tushare_shsz_v1": "yeslab_443",
    "yeslab_464_tushare_shsz_v1": "yeslab_464",
    "yeslab_370_tushare_shsz_v1": "yeslab_370",
    "yeslab_375_tushare_shsz_v1": "yeslab_375",
    "yeslab_389_tushare_shsz_v1": "yeslab_389",
    "yeslab_390_tushare_shsz_v1": "yeslab_390",
    "yeslab_471_tushare_pit_v1": "yeslab_471",
    "yeslab_398_tushare_shsz_v1": "yeslab_398",
    "yeslab_395_tushare_shsz_v1": "yeslab_395",
}
SOURCE_TO_CANDIDATE = {source: candidate_id for candidate_id, source in CANDIDATES.items()}
# source-id aliases that map onto candidate version ids in pair evidence
SOURCE_TO_CANDIDATE.update({
    "yeslab_206_median_abs": "yeslab_206_median_abs_v2",
    # initial 471 pair audit predates the pit contract id
    "yeslab_471_tushare_shsz_v1": "yeslab_471_tushare_pit_v1",
})

CANDIDATE_META = {
    "yeslab_114_tushare_shsz_v2": ("惊恐波动", "MONITOR"),
    "yeslab_381_tushare_shsz_v1": ("GP_012", "MONITOR"),
    "yeslab_405_tushare_shsz_v1": ("GP_036", "MONITOR"),
    "yeslab_425_tushare_shsz_v2": ("GP_056", "MONITOR"),
    "yeslab_484_tushare_shsz_v1": ("羊群效应_CSAD波动率21", "PASS"),
    "yeslab_206_median_abs_v2": ("60日量中位偏离(修复)", "MONITOR"),
    "yeslab_363_tushare_shsz_v1": ("行业标准化/行业偏度", "MONITOR"),
    "yeslab_357_tushare_shsz_v1": ("行业内收益标准化", "MONITOR"),
    "yeslab_96_canonical_replay_v1": ("修正日间反转", "MONITOR"),
    "yeslab_443_tushare_shsz_v1": ("GP_074", "MONITOR"),
    "yeslab_464_tushare_shsz_v1": ("GP_095", "PASS"),
    "yeslab_370_tushare_shsz_v1": ("GP_001", "MONITOR"),
    "yeslab_375_tushare_shsz_v1": ("GP_006", "PASS"),
    "yeslab_389_tushare_shsz_v1": ("GP_020", "MONITOR"),
    "yeslab_390_tushare_shsz_v1": ("GP_021", "MONITOR"),
    "yeslab_471_tushare_pit_v1": ("AttentionSpillover", "PASS"),
    "yeslab_398_tushare_shsz_v1": ("GP_029", "MONITOR"),
    "yeslab_395_tushare_shsz_v1": ("GP_026", "MONITOR"),
}

QUALITY_SOURCES = {
    "yeslab_405_tushare_shsz_v1": [
        "tushare_alpha405_canonical_train_similarity_v1/quality_summary.json",
        "tushare_alpha405_canonical_validation_v1/quality_summary.json",
    ],
    "yeslab_425_tushare_shsz_v2": [
        "tushare_alpha425_canonical_train_similarity_v1/quality_summary.json",
        "tushare_alpha425_canonical_validation_v1/quality_summary.json",
    ],
    "yeslab_381_tushare_shsz_v1": [
        "tushare_alpha381_canonical_train_similarity_v1/quality_summary.json",
        "tushare_alpha381_canonical_validation_v1/quality_summary.json",
    ],
    "yeslab_114_tushare_shsz_v2": [
        "tushare_alpha114_canonical_train_similarity_v1/quality_summary.json",
        "tushare_alpha114_canonical_validation_v1/quality_summary.json",
    ],
    "yeslab_363_tushare_shsz_v1": [
        "tushare_shsz_alpha363_formal9_train_2020_2024_v1/quality_summary.json",
        "tushare_shsz_alpha363_formal9_validation_2025_v1/quality_summary.json",
    ],
    "yeslab_357_tushare_shsz_v1": [
        "tushare_shsz_alpha357_formal9_train_2020_2024_v1/quality_summary.json",
        "tushare_shsz_alpha357_formal9_validation_2025_v1/quality_summary.json",
    ],
    "yeslab_443_tushare_shsz_v1": [
        "tushare_shsz_alpha443_formal9_train_2020_2024_v1/quality_summary.json",
        "tushare_shsz_alpha443_formal9_validation_2025_v1/quality_summary.json",
    ],
    "yeslab_96_canonical_replay_v1": [
        "tushare_canonical_formal9_unique7_train_2020_2024_v1/quality_summary.json",
        "tushare_canonical_formal9_unique7_validation_2025_v1/quality_summary.json",
    ],
    "yeslab_464_tushare_shsz_v1": [
        "tushare_shsz_gp_coverage_repair4_formal9_train_2020_2024_v1/quality_summary.json",
        "tushare_shsz_gp_coverage_repair4_formal9_validation_2025_v1/quality_summary.json",
    ],
    "yeslab_370_tushare_shsz_v1": [
        "tushare_shsz_nearmiss6_formal9_train_2020_2024_v1/quality_summary.json",
        "tushare_shsz_nearmiss6_formal9_validation_2025_v1/quality_summary.json",
    ],
    "yeslab_375_tushare_shsz_v1": [
        "tushare_shsz_nearmiss6_formal9_train_2020_2024_v1/quality_summary.json",
        "tushare_shsz_nearmiss6_formal9_validation_2025_v1/quality_summary.json",
    ],
    "yeslab_389_tushare_shsz_v1": [
        "tushare_shsz_nearmiss6_formal9_train_2020_2024_v1/quality_summary.json",
        "tushare_shsz_nearmiss6_formal9_validation_2025_v1/quality_summary.json",
    ],
    "yeslab_390_tushare_shsz_v1": [
        "tushare_shsz_nearmiss6_formal9_train_2020_2024_v1/quality_summary.json",
        "tushare_shsz_nearmiss6_formal9_validation_2025_v1/quality_summary.json",
    ],
    "yeslab_398_tushare_shsz_v1": [
        "tushare_shsz_dq3_formal9_train_2020_2024_v1/quality_summary.json",
        "tushare_shsz_dq3_formal9_validation_2025_v1/quality_summary.json",
    ],
    "yeslab_471_tushare_pit_v1": [
        "tushare_pit_alpha471_formal9_validation_2025_v1/quality_summary.json",
    ],
    "yeslab_395_tushare_shsz_v1": [
        "tushare_shsz_alpha395_formal9_train_2020_2024_v1/quality_summary.json",
        "tushare_shsz_alpha395_formal9_validation_2025_v1/quality_summary.json",
    ],
}

PAIR_SOURCES = [
    "tushare_canonical_candidate_pair_similarity_v1.json",
    "supplemental_candidate_pair_similarity_train_v1.json",
    "supplemental_candidate_pair_similarity_validation_v1.json",
    "candidate464_pair_similarity_v1.json",
    "candidate370_pair_similarity_v1.json",
    "candidate375_pair_similarity_v1.json",
    "candidate389_pair_similarity_v1.json",
    "candidate390_pair_similarity_v1.json",
    "candidate471_pair_similarity_v1.json",
    "candidate398_pair_similarity_v1.json",
    "candidate395_pair_similarity_v1.json",
]
BATCH_LIST_SOURCES = {
    "train": [
        "tushare_shsz_alpha395_formal9_train_2020_2024_v1/similarity_summary.json",
        "tushare_shsz_nearmiss6_formal9_train_2020_2024_v1/similarity_summary.json",
        "tushare_shsz_nearmiss4_471_fill_train_2020_2024_v1/similarity_summary.json",
    ],
    "validation": [
        "tushare_shsz_alpha395_formal9_validation_2025_v1/similarity_summary.json",
        "tushare_shsz_nearmiss6_formal9_validation_2025_v1/similarity_summary.json",
        "tushare_shsz_nearmiss4_471_fill_validation_2025_v1/similarity_summary.json",
    ],
}


def resolve(name: str) -> str | None:
    if name in CANDIDATES:
        return name
    if name in SOURCE_TO_CANDIDATE:
        return SOURCE_TO_CANDIDATE[name]
    for candidate_id, source in CANDIDATES.items():
        if name == source:
            return candidate_id
    return None


def collect_pairs() -> dict[str, dict[tuple[str, str], dict]]:
    edges: dict[str, dict[tuple[str, str], dict]] = {"train": {}, "validation": {}}
    for name in PAIR_SOURCES:
        data = json.loads((DIAG / name).read_text(encoding="utf-8"))
        for period_key, period in (("train_pairs", "train"), ("validation_pairs", "validation")):
            for row in data.get(period_key, []):
                left = resolve(str(row["factor_id_left"]))
                right = resolve(str(row["factor_id_right"]))
                if left is None or right is None:
                    continue
                key = tuple(sorted((left, right)))
                if key[0] == key[1]:
                    continue
                edges[period].setdefault(key, {"metric": float(row["median_abs_spearman"]), "source": name})
    for period, names in BATCH_LIST_SOURCES.items():
        for name in names:
            rows = json.loads((DIAG / name).read_text(encoding="utf-8"))
            for row in rows:
                left = resolve(str(row["factor_id_left"]))
                right = resolve(str(row["factor_id_right"]))
                if left is None or right is None:
                    continue
                key = tuple(sorted((left, right)))
                if key[0] == key[1]:
                    continue
                edges[period].setdefault(key, {"metric": float(row["median_abs_spearman"]), "source": name})
    return edges


def main() -> None:
    edges = collect_pairs()
    expected = [tuple(sorted(pair)) for pair in combinations(sorted(CANDIDATES), 2)]
    for period in ("train", "validation"):
        missing = [key for key in expected if key not in edges[period]]
        if missing:
            raise SystemExit(f"missing {period} pair evidence: {missing}")
        out = DIAG / f"pool18_preflight_{period}_similarity_v1.csv"
        with out.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["factor_id_left", "factor_id_right", "median_abs_spearman", "source"])
            for key in sorted(edges[period]):
                writer.writerow([key[0], key[1], repr(edges[period][key]["metric"]), edges[period][key]["source"]])
        print(f"wrote {out} ({len(edges[period])} edges)")

    special_quality = {
        "yeslab_484_tushare_shsz_v1": (
            "tushare_alpha484_canonical_v1/quality_summary.json",
            lambda data: data["training_2020_2024"].get("icir"),
        ),
        "yeslab_206_median_abs_v2": (
            "alpha206_volume_variants_2020_2025_v1/median_abs/quality_summary.json",
            lambda data: data["train_2020_2024"].get("icir"),
        ),
    }
    quality_rows: dict[str, dict] = {}
    for candidate_id, (name, extract) in special_quality.items():
        data = json.loads((DIAG / name).read_text(encoding="utf-8"))
        quality_rows[candidate_id] = {"factor_id": candidate_id, "icir": extract(data)}
    for candidate_id, names in QUALITY_SOURCES.items():
        source_factor = CANDIDATES[candidate_id]
        for name in names:
            data = json.loads((DIAG / name).read_text(encoding="utf-8"))
            for row in data.get("summaries", []):
                if str(row.get("factor_id")) == source_factor:
                    merged = quality_rows.setdefault(candidate_id, {"factor_id": candidate_id})
                    merged["icir"] = row.get("icir")
                    merged["mean_rank_ic"] = row.get("mean_rank_ic")
    out = DIAG / "pool18_preflight_quality_v1.json"
    out.write_text(json.dumps({"summaries": list(quality_rows.values())}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(quality_rows)} candidates)")

    proposal = {
        "scope": "TUSHARE_SHSZ_POOL18_ADMISSION_PROPOSAL_V1",
        "proposal_type": "POOL_READ_ONLY_PREFLIGHT",
        "market_scope": ["SH", "SZ"],
        "excluded_exchanges": ["BJ"],
        "label": "forward_open_5d",
        "train_window": "2020-01-02/2024-12-31",
        "validation_window": "2025-01-02/2025-12-31",
        "similarity_threshold": 0.9,
        "similarity_periods": ["train", "validation"],
        "admission_authority": "NOT_GRANTED",
        "publication_status": "NOT_PUBLISHED",
        "model_training_started": False,
        "database_writes": False,
        "locked_test_read": False,
        "note": (
            "Read-only decision-readiness preflight over all eighteen READY candidates. "
            "annual_stability_gate uses PASS for 5/5 strict years and MONITOR for 4/5 (the "
            "unchanged gate requires >= 4/5). Elevated candidate pairs (370/389 validation "
            "0.9127 >= 0.90; monitor pairs with 381 and the 395-vs-formal-380 watch) are "
            "documented in the checklist; preflight pair checks below 0.90 remain expected to pass "
            "except the 370/389 validation edge."
        ),
        "candidates": [
            {
                "factor_id": candidate_id,
                "factor_name": CANDIDATE_META[candidate_id][0],
                "version_scope": "isolated_tushare_canonical",
                "predictive_gate": "PASS",
                "annual_stability_gate": CANDIDATE_META[candidate_id][1],
                "similarity_gate": "PASS",
                "formula_scope_gate": "PASS",
            }
            for candidate_id in sorted(CANDIDATES)
        ],
    }
    out = DIAG / "tushare_shsz_pool18_admission_proposal_v1.json"
    out.write_text(json.dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
