"""Build the consolidated final-status ledger for the whole 487-factor catalog.

Merges, in priority order:
  1. the explicit admission review checklist (authoritative current statuses),
  2. band evaluation ledgers (band10 / nearmiss6 / dq3 / repair4 decisions),
  3. the structural-degeneracy family rejections,
  4. previously documented canonical evaluations in the coverage ledger
     (precedent upgrades, monitored candidates, older rejects),
  5. the 2020 historical screen (everything else: screened only).

Read-only consolidation; no database writes, no locked-test reads, no
admission decisions.
"""

from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG = REPO_ROOT / "research_v2" / "artifacts" / "diagnostics"

FORMAL_NINE = {
    "yeslab_380", "yeslab_431", "yeslab_437", "yeslab_155",
    "yeslab_98", "yeslab_94", "yeslab_376", "yeslab_396", "yeslab_442",
}
DEGENERATE_FAMILY = {
    "yeslab_408", "yeslab_416", "yeslab_422", "yeslab_424", "yeslab_426",
    "yeslab_427", "yeslab_435", "yeslab_436", "yeslab_446", "yeslab_449",
    "yeslab_450", "yeslab_460", "yeslab_462", "yeslab_463", "yeslab_465",
    "yeslab_466", "yeslab_468", "yeslab_411", "yeslab_421",
}
# previously documented canonical evaluations (coverage ledger followups)
PRIOR_EVALUATED = {
    "yeslab_31": "REJECT_CANONICAL_STRICT_GATE (train ICIR 0.20-class fail, see alpha31/alpha62 rematerialization audit)",
    "yeslab_71": "REJECT_CANONICAL_STRICT_GATE (alpha62 rematerialization audit)",
    "yeslab_105": "REJECT_CANONICAL_ANNUAL_3_OF_5 (alpha83 rematerialization audit)",
    "yeslab_346": "REJECT_ANNUAL_STABILITY_FOLLOWUP",
    "yeslab_429": "MONITORED_RESEARCH_CANDIDATE_STRICT_ANNUAL_2_OF_5",
    "yeslab_401": "REJECT_2020_PREDICTIVE_GATE",
    "yeslab_385": "REJECT_REDUNDANT_WITH_CANDIDATE_405",
    "yeslab_362": "NO_ADMISSION_STRICT_ANNUAL_ICIR_FAIL_AND_FORMULA_REPLAY_BLOCKED",
    "yeslab_481": "REJECT_TRAINING_AGGREGATE_ICIR_BELOW_GATE",
    "yeslab_482": "REJECT_TRAINING_AGGREGATE_ICIR_BELOW_GATE",
    "yeslab_165": "REJECT_STRICT_ANNUAL_1_OF_5",
    "yeslab_168": "REJECT_STRICT_ANNUAL_1_OF_5",
    "yeslab_167": "MONITORED_RESEARCH_CANDIDATE_STRICT_ANNUAL_3_OF_5",
    "yeslab_95": "MONITORED_RESEARCH_CANDIDATE_STRICT_ANNUAL_2_OF_5",
    "yeslab_163": "REJECT_STRICT_ANNUAL_STABILITY_3_OF_5",
    "yeslab_393": "REJECT_REDUNDANT_WITH_FORMAL_376 (annual structure passes; documented in alpha_393_395_train_2020_2024_annual_v1.json)",
    "yeslab_397": "REJECT_STRICT_ANNUAL_STABILITY_2_OF_5 (pass7 canonical evaluation)",
    "yeslab_423": "REJECT_STRICT_ANNUAL_STABILITY_3_OF_5 (pass7 canonical evaluation)",
    "yeslab_46": "REJECT_CANONICAL_TRAINING_GATE_ANNUAL_1_OF_5 (secondary12 verified mapping, canonical ICIR 0.370)",
    "yeslab_140": "REJECT_CANONICAL_TRAINING_GATE_ANNUAL_1_OF_5 (identical to 46, mutual 0.997)",
    "yeslab_209": "REJECT_CANONICAL_TRAINING_GATE_ANNUAL_3_OF_5 (canonical ICIR 0.419)",
    "yeslab_211": "REJECT_CANONICAL_TRAINING_GATE_ANNUAL_1_OF_5 (canonical ICIR 0.337)",
    "yeslab_214": "REJECT_CANONICAL_TRAINING_GATE_ANNUAL_3_OF_5 (canonical ICIR 0.478)",
    "yeslab_392": "REJECT_CANONICAL_TRAINING_ICIR_0.4356_ANNUAL_1_OF_5 (borderline-band GP evaluated; uplift did not generalize)",
    "yeslab_83": "MAPPING_UNVERIFIED (replay mismatch 100%, max_abs 1.09; no canonical evaluation)",
    "yeslab_212": "MAPPING_UNVERIFIED (replay mismatch 99.99%, max_abs 1.49; no canonical evaluation)",
    "yeslab_252": "REJECT_TRAINING_AGGREGATE_ICIR_0.4957_BELOW_0.50 (mapping verified; annual 4/5)",
    "yeslab_260": "REJECT_CANONICAL_TRAINING_GATE_ANNUAL_2_OF_5 (mapping verified)",
    "yeslab_271": "REJECT_CANONICAL_TRAINING_GATE_ANNUAL_2_OF_5 (mapping verified)",
    "yeslab_272": "REJECT_CANONICAL_TRAINING_GATE_ANNUAL_1_OF_5 (mapping verified)",
    "yeslab_359": "REPLAY_BLOCKED_INDUSTRY_CLASS (industry classification PIT vs historical static; no canonical evaluation)",
}


def load_screen() -> dict[str, dict]:
    ledger: dict[str, dict] = {}
    for path in glob.glob(str(DIAG / "alpha_*_train_2020_quality_v1.json")):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        for fid, row in data.get("factors", {}).items():
            if isinstance(row, dict) and fid.startswith("yeslab_"):
                ledger[fid] = row
    return ledger


def main() -> None:
    import yaml
    catalog = yaml.safe_load((REPO_ROOT / "research_v2/configs/factors/source_catalog.yaml").read_text(encoding="utf-8"))
    names = {e["factor_id"]: str(e.get("source_factor_name", "")) for e in catalog["factors"]}
    screen = load_screen()

    checklist = json.loads((DIAG / "explicit_admission_review_checklist_v1.json").read_text(encoding="utf-8"))["checklists"]
    checklist_status: dict[str, dict] = {}
    for entry in checklist:
        source = entry.get("source_factor_id")
        if source:
            checklist_status[source] = entry

    hist_rows: dict[str, dict] = {}
    hist_path = DIAG / "historical_window_2019_robustness_v1" / "historical_window_2019_comparison.json"
    if hist_path.exists():
        hist = json.loads(hist_path.read_text(encoding="utf-8"))
        cand_to_src = {
            str(e.get("candidate_id")): str(e.get("source_factor_id"))
            for e in checklist
            if e.get("candidate_id")
        }
        for row in hist.get("rows", []):
            cid = str(row["factor_id"])
            hist_rows[cand_to_src.get(cid, cid)] = row

    band_decisions: dict[str, str] = {}
    for ledger_name in ("band10_train_evidence_v1.json", "nearmiss6_train_evidence_v1.json",
                        "dq3_train_evidence_v1.json", "gp_coverage_repair4_train_evidence_v1.json"):
        path = DIAG / ledger_name
        if not path.exists():
            continue
        for fid, row in json.loads(path.read_text(encoding="utf-8")).get("candidates", {}).items():
            decision = row.get("decision") or row.get("training_decision") or ""
            if decision:
                band_decisions[fid] = decision

    entries = []
    for fid in sorted(names, key=lambda x: int(x.replace("yeslab_", ""))):
        name = names[fid]
        row = screen.get(fid, {})
        entry = {
            "factor_id": fid,
            "factor_name": name,
            "screen_2020_signed_icir": round(abs(row.get("icir") or 0), 4) if row.get("icir") is not None else None,
            "screen_2020_days": row.get("valid_day_count"),
        }
        if fid in FORMAL_NINE:
            entry["final_status"] = "FORMAL_NINE"
            entry["status_class"] = "FORMAL"
        elif fid in checklist_status:
            cl = checklist_status[fid]
            entry["final_status"] = cl["status"]
            entry["candidate_id"] = cl.get("candidate_id")
            entry["status_class"] = "READY" if cl["status"].startswith("READY") else ("MONITORED" if cl["status"].startswith("MONITORED") else "REJECT_OR_BLOCKED")
        elif fid in band_decisions:
            entry["final_status"] = band_decisions[fid]
            entry["status_class"] = "READY" if band_decisions[fid].startswith(("CANDIDATE", "READY")) else "REJECT_OR_BLOCKED"
        elif fid in DEGENERATE_FAMILY:
            entry["final_status"] = "REJECT_STRUCTURAL_DEGENERACY_TURNOVER_SCALAR_FAMILY"
            entry["status_class"] = "REJECT_OR_BLOCKED"
        elif fid in PRIOR_EVALUATED:
            entry["final_status"] = PRIOR_EVALUATED[fid]
            entry["status_class"] = "MONITORED" if PRIOR_EVALUATED[fid].startswith("MONITORED") else "REJECT_OR_BLOCKED"
        else:
            entry["final_status"] = "NOT_EVALUATED_BEYOND_2020_SCREEN"
            entry["status_class"] = "SCREENED_ONLY"
        if fid in hist_rows:
            hr = hist_rows[fid]
            entry["historical_window_2019"] = {
                "ic_full_market": hr.get("ic_2019_full"),
                "icir_full_market": hr.get("icir_2019_full"),
                "ic_adv20_liquid": hr.get("ic_2019_liquid"),
                "icir_adv20_liquid": hr.get("icir_2019_liquid"),
                "verdict": hr.get("classify_overall"),
                "note": "pre-training-period robustness window; Tushare-sourced replay and labels validated equivalent to V2 on 2020",
            }
        entries.append(entry)

    classes = Counter(e["status_class"] for e in entries)
    ready = [e["factor_id"] for e in entries if e["status_class"] == "READY"]
    formal = [e["factor_id"] for e in entries if e["status_class"] == "FORMAL"]
    monitored = [e["factor_id"] for e in entries if e["status_class"] == "MONITORED"]
    rejected = [e["factor_id"] for e in entries if e["status_class"] == "REJECT_OR_BLOCKED"]
    payload = {
        "scope": "FACTOR_CATALOG_FINAL_STATUS_V1",
        "market_scope": "SH/SZ",
        "excluded_exchanges": ["BJ"],
        "catalog_factor_count": len(entries),
        "status_class_counts": dict(classes),
        "summary": {
            "formal_nine": sorted(formal),
            "ready_candidates": sorted(ready),
            "monitored": sorted(monitored),
            "rejected_or_blocked_count": len(rejected),
        },
        "status_definitions": {
            "FORMAL": "already in the formal nine factor set",
            "READY": "admission-ready with complete evidence; awaiting explicit human admission decision",
            "MONITORED": "research candidate kept under monitoring (annual stability etc.), not admission-ready",
            "REJECT_OR_BLOCKED": "rejected or blocked with a documented reason",
            "SCREENED_ONLY": "only the historical 2020 screen exists; no canonical evaluation",
        },
        "entries": entries,
        "database_writes": False,
        "locked_test_read": False,
        "formal_factor_set_changed": False,
        "model_training_started": False,
    }
    out = DIAG / "factor_catalog_final_status_v1.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print("status class counts:", dict(classes))
    print("READY:", len(ready), sorted(ready))
    print("monitored:", sorted(monitored))


if __name__ == "__main__":
    main()
