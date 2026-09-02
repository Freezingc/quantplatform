"""Assemble the admission decision pack for all READY candidates.

Consolidates per-candidate evidence (quality gates, nine-edge similarity,
pool pair constraints, monthly stability, rank turnover, regime flags) into a
single machine-readable artifact that supports the human admission decision.

Read-only consolidation; no database writes, no locked-test reads, no
admission decisions.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG = REPO_ROOT / "research_v2" / "artifacts" / "diagnostics"


def load(name):
    path = DIAG / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    checklist = load("explicit_admission_review_checklist_v1.json")["checklists"]
    ready = [c for c in checklist if str(c.get("status", "")).startswith("READY")]

    turnover = load("pool23_rank_turnover_v1.json") or load("pool18_rank_turnover_v1.json") or {}
    turnover_candidates = turnover.get("candidates", {})
    regime = load("pool18_market_regime_robustness_v1.json") or {}
    regime_candidates = dict(regime.get("candidates", {}))
    pass7_regime = load("pass7_market_regime_robustness_v1.json") or {}
    for fid, windows in pass7_regime.get("candidates", {}).items():
        regime_candidates.setdefault(f"{fid}_tushare_shsz_v1", windows)

    monthly_rows = {}
    for path in glob.glob(str(DIAG / "*_validation_2025_monthly_v1/monthly_stability.json")):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cid = data.get("candidate_id")
        if cid:
            monthly_rows[cid] = data

    pack = []
    for entry in ready:
        cid = entry["candidate_id"]
        fid = entry.get("source_factor_id", "")
        item = {
            "candidate_id": cid,
            "source_factor_id": fid,
            "factor_name": entry.get("factor_name"),
            "status": entry.get("status"),
            "pair_or_combination_note": entry.get("pair_or_combination_note"),
            "quality": entry.get("quality"),
            "monthly_stability_2025": None,
            "rank_turnover": None,
            "regime_robustness": None,
            "evidence": entry.get("evidence", []),
        }
        m = monthly_rows.get(cid)
        if m:
            item["monthly_stability_2025"] = {
                "signed_mean_pass_months": m.get("signed_mean_pass_months"),
                "directional_share_pass_months": m.get("directional_share_pass_months"),
                "strict_icir_pass_months": m.get("strict_icir_pass_months"),
            }
        t = turnover_candidates.get(cid)
        if t:
            item["rank_turnover"] = {
                "train_autocorr": (t.get("train") or {}).get("rank_autocorr_mean"),
                "train_quintile_transition": (t.get("train") or {}).get("quintile_transition_rate"),
                "validation_quintile_transition": (t.get("validation") or {}).get("quintile_transition_rate"),
            }
        r = regime_candidates.get(cid)
        if r:
            flags = []
            for window in ("train", "validation"):
                w = r.get(window, {})
                for regime_name in ("above_ma120", "below_ma120", "low_vol", "mid_vol", "high_vol"):
                    cell = w.get(regime_name, {})
                    if cell.get("days", 0) >= 30 and (cell.get("signed_mean_rank_ic") or 0) <= 0:
                        flags.append(f"{window}:{regime_name}")
            item["regime_robustness"] = {"red_flags": flags}
        pack.append(item)

    payload = {
        "scope": "ADMISSION_DECISION_PACK_V1",
        "purpose": "Consolidated per-candidate evidence for the human admission decision over all READY candidates.",
        "ready_candidate_count": len(pack),
        "market_scope": "SH/SZ",
        "excluded_exchanges": ["BJ"],
        "label": "forward_open_5d:v1",
        "decision_rules": {
            "gates": "signed RankIC > 0.02, signed ICIR >= 0.50, directional day share >= 0.55, annual strict gate >= 4/5 years",
            "redundancy": "median abs Spearman >= 0.90 blocks joint admission; >= 0.80 requires monitoring",
            "authority": "Admission is a human decision; this pack is evidence only.",
        },
        "candidates": pack,
        "database_writes": False,
        "locked_test_read": False,
        "formal_factor_set_changed": False,
        "model_training_started": False,
    }
    out = DIAG / "admission_decision_pack_v1.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} with {len(pack)} candidates")
    missing_monthly = [p["candidate_id"] for p in pack if not p["monthly_stability_2025"]]
    missing_turnover = [p["candidate_id"] for p in pack if not p["rank_turnover"]]
    missing_regime = [p["candidate_id"] for p in pack if not p["regime_robustness"]]
    print("missing monthly:", missing_monthly)
    print("missing turnover:", missing_turnover)
    print("missing regime:", missing_regime)


if __name__ == "__main__":
    main()
