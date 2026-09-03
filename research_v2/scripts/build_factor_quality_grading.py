"""Factor quality grading for all factors with complete evidence.

Grades the 9 formal factors and 23 READY candidates on four transparent
dimensions, each scored 0-3:

  1. train_predictive  - signed training ICIR (2020-2024)
     3: >=0.65 | 2: >=0.55 | 1: >=0.50 | 0: <0.50 (fails project gate)
  2. oos_validation    - signed validation ICIR (2025)
     3: >=0.80 | 2: >=0.65 | 1: >=0.50 | 0: <0.50
  3. regime_robustness - minimum official-universe ICIR across the three
     pre-training years 2017/2018/2019
     3: >=0.55 | 2: >=0.45 | 1: >=0.30 | 0: <0.30 (collapse)
  4. uniqueness        - max nine-edge median abs Spearman (train/validation)
     3: <0.70 | 2: <0.80 | 1: <0.90 | 0: >=0.90

Grade from the 0-12 sum: S>=11, A 9-10, B 7-8, C 5-6, D <=4.
Hard overrides: regime collapse (dimension 3 = 0) caps at D; a failed core
gate (train ICIR < 0.50) caps at D. Read-only; no database writes.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG = REPO_ROOT / "research_v2" / "artifacts" / "diagnostics"
BRIEF = REPO_ROOT / "docs" / "handoffs" / "2026-09-01-admission-decision-brief.md"


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def section(text: str, header: str) -> str:
    start = text.index(header)
    nxt = text.find("\n## ", start + len(header))
    return text[start:nxt if nxt != -1 else len(text)]


def parse_table_rows(sec: str):
    rows = []
    for line in sec.splitlines():
        if line.startswith("|") and "---" not in line:
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows


def score_train(v):
    if v is None:
        return None
    return 3 if v >= 0.65 else 2 if v >= 0.55 else 1 if v >= 0.50 else 0


def score_oos(v):
    if v is None:
        return None
    return 3 if v >= 0.80 else 2 if v >= 0.65 else 1 if v >= 0.50 else 0


def score_regime(v):
    if v is None:
        return None
    return 3 if v >= 0.55 else 2 if v >= 0.45 else 1 if v >= 0.30 else 0


def score_unique(edge):
    if edge is None:
        return None
    return 3 if edge < 0.70 else 2 if edge < 0.80 else 1 if edge < 0.90 else 0


def grade_of(scores):
    if any(s is None for s in scores.values()):
        return "INCOMPLETE"
    total = sum(scores.values())
    if scores["regime_robustness"] == 0 or scores["train_predictive"] == 0:
        return "D"
    return "S" if total >= 11 else "A" if total >= 9 else "B" if total >= 7 else "C" if total >= 5 else "D"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    text = BRIEF.read_text(encoding="utf-8")

    # Pre-training official ICIR for READY candidates
    pretrain = {}
    for year in (2017, 2018, 2019):
        comp = load_json(DIAG / f"historical_window_{year}_robustness_v1" / f"historical_window_{year}_comparison.json")
        for r in comp["rows"]:
            pretrain.setdefault(r["factor_id"], {})[year] = r.get("icir_official_cn_liquid_a")

    # Pre-training official ICIR for formal nine
    formal_pretrain = {}
    for year in (2017, 2018, 2019):
        f9 = load_json(DIAG / "formal9_pretraining_windows_v1" / str(year) / f"historical_window_{year}_robustness.json")
        for fid, c in f9["candidates"].items():
            formal_pretrain.setdefault(fid, {})[year] = c.get("official_cn_liquid_a", {}).get("icir")

    entries = []

    # ---- READY candidates from brief section 3 ----
    sec = section(text, "## 3. \u5341\u4e2a READY \u5019\u9009\u4e00\u89c8")
    for row in parse_table_rows(sec)[1:]:
        fid = row[0]
        if not fid.startswith("yeslab_"):
            continue
        train_cell, val_cell, nine_cell = row[2], row[3], row[5]
        t_icir = float(train_cell.split("/")[1]) if "/" in train_cell else None
        if fid == "yeslab_471" and t_icir is None:
            t_icir = 0.70  # documented annual range 0.70 -> 1.01; conservative lower bound
        v_icir = float(val_cell.split("/")[1]) if "/" in val_cell else None
        nine_parts = nine_cell.split("/")
        edges = []
        for np_ in nine_parts[:2]:
            try:
                edges.append(float(np_.strip().replace("**", "")))
            except ValueError:
                pass
        max_edge = max(edges) if edges else None
        pt = pretrain.get(fid, {})
        regime_min = min((v for v in pt.values() if v is not None), default=None) if pt else None
        scores = {
            "train_predictive": score_train(t_icir),
            "oos_validation": score_oos(v_icir),
            "regime_robustness": score_regime(regime_min),
            "uniqueness": score_unique(max_edge),
        }
        entries.append({
            "factor_id": fid,
            "group": "READY_CANDIDATE",
            "train_icir": t_icir,
            "validation_icir": v_icir,
            "pretrain_official_icir": pt,
            "regime_min_official_icir": regime_min,
            "max_nine_edge": max_edge,
            "scores": scores,
            "total": sum(s for s in scores.values() if s is not None) if all(s is not None for s in scores.values()) else None,
            "grade": grade_of(scores),
        })

    # ---- formal nine from brief 4u + artifacts ----
    f9_status = load_json(DIAG / "formal_nine_current_status_index_v1.json")
    dirs = {"yeslab_376": -1, "yeslab_380": -1, "yeslab_396": -1, "yeslab_431": -1, "yeslab_437": -1,
            "yeslab_155": -1, "yeslab_94": -1, "yeslab_98": -1, "yeslab_442": 1}
    ver = load_json(DIAG / "admission_evidence_verification_v1.json")
    # nine-edges for formal factors vs the rest of the formal set: use 4v cross-period train matrix
    tv = load_json(DIAG / "formal9_regime_redundancy_train_val_v1" / "pretraining_redundancy_2020.json")
    FORMAL = set(dirs)

    def pair_val(payload, a, b):
        for p in payload["all_pairs"]:
            if {p["factor_id_left"], p["factor_id_right"]} == {a, b}:
                return p["median_abs_spearman"]
        return None

    for fid, d_ in dirs.items():
        st = f9_status["factors"][fid]
        t_icir = d_ * st["train_icir"]
        v_icir = d_ * st["validation_icir"]
        edges = [pair_val(tv, fid, o) for o in FORMAL if o != fid]
        edges = [e for e in edges if e is not None]
        max_edge = max(edges) if edges else None
        pt = formal_pretrain.get(fid, {})
        regime_min = min((v for v in pt.values() if v is not None), default=None) if pt else None
        scores = {
            "train_predictive": score_train(t_icir),
            "oos_validation": score_oos(v_icir),
            "regime_robustness": score_regime(regime_min),
            "uniqueness": score_unique(max_edge),
        }
        entries.append({
            "factor_id": fid,
            "group": "FORMAL",
            "train_icir": round(t_icir, 4),
            "validation_icir": round(v_icir, 4),
            "pretrain_official_icir": pt,
            "regime_min_official_icir": regime_min,
            "max_intra_formal_edge_train": max_edge,
            "scores": scores,
            "total": sum(s for s in scores.values() if s is not None) if all(s is not None for s in scores.values()) else None,
            "grade": grade_of(scores),
        })

    # ---- five pass7 candidates (documented in 4m/4n tables, not section 3) ----
    pass7_ev = load_json(DIAG / "pass7_train_evidence_v1.json")["candidates"]
    sim_train = load_json(DIAG / "tushare_shsz_pass7_formal9_train_2020_2024_v1" / "similarity_summary.json")
    sim_val = load_json(DIAG / "tushare_shsz_pass7_formal9_validation_2025_v1" / "similarity_summary.json")
    FORMAL_IDS = set(dirs)

    def max_formal_edge(pairs, fid):
        vals = [
            pr["median_abs_spearman"]
            for pr in pairs
            if fid in (pr["factor_id_left"], pr["factor_id_right"])
            and (set((pr["factor_id_left"], pr["factor_id_right"])) - {fid}) <= FORMAL_IDS
        ]
        return max(vals) if vals else None

    for fid in ("yeslab_371", "yeslab_372", "yeslab_379", "yeslab_384", "yeslab_406"):
        ev = pass7_ev[fid]
        t_icir = ev.get("train_signed_icir")
        v_icir = ev.get("validation_signed_icir")
        max_edge_candidates = [max_formal_edge(sim_train, fid), max_formal_edge(sim_val, fid)]
        max_edge = max((e for e in max_edge_candidates if e is not None), default=None)
        pt = pretrain.get(fid, {})
        regime_min = min((v for v in pt.values() if v is not None), default=None) if pt else None
        scores = {
            "train_predictive": score_train(t_icir),
            "oos_validation": score_oos(v_icir),
            "regime_robustness": score_regime(regime_min),
            "uniqueness": score_unique(max_edge),
        }
        entries.append({
            "factor_id": fid,
            "group": "READY_CANDIDATE",
            "train_icir": t_icir,
            "validation_icir": v_icir,
            "pretrain_official_icir": pt,
            "regime_min_official_icir": regime_min,
            "max_nine_edge": max_edge,
            "scores": scores,
            "total": sum(s for s in scores.values() if s is not None) if all(s is not None for s in scores.values()) else None,
            "grade": grade_of(scores),
        })

    order = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4, "INCOMPLETE": 5}
    entries.sort(key=lambda e: (order[e["grade"]], -(e["total"] if e["total"] is not None else -1)))

    print(f"{'grade':5s} {'factor':28s} {'group':15s} {'trnICIR':>8s} {'valICIR':>8s} {'regimeMin':>9s} {'edge':>6s} {'total':>5s}")
    for e in entries:
        fmt = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "  -  "
        print(f"{e['grade']:5s} {e['factor_id']:28s} {e['group']:15s} {fmt(e['train_icir']):>8s} {fmt(e['validation_icir']):>8s} {fmt(e['regime_min_official_icir']):>9s} {fmt(e.get('max_nine_edge') or e.get('max_intra_formal_edge_train')):>6s} {str(e['total']):>5s}")

    counts = {}
    for e in entries:
        counts[e["grade"]] = counts.get(e["grade"], 0) + 1
    payload = {
        "scope": "FACTOR_QUALITY_GRADING_V1",
        "method": "four dimensions (train predictive power, OOS validation, pre-training regime robustness, uniqueness), each 0-3; grade S>=11, A 9-10, B 7-8, C 5-6, D <=4; hard overrides: regime collapse or failed train gate caps at D",
        "dimension_thresholds": {
            "train_predictive": {"3": ">=0.65", "2": ">=0.55", "1": ">=0.50", "0": "<0.50"},
            "oos_validation": {"3": ">=0.80", "2": ">=0.65", "1": ">=0.50", "0": "<0.50"},
            "regime_robustness": {"3": ">=0.55", "2": ">=0.45", "1": ">=0.30", "0": "<0.30 collapse"},
            "uniqueness": {"3": "edge<0.70", "2": "<0.80", "1": "<0.90", "0": ">=0.90"},
        },
        "graded_count": len(entries),
        "grade_counts": counts,
        "entries": entries,
        "notes": [
            "yeslab_471 train ICIR uses the documented annual-range lower bound 0.70 (no aggregate in its evidence).",
            "formal factors' uniqueness dimension uses the max intra-formal-set edge (their admission predated the pool audit).",
            "grading is decision support only; admission gates and constraints in the checklist are unchanged.",
        ],
        "database_writes": False,
        "locked_test_read": False,
    }
    out = args.output / "factor_quality_grading_v1.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\ngrade counts: {counts}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
