"""Proofread the quantitative claims of the admission decision brief.

Cross-checks every IC/ICIR figure in the brief's regime-window tables
(sections 4m/4n/4q/4r/4s/4t/4u/4v) and the label-validation statistics (4w)
against the source diagnostic artifacts. Reports any discrepancy. Read-only.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG = REPO_ROOT / "research_v2" / "artifacts" / "diagnostics"
BRIEF = REPO_ROOT / "docs" / "handoffs" / "2026-09-01-admission-decision-brief.md"

FAILURES: list[str] = []
CHECKS = 0


def check(label: str, brief_val: object, artifact_val: object, tol: float = 0.0011) -> None:
    global CHECKS
    CHECKS += 1
    if brief_val is None or artifact_val is None:
        if brief_val != artifact_val:
            FAILURES.append(f"{label}: brief={brief_val} artifact={artifact_val}")
        return
    try:
        if abs(float(brief_val) - float(artifact_val)) > tol:
            FAILURES.append(f"{label}: brief={brief_val} artifact={artifact_val}")
    except (TypeError, ValueError):
        if str(brief_val) != str(artifact_val):
            FAILURES.append(f"{label}: brief={brief_val!r} artifact={artifact_val!r}")


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def section(text: str, header: str) -> str:
    start = text.index(header)
    nxt = text.find("\n## ", start + len(header))
    return text[start:nxt if nxt != -1 else len(text)]


def parse_table_rows(sec: str) -> list[list[str]]:
    rows = []
    for line in sec.splitlines():
        if line.startswith("|") and "---" not in line:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows.append(cells)
    return rows


def strip_md(s: str) -> str:
    return s.replace("**", "").strip()


def cell_float(s: str):
    s = strip_md(s).split("（")[0].strip()
    if s in ("—", "-", ""):
        return None
    return float(s)


def main() -> int:
    text = BRIEF.read_text(encoding="utf-8")

    # ---------- 4m: 2019 window (18 candidates), full/liquid ----------
    baseline = load_json(DIAG / "historical_window_2019_baseline_brief_table.json")
    comp19 = load_json(DIAG / "historical_window_2019_robustness_v1" / "historical_window_2019_comparison.json")
    rows19 = {r["factor_id"]: r for r in comp19["rows"]}
    sec = section(text, "## 4m.")
    for row in parse_table_rows(sec)[1:]:  # skip header
        fid = row[0]
        if not fid.startswith("yeslab_"):
            continue
        b = baseline.get(fid, {})
        r = rows19[fid]
        check(f"4m {fid} train", cell_float(row[1]), b.get("train_icir"))
        check(f"4m {fid} val", cell_float(row[2]), b.get("val_icir"))
        check(f"4m {fid} 2019full", cell_float(row[3]), r["icir_full_market"])
        check(f"4m {fid} 2019liq", cell_float(row[4]), r["icir_adv20_liquid"])

    # ---------- 4n: 2018 window table ----------
    comp18 = load_json(DIAG / "historical_window_2018_robustness_v1" / "historical_window_2018_comparison.json")
    rows18 = {r["factor_id"]: r for r in comp18["rows"]}
    sec = section(text, "## 4n.")
    for row in parse_table_rows(sec)[1:]:
        fid = row[0]
        if not fid.startswith("yeslab_"):
            continue
        r18, r19 = rows18[fid], rows19[fid]
        check(f"4n {fid} train", cell_float(row[1]), baseline.get(fid, {}).get("train_icir"))
        check(f"4n {fid} 2019full", cell_float(row[2]), r19["icir_full_market"])
        check(f"4n {fid} 2018full", cell_float(row[3]), r18["icir_full_market"])
        check(f"4n {fid} 2018liq", cell_float(row[4]), r18["icir_adv20_liquid"])

    # ---------- 4q: official universe upgrade summary ----------
    sec = section(text, "## 4q.")
    # 2019: 23/23 PASS, min official = 371 0.530
    off19 = {fid: r["icir_official_cn_liquid_a"] for fid, r in rows19.items()}
    n_pass19 = sum(1 for fid, r in rows19.items() if r["classify_official"] == "PASS")
    if n_pass19 != 23:
        FAILURES.append(f"4q 2019 official pass count: brief=23 artifact={n_pass19}")
    CHECKS_inc = True
    min19 = min(off19.values())
    m = re.search(r"2019 \| \*\*23/23 PASS\*\* \| 371（ICIR (\d\.\d+)", sec)
    if not m:
        FAILURES.append("4q 2019 line not found")
    else:
        check("4q 2019 min official", float(m.group(1)), min19)
    min18 = sorted(((r["icir_official_cn_liquid_a"], fid) for fid, r in rows18.items()))[:2]
    m_2018 = re.search(r"\| 2018 \| 21 PASS / 2 WATCH \| 443（ICIR (0\.\d+)）、425（ICIR (0\.\d+)）", sec)
    if not m_2018:
        FAILURES.append("4q 2018 table row not found")
    else:
        check("4q 2018 443 official", float(m_2018.group(1)), rows18["yeslab_443"]["icir_official_cn_liquid_a"])
        check("4q 2018 425 official", float(m_2018.group(2)), rows18["yeslab_425"]["icir_official_cn_liquid_a"])

    # ---------- 4r: 2017 window (inline lists) ----------
    comp17 = load_json(DIAG / "historical_window_2017_robustness_v1" / "historical_window_2017_comparison.json")
    rows17 = {r["factor_id"]: r for r in comp17["rows"]}
    sec = section(text, "## 4r.")
    for m in re.finditer(r"(?:yeslab_)?(\d{2,3}|206m)（(0\.\d+)）", sec):
        raw, val = m.group(1), float(m.group(2))
        fid = "yeslab_206_median_abs_v2" if raw == "206m" else f"yeslab_{raw}"
        if fid in rows17:
            check(f"4r {fid} 2017off", val, rows17[fid]["icir_official_cn_liquid_a"])
    # 443 collapse numbers
    r443 = rows17["yeslab_443"]
    m = re.search(r"443：官方 (0\.\d+) / liquid (0\.\d+) / full (0\.\d+)", sec)
    if m:
        check("4r 443 official", float(m.group(1)), r443["icir_official_cn_liquid_a"])
        check("4r 443 liquid", float(m.group(2)), r443["icir_adv20_liquid"])
        check("4r 443 full", float(m.group(3)), r443["icir_full_market"])

    # ---------- 4s: repair table ----------
    repair = load_json(DIAG / "regime_repair_443_v1" / "alpha443_repair_summary.json")
    ev = repair["evidence_official_universe_signed_icir"]
    sec = section(text, "## 4s.")
    expect = {
        "2017": (ev["2017"]["canonical"], ev["2017"]["repair"]),
        "2018": (ev["2018"]["canonical"], ev["2018"]["repair"]),
        "2019": (ev["2019"]["canonical"], ev["2019"]["repair"]),
        "train_2020_2024": (ev["train_2020_2024_aggregate"]["canonical"], ev["train_2020_2024_aggregate"]["repair"]),
        "validation_2025": (ev["validation_2025"]["canonical"], ev["validation_2025"]["repair"]),
    }
    for row in parse_table_rows(sec):
        key = row[0]
        if key in expect:
            check(f"4s {key} canonical", cell_float(row[1]), expect[key][0])
            check(f"4s {key} repair", cell_float(row[2]), expect[key][1])

    # ---------- 4u: formal nine table ----------
    f9 = load_json(DIAG / "formal9_pretraining_windows_v1" / "2017" / "historical_window_2017_robustness.json")
    f9_18 = load_json(DIAG / "formal9_pretraining_windows_v1" / "2018" / "historical_window_2018_robustness.json")
    f9_19 = load_json(DIAG / "formal9_pretraining_windows_v1" / "2019" / "historical_window_2019_robustness.json")
    status_idx = load_json(DIAG / "formal_nine_current_status_index_v1.json")
    dirs = {"yeslab_376": -1, "yeslab_380": -1, "yeslab_396": -1, "yeslab_431": -1, "yeslab_437": -1,
            "yeslab_155": -1, "yeslab_94": -1, "yeslab_98": -1, "yeslab_442": 1}
    sec = section(text, "## 4u.")
    for row in parse_table_rows(sec)[1:]:
        fid = row[0]
        if not fid.startswith("yeslab_"):
            continue
        d = dirs[fid]
        st = status_idx["factors"][fid]
        tr = d * st["train_icir"]
        va = d * st["validation_icir"]
        check(f"4u {fid} train", cell_float(row[1]), tr)
        check(f"4u {fid} val", cell_float(row[2]), va)
        check(f"4u {fid} 2017off", cell_float(row[3]), f9["candidates"][fid]["official_cn_liquid_a"]["icir"])
        check(f"4u {fid} 2018off", cell_float(row[4]), f9_18["candidates"][fid]["official_cn_liquid_a"]["icir"])
        check(f"4u {fid} 2019off", cell_float(row[5]), f9_19["candidates"][fid]["official_cn_liquid_a"]["icir"])

    # ---------- 4v: formal-formal cross-period table ----------
    v2 = {y: load_json(DIAG / "pretraining_regime_redundancy_v2" / f"pretraining_redundancy_{y}.json") for y in (2017, 2018, 2019)}
    tv = load_json(DIAG / "formal9_regime_redundancy_train_val_v1" / "pretraining_redundancy_2020.json")
    vv = load_json(DIAG / "formal9_regime_redundancy_train_val_v1" / "pretraining_redundancy_2025.json")

    def pair_val(payload, a, b):
        for p in payload["all_pairs"]:
            if {p["factor_id_left"], p["factor_id_right"]} == {a, b}:
                return p["median_abs_spearman"]
        return None

    FORMAL_SET = {"yeslab_380", "yeslab_431", "yeslab_437", "yeslab_155", "yeslab_98", "yeslab_94", "yeslab_376", "yeslab_396", "yeslab_442"}
    sec = section(text, "## 4v.")
    for row in parse_table_rows(sec):
        if "×" not in row[0]:
            continue
        a, b = [x.strip() for x in row[0].split("×")]
        if a in FORMAL_SET and b in FORMAL_SET:
            check(f"4v {a}x{b} train", cell_float(row[1]), pair_val(tv, a, b))
            check(f"4v {a}x{b} val", cell_float(row[2]), pair_val(vv, a, b))
            check(f"4v {a}x{b} 2017", cell_float(row[3]), pair_val(v2[2017], a, b))
            check(f"4v {a}x{b} 2018", cell_float(row[4]), pair_val(v2[2018], a, b))
            check(f"4v {a}x{b} 2019", cell_float(row[5]), pair_val(v2[2019], a, b))
        elif (a in FORMAL_SET) != (b in FORMAL_SET):
            # candidate x formal table: columns pair | 2017 | 2018 | 2019 | train nine-edge | note
            check(f"4v {a}x{b} 2017", cell_float(row[1]), pair_val(v2[2017], a, b))
            check(f"4v {a}x{b} 2018", cell_float(row[2]), pair_val(v2[2018], a, b))
            check(f"4v {a}x{b} 2019", cell_float(row[3]), pair_val(v2[2019], a, b))

    # ---------- 4w: label validation statistics ----------
    v2018_full = load_json(DIAG / "historical_label_replica_2018_official_validation_v1" / "validation_result.json")
    v2018 = v2018_full.get("validation", v2018_full)
    sec = section(text, "## 4w.")
    m = re.search(r"(\d{6}) 行合并、双有效 (\d{6}) 行", sec)
    if m:
        check("4w 2018 merged rows", int(m.group(1)), v2018["merged_rows"])
        check("4w 2018 both-valid rows", int(m.group(2)), v2018["both_valid_rows"])
    v2017_full = load_json(DIAG / "historical_label_replica_2017_official_validation_v1" / "validation_result.json")
    v2017 = v2017_full.get("validation", v2017_full)
    sec4r = section(text, "## 4r.")
    m = re.search(r"双有效 (\d{6}) 行", sec4r)
    if m:
        check("4r 2017 both-valid rows", int(m.group(1)), v2017["both_valid_rows"])
    m = re.search(r"(\d{4}) 行分歧", sec4r)
    if m:
        disagree = v2017["merged_rows"] - round(v2017["label_valid_agreement"] * v2017["merged_rows"])
        check("4r 2017 disagreement rows", int(m.group(1)), disagree)

    # ---------- Section 3: main candidate table ----------
    ver = load_json(DIAG / "admission_evidence_verification_v1.json")
    ver_by_fid = {c["source_factor_id"]: c for c in ver["candidates"]}
    nearmiss = load_json(DIAG / "nearmiss6_train_evidence_v1.json")["candidates"]
    dq3 = load_json(DIAG / "dq3_train_evidence_v1.json")["candidates"]
    gp4 = load_json(DIAG / "gp_coverage_repair4_train_evidence_v1.json")["candidates"]
    pass7 = load_json(DIAG / "pass7_train_evidence_v1.json")["candidates"]
    directions = {"yeslab_471": 1}

    def val2025_from_csv(fid: str):
        import pandas as pd
        folder = DIAG / f"{fid}_validation_2025_monthly_v1"
        path = folder / "daily_rank_ic.csv"
        if not path.is_file():
            return None, None
        df = pd.read_csv(path)
        col = [c for c in df.columns if "ic" in c.lower()][0]
        ic = df[col].dropna()
        dr = directions.get(fid, -1)
        return float(dr * ic.mean()), float(dr * ic.mean() / ic.std()) if ic.std() > 0 else None

    sec = section(text, "## 3. \u5341\u4e2a READY \u5019\u9009\u4e00\u89c8")
    for row in parse_table_rows(sec)[1:]:
        fid = row[0]
        if not fid.startswith("yeslab_"):
            continue
        train_cell, val_cell, nine_cell = row[2], row[3], row[5]
        t_ic = cell_float(train_cell.split("/")[0]) if "/" in train_cell else None
        t_icir = cell_float(train_cell.split("/")[1]) if "/" in train_cell else None
        v_parts = val_cell.split("/")
        v_ic, v_icir = cell_float(v_parts[0]), cell_float(v_parts[1]) if len(v_parts) > 1 else None
        nine_parts = nine_cell.split("/")
        n_tr = cell_float(nine_parts[0]) if nine_parts else None
        n_va = cell_float(nine_parts[1]) if len(nine_parts) > 1 else None

        ver_fid = "yeslab_206" if fid == "yeslab_206_median_abs_v2" else fid
        if ver_fid in ver_by_fid:
            c = ver_by_fid[ver_fid]
            check(f"S3 {fid} train IC", t_ic, c["train"]["signed_mean_rank_ic"])
            check(f"S3 {fid} train ICIR", t_icir, c["train"]["signed_icir"])
            check(f"S3 {fid} val IC", v_ic, c["validation"]["signed_mean_rank_ic"])
            check(f"S3 {fid} val ICIR", v_icir, c["validation"]["signed_icir"])
            check(f"S3 {fid} nine train", n_tr, c["formal_nine_train"]["maximum_median_abs_spearman"])
            check(f"S3 {fid} nine val", n_va, c["formal_nine_validation"]["maximum_median_abs_spearman"])
        elif fid in nearmiss:
            c = nearmiss[fid]
            check(f"S3 {fid} train IC", t_ic, c.get("signed_ic"))
            check(f"S3 {fid} train ICIR", t_icir, c.get("signed_icir"))
            check(f"S3 {fid} nine train", n_tr, c.get("nine_edge_max"))
            csv_ic, csv_icir = val2025_from_csv(fid)
            check(f"S3 {fid} val IC", v_ic, csv_ic)
            check(f"S3 {fid} val ICIR", v_icir, csv_icir)
        elif fid in dq3:
            c = dq3[fid]
            check(f"S3 {fid} train IC", t_ic, c.get("signed_ic"))
            check(f"S3 {fid} train ICIR", t_icir, c.get("signed_icir"))
            check(f"S3 {fid} nine train", n_tr, c.get("nine_edge_max"))
            csv_ic, csv_icir = val2025_from_csv(fid)
            check(f"S3 {fid} val IC", v_ic, csv_ic)
            check(f"S3 {fid} val ICIR", v_icir, csv_icir)
        elif fid in gp4:
            c = gp4[fid]
            check(f"S3 {fid} train IC", t_ic, c["training"]["signed_mean_rank_ic"])
            check(f"S3 {fid} train ICIR", t_icir, c["training"]["signed_icir"])
            check(f"S3 {fid} nine train", n_tr, c.get("nine_edge_train", {}).get("maximum_median_abs_spearman"))
            csv_ic, csv_icir = val2025_from_csv(fid)
            check(f"S3 {fid} val IC", v_ic, csv_ic)
            check(f"S3 {fid} val ICIR", v_icir, csv_icir)
        elif fid in pass7:
            c = pass7[fid]
            check(f"S3 {fid} train ICIR", t_icir, c.get("train_signed_icir"))
            check(f"S3 {fid} val ICIR", v_icir, c.get("validation_signed_icir"))
            check(f"S3 {fid} nine val", n_va, c.get("nine_edge_val_max"))
            csv_ic, csv_icir = val2025_from_csv(fid)
            check(f"S3 {fid} val IC", v_ic, csv_ic)
        elif fid == "yeslab_395":
            c = load_json(DIAG / "tushare_alpha395_precedent_upgrade_review_v1.json")
            check(f"S3 {fid} train IC", t_ic, c["training_2020_2024"]["signed_mean_rank_ic"])
            check(f"S3 {fid} train ICIR", t_icir, c["training_2020_2024"]["signed_icir"])
            check(f"S3 {fid} val IC", v_ic, c["validation_2025"]["signed_mean_rank_ic"])
            check(f"S3 {fid} val ICIR", v_icir, c["validation_2025"]["signed_icir"])
            check(f"S3 {fid} nine train", n_tr, c["similarity_to_formal_nine_train"]["maximum_median_abs_spearman"])
            if "similarity_to_formal_nine_validation" in c:
                check(f"S3 {fid} nine val", n_va, c["similarity_to_formal_nine_validation"]["maximum_median_abs_spearman"])
        elif fid == "yeslab_471":
            csv_ic, csv_icir = val2025_from_csv(fid)
            check(f"S3 {fid} val IC", v_ic, csv_ic)
            check(f"S3 {fid} val ICIR", v_icir, csv_icir)
        else:
            FAILURES.append(f"S3 {fid}: no source mapping")

    # ---------- 4i: vol4 rejection table ----------
    vol4 = load_json(DIAG / "vol4_train_evidence_v1.json")["candidates"]
    sec = section(text, "## 4i.")
    vol4_ids = {"252": "yeslab_252", "260": "yeslab_260", "271": "yeslab_271", "272": "yeslab_272"}
    for row in parse_table_rows(sec)[1:]:
        head = row[0]
        for short, fid in vol4_ids.items():
            if head.startswith(short):
                check(f"4i {fid} train ICIR", cell_float(row[1]), vol4[fid]["train_signed_icir"])

    # ---------- 4k: alpha167 window scan table ----------
    scan167_full = load_json(DIAG / "alpha167_window_variant_exploration_v1" / "window_variant_exploration.json")
    scan167 = scan167_full.get("windows") or scan167_full.get("results")
    sec = section(text, "## 4k.")
    for row in parse_table_rows(sec)[1:]:
        win = row[0].split("（")[0].strip()
        key = f"window_{win}"
        if key not in scan167:
            continue
        w = scan167[key]
        check(f"4k window {win} train IC", cell_float(row[1]), w["train_signed_mean_rank_ic"])
        check(f"4k window {win} ICIR", cell_float(strip_md(row[2])), w["train_signed_icir"])
        check(f"4k window {win} annual", int(row[3].split("/")[0]), w["strict_annual_icir_pass_years"])

    # ---------- 4l: GP_023 near-threshold closure ----------
    q392 = load_json(DIAG / "tushare_shsz_gp023_formal9_train_2020_2024_v1" / "quality_summary.json")
    s392 = [e for e in q392["summaries"] if e.get("factor_id") == "yeslab_392"]
    if s392:
        sec = section(text, "## 4l.")
        m = re.search(r"ICIR (0\.\d+)（<0\.50）", sec)
        if m:
            check("4l 392 train ICIR", float(m.group(1)), abs(s392[0]["icir"]))
        replay392 = load_json(DIAG / "formula_replay_gp023_tushare_2025-01_v1.json")["rows"][0]
        m = re.search(r"（(\d+)/(\d+)，([\d.e\-]+)）", sec)
        if m:
            check("4l 392 mismatch rows", int(m.group(1)), replay392["mismatch_rows"])
            check("4l 392 overlap rows", int(m.group(2)), replay392["overlap_finite_rows"])

    print(f"checks run: {CHECKS}")
    if FAILURES:
        print(f"DISCREPANCIES ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL BRIEF NUMBERS VERIFIED against source artifacts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
