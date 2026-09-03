"""Apply the human admission decisions recorded in the decision sheet.

Reads the decisions from the consolidated decision sheet (or a decisions JSON
override), validates them against the mutual-exclusion / monitor constraints,
computes the frozen factor set, and - only with --apply - applies migrations,
writes the frozen factor-set contract into factor_set_frozen, and updates the
checklist statuses. Default is a read-only dry-run report; nothing is written
unless decisions exist and --apply is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG = REPO_ROOT / "research_v2" / "artifacts" / "diagnostics"
SHEET = REPO_ROOT / "docs" / "handoffs" / "2026-09-03-admission-decision-sheet.md"
TARGET_DATABASE = "dfs://QuantResearchV2Test"
REPAIR_VARIANT_443 = "yeslab_443_reg30_max20_v1"

# Hard mutual-exclusion pairs: both admitted => conflict.
EXCLUSIVE_PAIRS = [
    ("yeslab_114", "yeslab_484"),
    ("yeslab_357", "yeslab_363"),
    ("yeslab_379", "yeslab_375"),
    ("yeslab_370", "yeslab_372"),
    ("yeslab_379", "yeslab_384"),
]
# Monitor pairs: both admitted => warn (needs combination review).
MONITOR_PAIRS = [
    ("yeslab_381", "yeslab_405"),
    ("yeslab_381", "yeslab_443"),
    ("yeslab_398", "yeslab_381"),
    ("yeslab_114", "yeslab_471"),
]

DECISION_ADMIT = {"准入", "admit", "ADMIT", "yes", "Y"}
DECISION_REJECT = {"不入", "reject", "REJECT", "no", "N"}
DECISION_DEFER = {"缓议", "defer", "DEFER", "hold"}


def parse_sheet(sheet_path: Path) -> dict:
    text = sheet_path.read_text(encoding="utf-8")
    factor_decisions = {}
    special = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # table A rows: | idx | yeslab_x | ... | decision |
        if len(cells) >= 12 and re.fullmatch(r"\d+", cells[0]) and cells[1].startswith("yeslab_"):
            fid = cells[1]
            decision = cells[11]  # 决定 column (recommendation is cells[10], not a decision)
            if decision and decision != "☐":
                factor_decisions[fid] = decision
        # special rows: | D1 | ... | my-recommendation | decision |
        if len(cells) >= 5 and re.fullmatch(r"D\d+", cells[0]):
            decision = cells[4]
            if decision and decision != "☐":
                special[cells[0]] = decision
    return {"factors": factor_decisions, "special": special}


def classify(decision: str) -> str:
    if decision in DECISION_ADMIT:
        return "admit"
    if decision in DECISION_REJECT:
        return "reject"
    if decision in DECISION_DEFER:
        return "defer"
    return "unknown"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", type=Path, default=SHEET)
    parser.add_argument("--decisions-json", type=Path, default=None,
                        help="optional JSON override: {\"factors\": {fid: decision}, \"special\": {D1: ...}}")
    parser.add_argument("--apply", action="store_true", help="actually write (default dry-run)")
    parser.add_argument("--frozen-date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--evidence-commit", default="")
    args = parser.parse_args(argv)

    if args.decisions_json:
        decisions = json.loads(args.decisions_json.read_text(encoding="utf-8"))
    else:
        decisions = parse_sheet(args.sheet)

    factor_dec = {fid: classify(d) for fid, d in decisions.get("factors", {}).items()}
    special = decisions.get("special", {})
    admitted = sorted(fid for fid, c in factor_dec.items() if c == "admit")
    rejected = sorted(fid for fid, c in factor_dec.items() if c == "reject")
    deferred = sorted(fid for fid, c in factor_dec.items() if c == "defer")
    unknown = sorted(fid for fid, c in factor_dec.items() if c == "unknown")

    print(f"decisions parsed: {len(factor_dec)} factors "
          f"(admit {len(admitted)}, reject {len(rejected)}, defer {len(deferred)}, unknown {len(unknown)})")
    print(f"special decisions: {special}")
    if not factor_dec:
        print("\nNo decisions recorded yet. Fill the 决定 column in the decision sheet, then re-run.")
        return 0
    if unknown:
        print(f"WARNING: unrecognized decision values for: {unknown}")

    # D1: 443 handling
    d1 = special.get("D1", "")
    frozen = set(admitted)
    if "yeslab_443" in frozen:
        if d1 in ("A", "a"):
            frozen.discard("yeslab_443")
            frozen.add(REPAIR_VARIANT_443)
            print("D1=A: canonical 443 replaced by repair variant in the frozen set.")
        elif d1 in ("B", "b"):
            frozen.discard("yeslab_443")
            print("D1=B: canonical 443 kept MONITORED (not admitted).")
        else:
            print("WARNING: 443 admitted but D1 not decided; excluding 443 pending D1.")
            frozen.discard("yeslab_443")

    # constraint validation
    conflicts, warnings = [], []
    for a, b in EXCLUSIVE_PAIRS:
        if a in frozen and b in frozen:
            conflicts.append(f"EXCLUSIVE pair both admitted: {a} & {b}")
    for a, b in MONITOR_PAIRS:
        if a in frozen and b in frozen:
            warnings.append(f"MONITOR pair both admitted (needs combination review): {a} & {b}")

    print(f"\nFrozen set ({len(frozen)}): " + ", ".join(sorted(frozen)))
    if conflicts:
        print("\nCONFLICTS (must resolve before apply):")
        for c in conflicts:
            print("  -", c)
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print("  -", w)

    if conflicts:
        print("\nAborting: resolve exclusive-pair conflicts first.")
        return 1
    if not args.apply:
        print("\nDry-run only (no writes). Re-run with --apply to freeze.")
        return 0

    # ---- apply ----
    import sys
    sys.path.insert(0, str(REPO_ROOT / "research_v2" / "src"))
    import dolphindb as ddb
    from research_v2.db.migrations import MigrationRunner
    from research_v2.db.session import V2WriteSession

    grading = json.loads((DIAG / "factor_quality_grading_v1" / "factor_quality_grading_v1.json").read_text(encoding="utf-8"))
    grade_by_fid = {e["factor_id"]: e for e in grading["entries"]}
    checklist = json.loads((DIAG / "explicit_admission_review_checklist_v1.json").read_text(encoding="utf-8"))
    cand_by_fid = {e["source_factor_id"]: e.get("candidate_id", "") for e in checklist["checklists"]}

    watermark = hashlib.sha256(json.dumps(
        {"scope": "FACTOR_SET_FROZEN_V1", "frozen_date": args.frozen_date.isoformat(), "factors": sorted(frozen)},
        sort_keys=True).encode()).hexdigest()
    data_version = hashlib.sha256((watermark + str(args.frozen_date)).encode()).hexdigest()
    run_id = str(uuid.uuid4())

    rows = []
    for fid in sorted(frozen):
        g = grade_by_fid.get(fid, {})
        rec = {
            "frozen_date": args.frozen_date,
            "factor_id": fid,
            "factor_version": 1,
            "candidate_id": cand_by_fid.get(fid, fid if fid == REPAIR_VARIANT_443 else ""),
            "decision": "ADMITTED_REPAIR_VARIANT" if fid == REPAIR_VARIANT_443 else "ADMITTED",
            "grade": g.get("grade", ""),
            "monitors_json": json.dumps({"note": "see decision sheet monitors"}, ensure_ascii=False),
            "decision_point": "D1_repair" if fid == REPAIR_VARIANT_443 else "A_table",
            "evidence_commit": args.evidence_commit,
            "run_id": run_id,
            "data_version": data_version,
            "created_at": pd.Timestamp.utcnow().tz_localize(None),
            "source_watermark_id": watermark,
        }
        payload = {k: (v.isoformat() if isinstance(v, (date, pd.Timestamp)) else v) for k, v in rec.items() if k != "record_hash"}
        rec["record_hash"] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True).encode()).hexdigest()
        rows.append(rec)
    frame = pd.DataFrame(rows)

    client = ddb.session()
    client.connect("10.176.55.211", 8848, "xiahongjie", "123456")
    session = V2WriteSession(client, target_database=TARGET_DATABASE)
    MigrationRunner(session, migration_directory=REPO_ROOT / "research_v2" / "migrations").apply()

    existing = session.query(
        f"select count(*) as n from loadTable('{TARGET_DATABASE}','factor_set_frozen') "
        f"where frozen_date=date('{args.frozen_date:%Y.%m.%d}') and data_version='{data_version}'")
    n_existing = int(existing["n"][0]) if isinstance(existing, pd.DataFrame) and len(existing) else 0
    if n_existing:
        print(f"factor_set_frozen: already present ({n_existing} rows) - skipped.")
    else:
        stage = f"factorSetFrozenStage_{uuid.uuid4().hex[:12]}"
        session.upload_frame(stage, frame)
        cols = ", ".join(f"{c} as {c}" for c in frame.columns)
        session.execute(
            f"typed = select {cols} from {stage}\n"
            f"tableInsert{{loadTable('{TARGET_DATABASE}','factor_set_frozen')}}(typed)\n"
            f"undef(`{stage}, VAR)\nundef(`typed, VAR)")
        back = session.query(f"select count(*) as n from loadTable('{TARGET_DATABASE}','factor_set_frozen') where frozen_date=date('{args.frozen_date:%Y.%m.%d}')")
        print(f"factor_set_frozen: wrote {int(back['n'][0])} rows.")

    # artifact
    out = DIAG / "factor_set_frozen_v1.json"
    out.write_text(json.dumps({
        "scope": "FACTOR_SET_FROZEN_V1",
        "frozen_date": args.frozen_date.isoformat(),
        "factors": sorted(frozen),
        "count": len(frozen),
        "special_decisions": special,
        "conflicts": conflicts,
        "warnings": warnings,
        "data_version": data_version,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
