"""Is FULL_CROSS_SECTION_FACTOR_IDS complete? Scan the calculators instead of trusting the list.

audit_formula_replay_tushare.py:1024-1036 decides the replay panel like this: codes default to the
securities appearing in the factor's OWN upstream observations, and are only widened to the full market
when --full-market is passed AND requires_full_cross_section(factor_id) is true. requires_full_cross_section
is just an intersection with FULL_CROSS_SECTION_FACTOR_IDS, a hand-maintained frozenset of 34 ids.

A factor whose calculator applies a global cross-sectional operator -- cs_zscore, cs_rank, cs_max, or any
axis=1 reduction over a date-by-code frame -- has a value for every security that depends on WHICH OTHER
securities are in the panel. Auditing such a factor on its own observation codes rather than the full
market silently computes a different quantity. The set is therefore a safety registry, and an omission is
a silent under-scoping of any audit run for that factor.

This scans the calculator source of every REPLAY_DEFINITIONS entry for those operators and cross-tabulates
against the registry. It reports factors that HAVE global cross-sectional operators but are NOT registered.
It does not claim the registry is wrong where it registers a factor whose operators this scan missed --
the scan is a regex over source, and it says so per factor.

Note on impact: materialize_frozen_factor_daily.py does NOT consult requires_full_cross_section; it always
calls fetch_full_tushare_shsz_market. So published factor_daily values are unaffected. The exposure is in
the audit tool.

SAFETY: reads repository and calculator source only. No database, no weights, no writes except --out.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
CALC_ROOT = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/数据库/factor_calculators")

# a global cross-sectional operator on a date-by-code frame: any axis=1 reduction, or the named cs_*
# helpers used by the genetic-programming calculators, or an industry groupby.
PATTERNS = {
    "axis1_rank": r"\.rank\([^)]*axis\s*=\s*1",
    "axis1_mean": r"\.mean\([^)]*axis\s*=\s*1",
    "axis1_std": r"\.std\([^)]*axis\s*=\s*1",
    "axis1_max": r"\.max\([^)]*axis\s*=\s*1",
    "axis1_min": r"\.min\([^)]*axis\s*=\s*1",
    "axis1_median": r"\.median\([^)]*axis\s*=\s*1",
    "axis1_sum": r"\.sum\([^)]*axis\s*=\s*1",
    "cs_helper": r"\bcs_(?:rank|zscore|max|min|mean|std)\b",
    "industry_groupby": r"groupby\(\s*['\"]?industry|groupby\(\s*industry_map|groupby\(\s*groups",
    "transform_group": r"\.transform\(\s*['\"](?:mean|std|skew)['\"]\s*\)",
}


def scan(path: Path) -> dict:
    if not path.is_file():
        return {"file_found": False}
    text = path.read_text(encoding="utf-8")
    hits = {}
    for name, pat in PATTERNS.items():
        found = re.findall(pat, text)
        if found:
            hits[name] = len(found)
    return {"file_found": True, "lines": text.count("\n") + 1, "operators": hits,
            "operator_count": sum(hits.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    import audit_formula_replay_tushare as audit
    RD = audit.replay.REPLAY_DEFINITIONS
    registry = set(audit.FULL_CROSS_SECTION_FACTOR_IDS)

    dense9 = ["yeslab_206", "yeslab_357", "yeslab_381", "yeslab_425", "yeslab_443_reg30_max20_v1",
              "yeslab_464", "yeslab_471", "yeslab_484", "yeslab_96"]
    sparse6 = ["yeslab_372", "yeslab_375", "yeslab_384", "yeslab_389", "yeslab_390", "yeslab_406"]
    v2 = sorted(set(dense9) | set(sparse6))

    rows = {}
    for fid, definition in sorted(RD.items()):
        module = getattr(definition, "calculator_module", None)
        if not module:
            continue
        found = scan(CALC_ROOT / f"{module}.py")
        # calculators that subclass a shared core do their cross-sectional work in the base file
        base_hits = {}
        if found.get("file_found") and found["operator_count"] == 0:
            text = (CALC_ROOT / f"{module}.py").read_text(encoding="utf-8")
            m = re.search(r"class\s+\w+\((\w+)\)", text)
            if m:
                base = scan(CALC_ROOT / f"{m.group(1)}.py")
                if base.get("file_found"):
                    base_hits = {"base_class": m.group(1), **base}
        has_ops = bool(found.get("operator_count")) or bool(base_hits.get("operator_count"))
        rows[fid] = {
            "calculator_module": module,
            "factor_name": getattr(definition, "factor_name", None),
            "in_registry": fid in registry,
            "own_file_operators": found.get("operators", {}),
            "own_file_operator_count": found.get("operator_count", 0),
            "base_class_scan": base_hits or None,
            "has_global_cs_operators": has_ops,
            "in_frozen_v2": fid in v2,
            "frozen_v2_role": ("dense" if fid in dense9 else "sparse") if fid in v2 else None,
            "audit_would_under_scope": bool(has_ops and fid not in registry),
        }

    under = sorted(f for f, v in rows.items() if v["audit_would_under_scope"])
    registered_no_ops = sorted(f for f, v in rows.items() if v["in_registry"] and not v["has_global_cs_operators"])
    out = {
        "question": "Is FULL_CROSS_SECTION_FACTOR_IDS complete with respect to what the calculators actually do?",
        "registry_file": "research_v2/scripts/audit_formula_replay_tushare.py:120",
        "registry_size": len(registry),
        "where_the_registry_is_used": "audit_formula_replay_tushare.py:1030-1036 -- codes are widened to the "
                                      "full market only if --full-market AND requires_full_cross_section()",
        "impact_scope": "the AUDIT tool only. materialize_frozen_factor_daily.py does not consult "
                        "requires_full_cross_section and always fetches the full SH/SZ market, so published "
                        "factor_daily values are unaffected.",
        "replay_definitions_scanned": len(rows),
        "operator_patterns": PATTERNS,
        "factors_with_cs_operators_but_NOT_registered": under,
        "count_with_cs_operators_but_not_registered": len(under),
        "of_those_in_frozen_v2": [f for f in under if rows[f]["in_frozen_v2"]],
        "registered_but_no_operators_found_by_this_scan": registered_no_ops,
        "frozen_v2_detail": {f: {k: rows[f][k] for k in
                                 ("calculator_module", "in_registry", "own_file_operator_count",
                                  "own_file_operators", "base_class_scan", "has_global_cs_operators",
                                  "frozen_v2_role", "audit_would_under_scope")}
                             for f in v2 if f in rows},
        "frozen_v2_not_in_replay_definitions": [f for f in v2 if f not in rows],
        "caveat": "a regex scan of calculator source. It can miss operators built indirectly (a helper in "
                  "another module, a numpy reduction over axis 1 written out longhand) and it can over-count "
                  "where axis=1 is used on a frame that is not date-by-code. Every per-factor operator list "
                  "is recorded so a reader can check the classification rather than trust it.",
        "per_factor": rows,
        "database_writes": False, "weights_written": False,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_factor", "operator_patterns")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
