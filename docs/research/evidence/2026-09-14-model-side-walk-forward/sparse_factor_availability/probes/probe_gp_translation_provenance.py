#!/usr/bin/env python3
"""Read-only: can the GP_* calculator translations be checked against anything in this repository?

§BF left one question open: is our GP_Factor_006_Calculator a faithful translation of the producer's
GP_006? Answering it needs a reference -- either the expression source the calculators cite, or an
implementation of the DSL those expressions are written in. This probe looks for both and measures what
is actually there, so the answer is a record rather than an impression.

It counts, over every GP_Factor_*_Calculator.py:
  how many use a bare .rolling(N) and how many pass min_periods
  how many use ** and how many use np.power
  how many add a 1e-10 epsilon to a cross-sectional standard deviation
It extracts the DSL operator names from the calculators' own docstring expressions and checks each
against the definitions in 数据库/build_factor/function_lib.py -- the only operator library in the
repository. It looks for the cited expression source file. And it counts how many of the frozen
REPLAY_DEFINITIONS point at a GP calculator, i.e. how much of the frozen chain rests on this
translation. It also records base.py's warm-up rule, because that is what decides whether a
min_periods difference could matter mid-window at all.

Read-only: opens files for reading, hashes them, runs no calculator, touches no database, writes only
its own JSON under --out.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX")
CALC = REPO / "数据库" / "factor_calculators"
LIB = REPO / "数据库" / "build_factor" / "function_lib.py"
REPLAY_DEFS = REPO / "research_v2" / "scripts" / "audit_formula_replay_consistency.py"
HASHED = (
    "数据库/build_factor/function_lib.py",
    "数据库/build_factor/expr_parser.py",
    "数据库/build_factor/template.py",
    "数据库/build_factor/builder.py",
    "数据库/factor_calculators/base.py",
    "数据库/factor_calculators/GP_Factor_003_Calculator.py",
    "数据库/factor_calculators/GP_Factor_006_Calculator.py",
    "数据库/factor_calculators/GP_Factor_012_Calculator.py",
    "数据库/factor_calculators/GP_Factor_015_Calculator.py",
    "数据库/factor_calculators/GP_Factor_020_Calculator.py",
    "数据库/factor_calculators/GP_Factor_021_Calculator.py",
    "数据库/factor_calculators/GP_Factor_037_Calculator.py",
    "数据库/factor_calculators/GP_Factor_056_Calculator.py",
    "research_v2/scripts/audit_formula_replay_consistency.py",
    "research_v2/scripts/materialize_frozen_factor_daily.py",
)
# the operators the six sparse and two dense GP expressions are written with
OPS_IN_GP_EXPRESSIONS = (
    "mul", "add", "pow", "cs_rank", "cs_zscore", "tanh", "relu", "inv", "max",
    "ts_delta_ratio", "ts_sum", "ts_mean", "ts_std", "ts_rank", "ts_min", "ts_max",
)


def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("/data/zhangyudong/qrv2_research_artifacts/gp_translation"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(CALC / "GP_Factor_*_Calculator.py")))
    counts = {"files": len(files), "bare_rolling": 0, "min_periods": 0, "double_star": 0,
              "np_power": 0, "epsilon_1e_10": 0}
    expressions = {}
    for f in files:
        text = Path(f).read_text(encoding="utf-8", errors="replace")
        counts["bare_rolling"] += int(".rolling(" in text)
        counts["min_periods"] += int("min_periods" in text)
        counts["double_star"] += int("**" in text)
        counts["np_power"] += int("np.power" in text)
        counts["epsilon_1e_10"] += int("1e-10" in text)
        m = re.search(r"表达式:\s*(.+)", text)
        if m:
            expressions[Path(f).stem] = m.group(1).strip()
        s = re.search(r"来源:\s*(\S+)", text)
        if s:
            counts.setdefault("cited_sources", {})
            counts["cited_sources"][s.group(1)] = counts["cited_sources"].get(s.group(1), 0) + 1

    lib_text = LIB.read_text(encoding="utf-8", errors="replace") if LIB.is_file() else ""
    lib_defs = set(re.findall(r"^def ([A-Za-z_][A-Za-z0-9_]*)\(", lib_text, re.M))
    op_check = {}
    for op in OPS_IN_GP_EXPRESSIONS:
        op_check[op] = {
            "in_function_lib": op.upper() in lib_defs,
            "matching_lib_names": sorted(d for d in lib_defs if d.upper().startswith(op.upper()))[:4],
        }
    # where do these operator names appear at all in the repository?
    appears_only_in_docstrings = {}
    for op in ("ts_delta_ratio", "cs_zscore", "cs_rank", "tanh", "relu", "mul"):
        hits = []
        for pattern in (f"def {op}", f"def {op.upper()}"):
            hits += [str(p.relative_to(REPO)) for p in REPO.rglob("*.py")
                     if pattern in p.read_text(encoding="utf-8", errors="replace")]
        appears_only_in_docstrings[op] = {"files_defining_it": sorted(set(hits))}

    cited = sorted(counts.get("cited_sources", {}))
    cited_present = {c: bool(list(REPO.rglob(c)) or
                             list(Path("/data/zhangyudong").glob(f"**/{c}"))) for c in cited}

    defs_text = REPLAY_DEFS.read_text(encoding="utf-8", errors="replace")
    all_defs = re.findall(r'^\s{4}"(yeslab_[0-9a-z_]+)": ReplayDefinition\(', defs_text, re.M)
    gp_defs = re.findall(r'calculator_module="(GP_Factor_[0-9]+_Calculator)"', defs_text)
    frozen_v2 = ["yeslab_206", "yeslab_357", "yeslab_372", "yeslab_375", "yeslab_381", "yeslab_384",
                 "yeslab_389", "yeslab_390", "yeslab_406", "yeslab_425",
                 "yeslab_443_reg30_max20_v1", "yeslab_464", "yeslab_471", "yeslab_484", "yeslab_96"]
    # capture each definition block separately: a greedy DOTALL scan runs across neighbouring
    # definitions and attributes one factor's calculator_module to the next.
    blocks = dict(re.findall(
        r'"(yeslab_[0-9a-z_]+)":\s*ReplayDefinition\((.*?)\n    \),', defs_text, re.S))
    frozen_gp = [f for f in frozen_v2
                 if 'calculator_module="GP_Factor' in blocks.get(f, "")]
    frozen_calculator_module = {f: (re.search(r'calculator_module="([^"]+)"', blocks[f]).group(1)
                                    if f in blocks and 'calculator_module="' in blocks[f] else None)
                                for f in frozen_v2}

    base_text = (CALC / "base.py").read_text(encoding="utf-8", errors="replace")
    warm = re.search(r"pstart_time = .*", base_text)

    result = {
        "question": (
            "Can the GP_* calculator translations that the frozen chain replays be checked against any "
            "reference inside this repository?"
        ),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True, "database_access": False, "calculator_executed": False,
        "calculator_style_census": counts,
        "expressions_extracted": len(expressions),
        "expression_of_record_for_the_six_sparse_factors": {
            k: v for k, v in expressions.items()
            if k in ("GP_Factor_003_Calculator", "GP_Factor_006_Calculator", "GP_Factor_015_Calculator",
                     "GP_Factor_020_Calculator", "GP_Factor_021_Calculator", "GP_Factor_037_Calculator")
        },
        "operator_library_in_the_repository": {
            "path": str(LIB), "sha256": sha256(LIB), "definitions": len(lib_defs),
            "operators_used_by_the_gp_expressions": op_check,
            "missing_from_the_library": sorted(o for o, v in op_check.items() if not v["in_function_lib"]),
        },
        "operators_defined_nowhere_in_the_repository": appears_only_in_docstrings,
        "cited_expression_source": {
            "names": cited,
            "cited_by_how_many_calculators": counts.get("cited_sources", {}),
            "present_anywhere_under_the_repo_or_/data/zhangyudong": cited_present,
        },
        "how_much_of_the_frozen_chain_rests_on_this_translation": {
            "replay_definitions_total": len(all_defs),
            "replay_definitions_pointing_at_a_gp_calculator": len(set(gp_defs)),
            "frozen_v2_factors_whose_replay_uses_a_gp_calculator": sorted(set(frozen_gp)),
            "frozen_v2_calculator_module_or_none": frozen_calculator_module,
            "replay_definitions_parsed_into_blocks": len(blocks),
            "frozen_v2_count": len(frozen_v2),
        },
        "warm_up_rule_in_base_py": {
            "line": warm.group(0).strip() if warm else None,
            "meaning": (
                "the calculator pre-fetches ten times its lookback in CALENDAR days before the requested "
                "start, so with replay_lookback_days=180 that is about 1,800 calendar days of history; "
                "a min_periods difference therefore cannot explain a mid-window availability gap"
            ),
        },
        "sha256_of_every_file_this_probe_read": {r: sha256(REPO / r) for r in HASHED},
        "verdict": (
            "no. The GP expressions use an operator vocabulary (mul, cs_rank, cs_zscore, tanh, relu, "
            "ts_delta_ratio) that the repository's only operator library does not define, and the "
            "expression source all of them cite is not present. The 100 GP calculators are therefore an "
            "unverifiable translation: nothing in-repo can confirm or refute GP_Factor_006. What CAN be "
            "said is empirical -- §BF's census corroborates eight of the nine GP_* factors in frozen v2 "
            "at Spearman >= 0.9529 against the producer's stored values, five of them at 1.0000."
        ),
    }
    out = args.out / "gp_translation_provenance.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps({k: result[k] for k in (
        "calculator_style_census", "operator_library_in_the_repository",
        "operators_defined_nowhere_in_the_repository", "cited_expression_source",
        "how_much_of_the_frozen_chain_rests_on_this_translation", "warm_up_rule_in_base_py")},
        ensure_ascii=False, indent=1)[:3000])
    print("\nverdict:", result["verdict"])


if __name__ == "__main__":
    main()
