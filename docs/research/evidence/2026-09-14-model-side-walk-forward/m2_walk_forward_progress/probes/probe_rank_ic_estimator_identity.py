#!/usr/bin/env python3
"""Read-only probe: is `ensemble_validation_rank_ic` the same estimator as the baseline's `mean_rank_ic`?

The 14:57 entry compared M2's winner against six prior carriers and the frozen baseline on "the same F1
validation slice", and then listed as its own biggest unclosed assumption that nothing had verified
`ensemble_validation_rank_ic` and `mean_rank_ic` are the same estimator -- if they are not, that whole
standings table is comparing numbers defined differently. §BO.6 made the question sharper by finding a
case where the per-seed and the ensemble statistic name opposite best levels.

This probe answers it from source, with an AST parse rather than a text diff, and does not import or
execute anything from the repository: every file is opened and parsed as data.

Three claims are checked:

  1. `training/search_metrics.py:validation_rank_ic` returns `summarize_rank_ic(...).mean_rank_ic`, so
     the search metric IS a mean_rank_ic, not a different functional.
  2. The search path and the baseline path call `compute_daily_rank_ic` with the same effective
     validity rule (floor 30, score health NOT enforced), so they average over the same day set even
     though one passes include_invalid=False and the other True.
  3. The OOS path in `evaluation/model_evaluator.py` passes require_score_health=True, so the OOS
     mean_rank_ic is NOT the same estimator as the validation one -- a source-level reason the two
     levels are not comparable, on top of them covering different years.

It also checks whether the driver's private `_validation_rank_ic` is the same function as the shared
`search_metrics.validation_rank_ic`, by comparing their ASTs with the name and docstring normalised
away. search_metrics.py's own docstring says it exists so the number is "defined once"; this measures
whether that is true by construction or only by coincidence.

Safety: parses repository files as text. No import, no execution, no database, no weight loading,
nothing written into any artifact directory.
"""
from __future__ import annotations

import argparse
import ast
import json
from datetime import datetime
from pathlib import Path

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX")

# the production files that call compute_daily_rank_ic, found by grep and listed here so the probe
# fails loudly if one is added or removed rather than silently analysing a stale set. Each file is
# walked exactly once and every call is attributed to its real enclosing function, so the row count
# equals the call count. (An earlier draft keyed this table by (file, function) and listed
# model_evaluator.py twice, which produced four rows for two call sites.)
CITED_FILES = (
    "src/research_v2/training/search_metrics.py",
    "src/research_v2/baselines/orchestration.py",
    "src/research_v2/training/validation.py",
    "src/research_v2/evaluation/model_evaluator.py",
    "src/research_v2/datasets/selection.py",
    "scripts/run_real_candidate_training.py",
)

ROLE_HINTS = {
    ("src/research_v2/training/search_metrics.py", "validation_rank_ic"):
        "THE SEARCH SELECTION METRIC (what the driver's argmax ranks on)",
    ("scripts/run_real_candidate_training.py", "_validation_rank_ic"):
        "the driver's private copy of the same metric",
    ("src/research_v2/baselines/orchestration.py", "_evaluate_predictions"):
        "BASELINE VALIDATION (the other side of the standings table)",
    ("src/research_v2/training/validation.py", "evaluate_validation_batches"):
        "inner validation during training",
    ("src/research_v2/evaluation/model_evaluator.py", "evaluate_frozen_oos"):
        "OOS ACCEPTANCE PATH (candidate at :65, baseline at :86)",
    ("src/research_v2/datasets/selection.py", "select_training_factors"):
        "training-factor selection",
}


def parse(rel: str) -> ast.Module:
    return ast.parse((REPO / "research_v2" / rel).read_text(encoding="utf-8"))


def find_calls(tree: ast.Module, func_name: str) -> list[ast.Call]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            nm = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
            if nm == func_name:
                out.append(node)
    return out


def kwarg_summary(call: ast.Call) -> dict:
    """Render each keyword argument as source text; absent means 'takes the default'."""
    out = {}
    for kw in call.keywords:
        try:
            out[kw.arg] = ast.unparse(kw.value)
        except Exception:
            out[kw.arg] = "<unrenderable>"
    return out


def normalise_body(fn: ast.FunctionDef) -> str:
    """AST of a function body with its docstring removed, so only executable content is compared."""
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return "\n".join(ast.dump(st, annotate_fields=True, include_attributes=False) for st in body)


def get_func(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def main() -> None:
    global REPO
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--out", type=Path,
                    default=Path("/data/zhangyudong/qrv2_research_artifacts/rank_ic_estimator_identity"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    REPO = args.repo

    # ---- 1. what does the search metric actually return?
    sm_tree = parse("src/research_v2/training/search_metrics.py")
    sm_fn = get_func(sm_tree, "validation_rank_ic")
    ret = [n for n in ast.walk(sm_fn) if isinstance(n, ast.Return)][-1]
    returns_mean_rank_ic = "mean_rank_ic" in ast.unparse(ret)
    calls_summarize = "summarize_rank_ic" in ast.unparse(ret)
    docstring = ast.get_docstring(sm_fn) or ""

    # ---- 2/3. every production call site's effective arguments
    trees = {rel: parse(rel) for rel in CITED_FILES}
    sites = []
    for rel, tree in trees.items():
        # map every call node to the innermost function that contains it
        enclosing = {}
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            for sub in ast.walk(fn):
                if isinstance(sub, ast.Call) and sub is not fn:
                    enclosing.setdefault(id(sub), fn.name)
        for call in find_calls(tree, "compute_daily_rank_ic"):
            kw = kwarg_summary(call)
            include_invalid = kw.get("include_invalid")
            require_health = kw.get("require_score_health")
            # compute_daily_rank_ic: enforce = include_invalid if require_score_health is None else require
            if require_health is None:
                enforced = "False" if include_invalid in (None, "False") else str(include_invalid)
            else:
                enforced = require_health
            fname = enclosing.get(id(call), "<module level>")
            sites.append({
                "file": f"research_v2/{rel}", "line": call.lineno, "enclosing_function": fname,
                "role": ROLE_HINTS.get((rel, fname), ""),
                "minimum_cross_section": kw.get("minimum_cross_section", "<default 30>"),
                "include_invalid": include_invalid or "<default False>",
                "require_score_health": require_health or "<default None -> follows include_invalid>",
                "score_column": kw.get("score_column", "<default 'score'>"),
                "EFFECTIVE_score_health_enforced": enforced,
            })
    sites.sort(key=lambda s: (s["file"], s["line"]))
    seen = {(s["file"], s["line"]) for s in sites}
    assert len(seen) == len(sites), "duplicate call-site rows; the walk is not one-per-call"

    # ---- the two definitions of the selection metric: same function or a coincidental duplicate?
    drv_tree = parse("scripts/run_real_candidate_training.py")
    drv_fn = get_func(drv_tree, "_validation_rank_ic")
    a_norm, b_norm = normalise_body(sm_fn), normalise_body(drv_fn)
    identical = a_norm == b_norm

    # ---- who actually selects the winning config?
    selectors = []
    for node in ast.walk(drv_tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "max":
            src = ast.unparse(node)
            if "ensemble_validation_rank_ic" in src:
                selectors.append({"line": node.lineno, "source": src})
    # Are search.py's two selection-layer functions reachable from production at all? Scan EVERY
    # production python file (not a hand-picked two) for a reference that is not the definition, and
    # also look inside `__all__` and import statements, so "dead" is an audited claim not an inference.
    def references(name: str) -> dict:
        hits, defs, imports, dunder_all = [], [], [], []
        roots = [q for q in (REPO / "research_v2").rglob("*.py") if "tests" not in q.parts]
        for path in sorted(roots):
            rel = str(path.relative_to(REPO))
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    defs.append(f"{rel}:{node.lineno}")
                elif isinstance(node, (ast.ImportFrom,)):
                    if any(a.name == name for a in node.names):
                        imports.append(f"{rel}:{node.lineno}")
                elif isinstance(node, ast.Name) and node.id == name:
                    hits.append(f"{rel}:{node.lineno}")
                elif isinstance(node, ast.Attribute) and node.attr == name:
                    hits.append(f"{rel}:{node.lineno}")
                elif isinstance(node, ast.Constant) and node.value == name:
                    dunder_all.append(f"{rel}:{node.lineno}")
        return {"definition_sites": defs, "import_sites": imports,
                "name_or_attribute_references_outside_the_definition": hits,
                "string_literal_occurrences_eg___all__": dunder_all}

    smc = references("select_model_config")
    bprt = references("build_pre_registered_trials")
    smc_refs = smc["name_or_attribute_references_outside_the_definition"]
    bprt_refs = bprt["name_or_attribute_references_outside_the_definition"]
    search_rel = "src/research_v2/training/search.py"
    if search_rel not in trees:
        trees[search_rel] = parse(search_rel)

    result = {
        "question": ("Is the search's ensemble_validation_rank_ic the same estimator as the baseline's "
                     "mean_rank_ic, over the same day set, and which function actually selects the winner?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True, "database_access": False, "weights_loaded": False,
        "repository_module_imported_or_executed": False,
        "method": "every repository file is opened and parsed with ast; nothing is imported or run",
        "claim_1_the_search_metric_is_a_mean_rank_ic": {
            "return_statement": ast.unparse(ret),
            "returns_summarize_rank_ic": calls_summarize,
            "returns_the_mean_rank_ic_field": returns_mean_rank_ic,
            "verdict": ("SAME ESTIMATOR FAMILY: the search metric is literally "
                        "summarize_rank_ic(compute_daily_rank_ic(...)).mean_rank_ic"
                        if (calls_summarize and returns_mean_rank_ic) else "DIFFERENT"),
            "module_docstring_first_line": (ast.get_docstring(sm_tree) or "").splitlines()[0],
            "function_docstring_says_floor_is_shared": "same floor" in docstring,
        },
        "claim_2_and_3_every_production_call_site": sites,
        "day_set_equivalence": {
            "rule_in_compute_daily_rank_ic": (
                "rank_ic_valid is set only when len(scored) >= minimum_cross_section AND "
                "(not enforce_score_health OR score_health_valid); with include_invalid=False the "
                "invalid rows are dropped before summarize_rank_ic, with include_invalid=True they are "
                "kept but summarize_rank_ic filters on rank_ic_valid & finite -- so both average over "
                "exactly the same set of days"),
            "search_floor": next(s["minimum_cross_section"] for s in sites if "SEARCH SELECTION METRIC" in s["role"]),
            "baseline_floor": next(s["minimum_cross_section"] for s in sites if "BASELINE VALIDATION" in s["role"]),
            "search_score_health_enforced": next(s["EFFECTIVE_score_health_enforced"] for s in sites if "SEARCH SELECTION METRIC" in s["role"]),
            "baseline_score_health_enforced": next(s["EFFECTIVE_score_health_enforced"] for s in sites if "BASELINE VALIDATION" in s["role"]),
            "oos_score_health_enforced": sorted({s["EFFECTIVE_score_health_enforced"] for s in sites if "OOS" in s["role"]}),
            "verdict_validation_paths_comparable": (
                next(s["EFFECTIVE_score_health_enforced"] for s in sites if "SEARCH SELECTION METRIC" in s["role"])
                == next(s["EFFECTIVE_score_health_enforced"] for s in sites if "BASELINE VALIDATION" in s["role"])),
            "verdict_oos_path_differs": "True" in {s["EFFECTIVE_score_health_enforced"] for s in sites if "OOS" in s["role"]},
        },
        "the_duplicate_definition": {
            "shared_module": "research_v2/src/research_v2/training/search_metrics.py:validation_rank_ic",
            "driver_private_copy": "research_v2/scripts/run_real_candidate_training.py:_validation_rank_ic",
            "ast_identical_after_removing_name_and_docstring": identical,
            "importers_of_the_shared_module": [
                "research_v2/src/research_v2/training/temporal_search.py:47",
                "research_v2/src/research_v2/training/cross_section_search.py:21"],
            "driver_call_sites_of_its_own_copy": sorted(
                {n.lineno for n in ast.walk(drv_tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "_validation_rank_ic"}),
            "verdict": ("the two agree today, so the numbers are consistent -- but the shared module's "
                        "docstring says it exists so the metric is 'defined once', and the largest "
                        "consumer does not use it, so that guarantee currently holds by coincidence "
                        "rather than by construction" if identical else
                        "THE TWO DEFINITIONS DIFFER; every number that mixes them is suspect"),
        },
        "who_selects_the_winner": {
            "argmax_on_ensemble_rank_ic_in_the_driver": selectors,
            "n_such_sites": len(selectors),
            "select_model_config_scan": smc,
            "select_model_config_is_production_dead": not smc_refs and not smc["import_sites"],
            "build_pre_registered_trials_scan": bprt,
            "build_pre_registered_trials_is_production_dead": not bprt_refs and not bprt["import_sites"],
            "dead_code_test": ("a function is production-dead here if no non-test file imports it and "
                               "no non-test file names it outside its own definition; tests are excluded "
                               "from the scan by path, so a test-only function still counts as dead"),
        },
    }
    out = args.out / "rank_ic_estimator_identity.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    c1 = result["claim_1_the_search_metric_is_a_mean_rank_ic"]
    print(f"\nclaim 1: {c1['verdict']}")
    print(f"  return statement: {c1['return_statement']}")
    print("\nclaim 2/3: every production call site of compute_daily_rank_ic")
    print(f"  {'file:line':<52} {'floor':<10} {'incl_inv':<12} {'req_health':<12} {'EFFECTIVE':<10} {'score_col':<12} role")
    for s in sites:
        print(f"  {s['file'].split('research_v2/')[-1]+':'+str(s['line']):<52} "
              f"{str(s['minimum_cross_section']):<10} {str(s['include_invalid']):<12} "
              f"{str(s['require_score_health'])[:12]:<12} {str(s['EFFECTIVE_score_health_enforced']):<10} "
              f"{str(s['score_column'])[:12]:<12} {s['role']}")
    d = result["day_set_equivalence"]
    print(f"\n  validation paths comparable: {d['verdict_validation_paths_comparable']}")
    print(f"  OOS path differs:            {d['verdict_oos_path_differs']}  (enforced={d['oos_score_health_enforced']})")
    dup = result["the_duplicate_definition"]
    print(f"\nduplicate definition AST-identical: {dup['ast_identical_after_removing_name_and_docstring']}")
    print(f"  driver call sites of its own copy: {dup['driver_call_sites_of_its_own_copy']}")
    w = result["who_selects_the_winner"]
    print(f"\nwinner selection: {w['n_such_sites']} argmax-on-ensemble sites in the driver:")
    for s in w["argmax_on_ensemble_rank_ic_in_the_driver"]:
        print(f"  :{s['line']}  {s['source'][:96]}")
    for nm in ("select_model_config", "build_pre_registered_trials"):
        sc = w[f"{nm}_scan"]
        print(f"  {nm}: production-dead={w[f'{nm}_is_production_dead']}")
        print(f"     definitions        {sc['definition_sites']}")
        print(f"     non-test imports   {sc['import_sites']}")
        print(f"     non-test references{sc['name_or_attribute_references_outside_the_definition']}")
        print(f"     string literals    {sc['string_literal_occurrences_eg___all__']}")


if __name__ == "__main__":
    main()
