"""Which code path actually produces each of the nine dense factors in factor_daily?

The fidelity census compared factor_daily against the upstream factor named in source_catalog.yaml and
reported yeslab_206 as the least faithful of the nine, with a replayed standard deviation about twice the
stored one. That comparison assumed the replayed column is meant to be the catalogued factor. Reading
materialize_frozen_factor_daily.py says otherwise: MEDIAN_206 = "yeslab_206" is special-cased at
replay_factor_values() and routed to materialize_alpha206_median_repair, i.e. to
compute_variants(market)["median_abs"], the approved median-deviation repair -- NOT to the
Bias_Vol_Demean_60_Calculator that REPLAY_DEFINITIONS["yeslab_206"] names, and not to the upstream
series the catalog points at. The module docstring says so in terms: "the 206 median-repair and 443
reg30_max20 repair variant use their approved formulas".

So for those two, replayed-versus-stored is a repair-versus-canonical comparison and a large gap is the
designed outcome, not a fidelity defect. This resolves, for every one of the nine, which of the three
paths produces it, and re-labels the census tiers accordingly. Nothing here is typed: the paths come from
the source, the definitions from REPLAY_DEFINITIONS, the upstream names from the catalog, and the tiers
from the census measurement files.

SAFETY: reads repository source and local JSON only. No database, no weights, no writes except --out.
"""
from __future__ import annotations
import argparse, ast, json, re, sys
from pathlib import Path

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
MATERIALIZE = REPO / "scripts" / "materialize_frozen_factor_daily.py"
CATALOG = REPO / "configs" / "factors" / "source_catalog.yaml"
DENSE9 = ["yeslab_206", "yeslab_357", "yeslab_381", "yeslab_425", "yeslab_443_reg30_max20_v1",
          "yeslab_464", "yeslab_471", "yeslab_484", "yeslab_96"]


def parse_special_cases(text: str) -> dict:
    """Read the module constants and the body of replay_factor_values, not a hand-copied summary."""
    consts = dict(re.findall(r'^(REPAIR_VARIANT_443|MEDIAN_206)\s*=\s*"([^"]+)"', text, re.M))
    pit = re.search(r"^PIT_FACTORS\s*=\s*\{([^}]*)\}", text, re.M)
    pit_set = sorted(re.findall(r'"([^"]+)"', pit.group(1))) if pit else []
    body = text[text.index("def replay_factor_values"):text.index("def load_official_universe")]
    routes = []
    for m in re.finditer(r"if fid == (\w+):\s*\n\s*return ([^\n]+)", body):
        routes.append({"constant": m.group(1), "call": m.group(2).strip()})
    variant_call = re.search(r"factor_wide = ([^\n]+)", body)
    return {"constants": consts, "PIT_FACTORS": pit_set, "routes": routes,
            "repair_variant_call": variant_call.group(1).strip() if variant_call else None,
            "fallback": "replay._replay_factor(session, replay.REPLAY_DEFINITIONS[fid], ..., "
                        "industry_provider=prov if fid in PIT_FACTORS else None)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    text = MATERIALIZE.read_text(encoding="utf-8")
    paths = parse_special_cases(text)
    const_to_id = {v: k for k, v in paths["constants"].items()}

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    import yaml
    import audit_formula_replay_consistency as replay
    RD = replay.REPLAY_DEFINITIONS
    cat = yaml.safe_load(CATALOG.read_text())
    by_id = {f["factor_id"]: f for f in cat["factors"]}

    tiers = {}
    for p in args.census:
        d = json.loads(p.read_text())
        for fid, v in d.get("per_factor", {}).items():
            tiers.setdefault(fid, []).append({"file": p.name, **{
                k: v[k] for k in ("exact_match_rate_max", "spearman_min", "spearman_median",
                                  "median_relative_difference_median", "classification_by_day")
                if k in v}})

    rows = {}
    for fid in DENSE9:
        const = const_to_id.get(fid)
        d = RD.get(fid)
        entry = by_id.get(fid)
        if const == "MEDIAN_206":
            path, producer = "REPAIR special-case", "materialize_alpha206_median_repair -> compute_variants(market)['median_abs']"
        elif const == "REPAIR_VARIANT_443":
            path, producer = "REPAIR special-case", paths["repair_variant_call"] or "compute_gp074_variant(wide, 30, 20)"
        elif d is not None:
            path = "REPLAY_DEFINITIONS + industry provider" if fid in paths["PIT_FACTORS"] else "REPLAY_DEFINITIONS"
            rl = getattr(d, "replay_lookback_days", None)
            extra = f", replay_lookback {rl}" if rl else ""
            producer = f"{getattr(d, 'calculator_module', None)} (lookback {getattr(d, 'lookback_days', None)}{extra})"
        else:
            path, producer = "UNKNOWN", "no REPLAY_DEFINITIONS entry and no special case"
        rows[fid] = {
            "production_path": path,
            "producer": producer,
            "constant_in_materialize": paths["constants"].get(const) if const else None,
            "in_REPLAY_DEFINITIONS": d is not None,
            "replay_definition_factor_name": getattr(d, "factor_name", None),
            "replay_definition_store_factor": getattr(d, "store_factor", None),
            "replay_definition_calculator": getattr(d, "calculator_module", None),
            "in_source_catalog": entry is not None,
            "catalog_source_table": entry.get("source_table") if entry else None,
            "catalog_source_factor_name": entry.get("source_factor_name") if entry else None,
            "takes_industry_provider": fid in paths["PIT_FACTORS"],
            "fidelity_comparison_is_meaningful": path.startswith("REPLAY_DEFINITIONS"),
            "why": ("the replayed column IS the catalogued factor, so replayed-versus-stored is a "
                    "fidelity test" if path.startswith("REPLAY_DEFINITIONS") else
                    "the replayed column is an APPROVED REPAIR VARIANT of a different formula, so "
                    "replayed-versus-stored compares repair against canonical and a large gap is the "
                    "designed outcome, not a fidelity defect"),
            "census_tiers": tiers.get(fid, []),
        }

    meaningful = [f for f, v in rows.items() if v["fidelity_comparison_is_meaningful"]]
    repair = [f for f, v in rows.items() if not v["fidelity_comparison_is_meaningful"]]
    out = {
        "question": "Which code path produces each dense factor in factor_daily, and for which of them is "
                    "a replayed-versus-upstream fidelity comparison meaningful at all?",
        "source_of_truth": str(MATERIALIZE),
        "special_cases_parsed_from_source": paths,
        "REPLAY_DEFINITIONS_entry_count": len(RD),
        "dense_nine": rows,
        "fidelity_comparison_meaningful_for": meaningful,
        "approved_repair_variants_excluded_from_the_fidelity_census": repair,
        "verdict": (
            f"{len(repair)} of the {len(rows)} dense factors ({', '.join(repair)}) are approved repair "
            f"variants routed around REPLAY_DEFINITIONS, so the census's divergence figures for them "
            "measure repair-versus-canonical and must not be read as fidelity defects. The remaining "
            f"{len(meaningful)} are genuine fidelity measurements."
        ),
        "database_writes": False, "weights_written": False,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("dense_nine", "special_cases_parsed_from_source")},
                     ensure_ascii=False, indent=1))
    print("\nspecial cases parsed from source:", json.dumps(paths, ensure_ascii=False, indent=1))
    print(f"\n{'factor':28}{'path':42}{'catalog?':9}{'fidelity test meaningful?':26}producer")
    for fid, v in rows.items():
        print(f"{fid:28}{v['production_path']:42}{str(v['in_source_catalog']):9}"
              f"{str(v['fidelity_comparison_is_meaningful']):26}{v['producer']}")


if __name__ == "__main__":
    main()
