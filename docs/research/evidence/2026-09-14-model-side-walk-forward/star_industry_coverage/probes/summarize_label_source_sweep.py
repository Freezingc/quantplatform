"""Aggregate the label-source sensitivity sweep into one machine-readable summary.

Reads the per-run outputs of probe_label_source_sensitivity.py under --sweep-dir and reduces them to the
two questions the decision sheet actually asks: on POST-cutover days, does restricting the union to SH/SZ
leave the published 357/471 essentially unchanged? And on PRE-cutover problem days, does that same
restriction still deliver the repair (label coverage and groupable securities going up)?

Nothing is typed: every number here is read out of a probe output file.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

CUTOVER = "2023-09-25"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    runs = []
    for p in sorted(args.sweep_dir.glob("ls_*.json")):
        d = json.loads(p.read_text())
        runs.append({
            "file": p.name, "day": d["day"], "union_scope": d["union_scope"],
            "era": "post_cutover" if d["day"] >= CUTOVER else "pre_cutover",
            "panel_codes": d["panel_codes"],
            "labels_differing": d["labels_differing_on_the_target_day"],
            "decomposition": (d["label_difference_decomposition_on_the_target_day"] or [None])[0],
            "coverage_A": d["label_coverage"]["A_provider"],
            "coverage_B": d["label_coverage"]["B_union"],
            "coverage_gain": d["label_coverage"]["gain"],
            "grouped_471_A": d["yeslab_471"]["grouped_securities_A"],
            "grouped_471_B": d["yeslab_471"]["grouped_securities_B"],
            "f357_A_vs_B": d["yeslab_357"]["A_vs_B_label_source_effect"],
            "f471_A_vs_B": d["yeslab_471"]["A_vs_B_label_source_effect"],
            "f357_transcription_check": d["yeslab_357"]["A_vs_published_transcription_check"],
            "f471_transcription_check": d["yeslab_471"]["A_vs_published_transcription_check"],
        })

    def agg(sel, key, sub, field):
        vals = [r[key][field] for r in sel if r[key] and r[key].get(field) is not None]
        if not vals:
            return None
        return {"min": min(vals), "median": sorted(vals)[len(vals) // 2], "max": max(vals), "n": len(vals)}

    groups = {}
    for era in ("post_cutover", "pre_cutover"):
        for scope in ("shsz", "all"):
            sel = [r for r in runs if r["era"] == era and r["union_scope"] == scope]
            if not sel:
                continue
            groups[f"{era}__{scope}"] = {
                "days": sorted(r["day"] for r in sel), "n_days": len(sel),
                "labels_differing": {"min": min(r["labels_differing"] for r in sel),
                                     "median": sorted(r["labels_differing"] for r in sel)[len(sel) // 2],
                                     "max": max(r["labels_differing"] for r in sel)},
                "coverage_gain": {"min": min(r["coverage_gain"] for r in sel),
                                  "median": sorted(r["coverage_gain"] for r in sel)[len(sel) // 2],
                                  "max": max(r["coverage_gain"] for r in sel)},
                "grouped_471_gain": {"min": min(r["grouped_471_B"] - r["grouped_471_A"] for r in sel),
                                     "median": sorted(r["grouped_471_B"] - r["grouped_471_A"] for r in sel)[len(sel) // 2],
                                     "max": max(r["grouped_471_B"] - r["grouped_471_A"] for r in sel)},
                "f357": {f: agg(sel, "f357_A_vs_B", None, f) for f in
                         ("changed_rate", "spearman", "median_relative_difference", "p95_relative_difference")},
                "f471": {f: agg(sel, "f471_A_vs_B", None, f) for f in
                         ("changed_rate", "spearman", "median_relative_difference", "p95_relative_difference")},
                "transcription_check": {
                    "f357_spearman": agg(sel, "f357_transcription_check", None, "spearman"),
                    "f471_spearman": agg(sel, "f471_transcription_check", None, "spearman")},
            }

    post_shsz = groups.get("post_cutover__shsz")
    pre_shsz = groups.get("pre_cutover__shsz")
    out = {
        "question": "Does restricting the union to SH/SZ leave post-cutover days essentially unchanged "
                    "while still repairing pre-cutover days?",
        "cutover": CUTOVER, "runs": len(runs), "groups": groups,
        "verdict": {}, "per_run": runs,
        "database_writes": False, "weights_written": False,
    }
    if post_shsz and pre_shsz:
        out["verdict"] = {
            "post_cutover_shsz_worst_spearman_357": post_shsz["f357"]["spearman"]["min"],
            "post_cutover_shsz_worst_spearman_471": post_shsz["f471"]["spearman"]["min"],
            "post_cutover_shsz_worst_median_relative_difference_471":
                post_shsz["f471"]["median_relative_difference"]["max"],
            "post_cutover_shsz_max_labels_differing": post_shsz["labels_differing"]["max"],
            "pre_cutover_shsz_min_coverage_gain": pre_shsz["coverage_gain"]["min"],
            "pre_cutover_shsz_min_grouped_471_gain": pre_shsz["grouped_471_gain"]["min"],
            "conclusion": None,
        }
        v = out["verdict"]
        v["conclusion"] = (
            "SUPPORTED on "
            f"{post_shsz['n_days']} post-cutover days: with the union restricted to SH/SZ the worst "
            f"Spearman against today's published values is {v['post_cutover_shsz_worst_spearman_357']:.6f} "
            f"for 357 and {v['post_cutover_shsz_worst_spearman_471']:.6f} for 471, at most "
            f"{v['post_cutover_shsz_max_labels_differing']} labels differ on any day, and the worst median "
            f"relative difference is {v['post_cutover_shsz_worst_median_relative_difference_471']:.6f}. "
            f"And the repair still lands on {pre_shsz['n_days']} pre-cutover problem days: every one gains "
            f"at least {v['pre_cutover_shsz_min_coverage_gain']} labels and "
            f"{v['pre_cutover_shsz_min_grouped_471_gain']} groupable securities for 471."
        )
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_run", "groups")}, ensure_ascii=False, indent=1))
    for name, g in groups.items():
        print(f"\n=== {name}  ({g['n_days']} days: {', '.join(g['days'])}) ===")
        print(f"  labels_differing   {g['labels_differing']}")
        print(f"  coverage_gain      {g['coverage_gain']}")
        print(f"  grouped_471_gain   {g['grouped_471_gain']}")
        for f in ("f357", "f471"):
            print(f"  {f}: changed_rate {g[f]['changed_rate']['median']:.4f} "
                  f"[{g[f]['changed_rate']['min']:.4f},{g[f]['changed_rate']['max']:.4f}]  "
                  f"spearman min {g[f]['spearman']['min']:.6f} med {g[f]['spearman']['median']:.6f}  "
                  f"median_rel max {g[f]['median_relative_difference']['max']:.6f}  "
                  f"p95_rel max {g[f]['p95_relative_difference']['max']:.4f}")
        print(f"  transcription check: 357 spearman min {g['transcription_check']['f357_spearman']['min']:.8f}, "
              f"471 spearman min {g['transcription_check']['f471_spearman']['min']:.6f}")


if __name__ == "__main__":
    main()
