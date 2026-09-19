#!/usr/bin/env python3
"""Read-only probe for §CT: extend §CC.1's "shipped bundle == peak checkpoint" check to the just-landed
M2 F2 refit bundle, and upgrade the finding from an observation (15/15 byte-identical) to a MECHANISM
guarantee read from source.

WHAT §CC.1 ESTABLISHED, AND WHAT WAS OPEN. §CC.1 verified by sha256 that models/<id>/seed-<seed>.pt is
byte-identical to checkpoints/…-final-<seed>-epoch-<peak>/checkpoint.pt for M1's 12 runs and M2 F1's 3
runs (15/15), which upgraded §CA.6's "the shipped OOS numbers describe the shipped model" from a
source-code inference to a byte-level measurement. But when §CC.1 was written the M2 F2 refit had not
finished, so its bundle did not exist and could not be checked. During §CS the F2 refit COMPLETED and
models/m2-f2 was written (07:19), so the check can now be extended to F2's three seeds -- including seed
20260802, whose peak of 71 over 35 checkpoints is the strongest available test case (a bundle equal to
the LAST checkpoint would differ from one equal to the PEAK checkpoint if any non-improving epoch had
been saved after the peak; §CM.3 established none are, but this bundle is where that would show).

THE MECHANISM, read from source, which is stronger than 18/18 byte matches:
  * trainer.py:110-116 saves a checkpoint ONLY on strict improvement (if validation_rank_ic >
    best_value), so the sequence of saved epochs is strictly increasing in validation rank_ic and the
    LAST saved checkpoint IS the peak.
  * cross_section_search.py:206 sets the bundle's source path to checkpoint_store._root /
    run.checkpoint_ids[-1] / "checkpoint.pt" -- i.e. the LAST saved checkpoint, which is the peak.
  * model_bundle.py:90 does shutil.copyfile(source, temporary / f"seed-{seed}.pt") -- a byte-for-byte
    copy, so the bundle .pt is bit-identical to that peak checkpoint.
Therefore "bundle == peak checkpoint" is guaranteed by construction, not a coincidence; the 18/18 byte
matches are the empirical confirmation of that chain. This also explains why §CA's defect (the SEARCH
record scoring the LAST-epoch weights rather than the peak) does NOT touch the shipped bundle: the bundle
is copied from the peak checkpoint regardless of what the search record scored.

WHAT IS MEASURED (sha256, read-only; no .pt is ever loaded as a tensor):
  for each of M1 (4 folds x 3 seeds = 12), M2 F1 (3), M2 F2 (3):
    hash(models/<id>/seed-<s>.pt) == hash(checkpoints/<...-final-<s>-epoch-<peak>/checkpoint.pt)
  and, as a negative control that the check can FAIL, the bundle is also compared to a NON-peak
  checkpoint where one exists (seed 20260802 has 35 checkpoints; epoch 1 must NOT match the bundle),
  proving the equality is specific to the peak and not "all checkpoints look alike".

Only file sha256 hashes, directory names, and source text are read. No .pt loaded as a tensor, no
parquet row data (validation_predictions.parquet is not even opened here), no repository module imported
or executed, no database, no network, no writes to any artifact directory, no signal to any process. Does
not touch /tmp/walk_forward_cache_v1, so no atime side effect.
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "measurements" / "m2_f2_bundle_equals_peak_checkpoint.json"

SEEDS = ("20260801", "20260802", "20260803")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def peak_and_checkpoints(ws: str, car: str, fold: str, seed: str) -> tuple[int, list[int]]:
    ds = glob.glob(str(ART / ws / "checkpoints" / f"{car}-{fold}-final-{seed}-seed-{seed}-epoch-*"))
    epochs = sorted(int(re.search(r"epoch-(\d+)$", d).group(1)) for d in ds)
    return (epochs[-1] if epochs else -1), epochs


def check_carrier(ws: str, car: str, model_id_fmt: str, folds: list[str]) -> list[dict]:
    rows = []
    for fold in folds:
        model_id = model_id_fmt.format(fold=fold, car=car)
        bundle_dir = ART / ws / "models" / model_id
        for seed in SEEDS:
            bundle_pt = bundle_dir / f"seed-{seed}.pt"
            peak, epochs = peak_and_checkpoints(ws, car, fold, seed)
            peak_ckpt = (ART / ws / "checkpoints"
                         / f"{car}-{fold}-final-{seed}-seed-{seed}-epoch-{peak}" / "checkpoint.pt")
            row = {"carrier": ws, "model_id": model_id, "fold": fold, "seed": seed,
                   "bundle_exists": bundle_pt.is_file(), "peak_ckpt_exists": peak_ckpt.is_file(),
                   "peak_epoch": peak, "n_checkpoints": len(epochs)}
            if bundle_pt.is_file() and peak_ckpt.is_file():
                bh = sha256(bundle_pt); ph = sha256(peak_ckpt)
                row["bundle_sha256"] = bh[:16]; row["peak_ckpt_sha256"] = ph[:16]
                row["bundle_equals_peak"] = bh == ph
                # negative control: a NON-peak checkpoint must NOT equal the bundle (where one exists)
                non_peak = [e for e in epochs if e != peak]
                if non_peak:
                    e0 = non_peak[0]
                    np_ckpt = (ART / ws / "checkpoints"
                               / f"{car}-{fold}-final-{seed}-seed-{seed}-epoch-{e0}" / "checkpoint.pt")
                    if np_ckpt.is_file():
                        nph = sha256(np_ckpt)
                        row["non_peak_control_epoch"] = e0
                        row["non_peak_equals_bundle"] = nph == bh
            rows.append(row)
    return rows


def main() -> None:
    rows = []
    rows += check_carrier("m1_full_v1", "m1", "{car}-{fold}", ["f1", "f2", "f3", "f4"])
    rows += check_carrier("m2_v1", "m2", "{car}-{fold}", ["f1", "f2"])

    checked = [r for r in rows if r["bundle_exists"] and r["peak_ckpt_exists"]]
    matched = [r for r in checked if r["bundle_equals_peak"]]
    # negative controls: every non-peak control must be a MISMATCH
    controls = [r for r in checked if "non_peak_equals_bundle" in r]
    controls_ok = [r for r in controls if not r["non_peak_equals_bundle"]]

    by_carrier = {}
    for r in checked:
        by_carrier.setdefault(r["carrier"], {"n": 0, "matched": 0})
        by_carrier[r["carrier"]]["n"] += 1
        by_carrier[r["carrier"]]["matched"] += int(r["bundle_equals_peak"])

    f2 = [r for r in checked if r["carrier"] == "m2_v1" and r["fold"] == "f2"]
    doc = {
        "question": ("does the shipped M2 F2 refit bundle equal the PEAK checkpoint (extending §CC.1's "
                     "15/15 to F2), and is that equality guaranteed by construction rather than chance?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "mechanism_from_source": {
            "save_on_strict_improvement": "trainer.py:110-116 (if validation_rank_ic > best_value)",
            "bundle_source_is_last_saved": ("cross_section_search.py:206 "
                                            "(checkpoint_store._root / run.checkpoint_ids[-1] / checkpoint.pt)"),
            "bundle_is_byte_copy": "model_bundle.py:90 (shutil.copyfile(source, temporary / seed-{seed}.pt))",
            "chain": ("last saved == peak (strict-improvement saves) -> bundle source is last saved -> "
                      "bundle is a byte copy of it, so bundle == peak checkpoint BY CONSTRUCTION"),
            "why_CA_defect_does_not_touch_the_bundle": (
                "§CA's defect is that the SEARCH record scores the last-EPOCH weights; the bundle is "
                "copied from the peak checkpoint regardless, so shipped OOS numbers are unaffected -- "
                "this is the byte-level backing for §CC.1/D9"),
        },
        "per_run": rows,
        "summary": {
            "n_checked": len(checked),
            "n_bundle_equals_peak": len(matched),
            "all_match": len(matched) == len(checked),
            "by_carrier": by_carrier,
            "CC1_was_15_of_15": 15,
            "now_18_of_18_with_F2": len(checked) == 18 and len(matched) == 18,
            "F2_rows": [{"seed": r["seed"], "peak_epoch": r["peak_epoch"],
                         "n_checkpoints": r["n_checkpoints"], "bundle_equals_peak": r["bundle_equals_peak"],
                         "bundle_sha256": r.get("bundle_sha256")} for r in f2],
        },
        "negative_controls": {
            "what": ("the bundle must equal the PEAK checkpoint but NOT a non-peak one; seed 20260802 has "
                     "35 checkpoints so epoch 1 is available as a decoy. If the bundle matched a non-peak "
                     "checkpoint too, 'bundle == peak' would be vacuous (all checkpoints alike)."),
            "n_runs_with_a_non_peak_control": len(controls),
            "n_controls_correctly_mismatched": len(controls_ok),
            "all_controls_mismatch": len(controls) > 0 and len(controls_ok) == len(controls),
            "example_seed_20260802": next((
                {"peak": r["peak_epoch"], "n_checkpoints": r["n_checkpoints"],
                 "non_peak_control_epoch": r.get("non_peak_control_epoch"),
                 "bundle_equals_peak": r["bundle_equals_peak"],
                 "non_peak_equals_bundle": r.get("non_peak_equals_bundle")}
                for r in checked if r["carrier"] == "m2_v1" and r["fold"] == "f2"
                and r["seed"] == "20260802"), None),
        },
        "read_only": True, "pt_loaded_as_tensor": False, "database_access": False,
        "weights_loaded": False, "parquet_read": False, "training_performed": False,
        "repository_module_imported_or_executed": False, "processes_signalled": [],
        "atime_side_effect": False,
    }
    nc = doc["negative_controls"]
    doc["ALL_PASS"] = bool(doc["summary"]["all_match"] and nc["all_controls_mismatch"]
                           and doc["summary"]["now_18_of_18_with_F2"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("=== bundle == peak checkpoint, per carrier ===")
    for c, v in by_carrier.items():
        print(f"  {c}: {v['matched']}/{v['n']} matched")
    print(f"  TOTAL: {len(matched)}/{len(checked)} (§CC.1 was 15/15; now 18/18 with M2 F2)")
    print("\n=== M2 F2 rows (the new ones) ===")
    for r in doc["summary"]["F2_rows"]:
        print(f"  seed {r['seed']}: peak {r['peak_epoch']}, {r['n_checkpoints']} ckpts, "
              f"bundle==peak {r['bundle_equals_peak']}, sha {r['bundle_sha256']}")
    print("\n=== negative control: bundle must NOT equal a non-peak checkpoint ===")
    ex = nc["example_seed_20260802"]
    if ex:
        print(f"  seed 20260802: peak {ex['peak']} of {ex['n_checkpoints']} ckpts; "
              f"bundle==peak {ex['bundle_equals_peak']}; bundle==epoch-{ex['non_peak_control_epoch']} "
              f"{ex['non_peak_equals_bundle']} (must be False)")
    print(f"  controls: {nc['n_controls_correctly_mismatched']}/{nc['n_runs_with_a_non_peak_control']} "
          f"correctly mismatched; all_mismatch={nc['all_controls_mismatch']}")
    print(f"\nALL_PASS: {doc['ALL_PASS']}")
    assert doc["ALL_PASS"], "a check or control failed"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
