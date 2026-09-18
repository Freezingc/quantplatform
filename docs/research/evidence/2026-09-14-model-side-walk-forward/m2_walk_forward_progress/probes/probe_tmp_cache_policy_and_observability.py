#!/usr/bin/env python3
"""Read-only probe for §CM: three things, each of which sharpens a claim this pack already published.

1. WHAT ACTUALLY CLEARS /tmp/walk_forward_cache_v1. §CL.7 listed "the cache files are deleted before
   M2 reaches F3/F4" as one of only two ways A15's row counts could stop holding, and said
   /tmp/walk_forward_cache_v1 "is not persistent storage", which invites reading the risk as a
   function of elapsed time. It is not. The policy line in /usr/lib/tmpfiles.d/tmp.conf is
   `D /tmp 1777 root root -`: type D, whose man-page gloss is "directory to create and REMOVE",
   against type d's "create and CLEANUP"; and the age field is `-`, which means no age-based cleanup.
   The boot unit runs `systemd-tmpfiles --create --remove --boot`, so /tmp is emptied at BOOT, while
   the daily systemd-tmpfiles-clean.timer runs only `--clean`, which with age `-` deletes nothing here.
   Both halves are measured, not read off documentation alone: last boot was 2026-08-11 15:41:08 and
   ZERO /tmp top-level entries have an mtime before that instant, with 13 created inside the boot hour
   -- a sharp cutoff, exactly what an empty-at-boot policy produces; and the six F3/F4 cache files
   written 09-08 have survived 10.7 days spanning ~10 daily clean cycles. So the risk is "the machine
   reboots", not "time passes" or "someone cleans /tmp on a schedule".

2. WHY A CACHE DELETION WITHOUT A RESTART IS LESS DANGEROUS THAN §CL.7(b) IMPLIED. filter_current
   (walk_forward_common.py:150-155) keeps rows whose data_version EQUALS the pinned expectation, and
   the versions dict is the one pinned at startup. current_versions (:134-148) sorts by published_at
   and lets later rows overwrite earlier ones, i.e. it selects the NEWEST published version per
   logical partition. So a cold re-assembly inside the SAME process re-filters by the SAME pinned
   versions, and reproduces the same rows provided the database still holds those versions.
   §CE's universe_versions_origin_20days.json shows the database retains rather than replaces: on all
   20 days it examined, 2 or 4 universe_daily versions coexist, and on 0 of 20 days does the
   lexicographic max equal the newest by created_at. So version retention is evidenced -- but only
   for universe_daily and only for those 20 days, which is why this probe records the evidence base
   and does not generalise it to factor_daily / label_daily across all training years.

3. THE OBSERVABILITY CEILING THAT EVERY TRAINING-DYNAMICS STATISTIC IN THIS PACK SITS UNDER, plus a
   fourth candidate excluded for §CK.8's open question. trainer.py:110-116 writes a checkpoint only
   on strict improvement, so the epochs a run actually executed after its peak leave no trace. Over
   M1's 336 runs only 24.4% of executed epochs (peak + patience, per §CJ.2) correspond to a saved
   checkpoint, and 24 runs have exactly ONE checkpoint -- for those, everything after epoch 1 is
   invisible. M2 is better at 43.8% because its peaks are larger. In 67.0% of M1's grid runs the number of
   saved checkpoints is strictly LESS than the peak epoch, which is why the numerator must be the
   checkpoint count and not the peak -- the first version of this probe made exactly that error and
   reported 49.7% for M1 instead of 24.4%.
   §CK.8 left open WHY M1's epochs/run is 20.12 -> 23.27 -> 19.09 -> 16.96, having excluded three
   candidates (config mix, panel size, validation signal strength). This probe excludes a fourth:
   "an influx of degenerate peak==1 runs". F3 and F4 do have more of them (9.9% and 11.1% versus
   3.7% on F1/F2), but holding that share at F1's level removes only 7.9% of the F2->F4 drop
   (6.309 -> 5.812). The real change is a whole-distribution leftward shift: the MEDIAN collapses
   from 15 (F2) to 3 (F3) and 4 (F4), and the peak<=2 share rises from 17.3% to 37.0% / 28.4%, while
   the trimmed means move far less than the raw means. A median of 3 on F3/F4 against 11/15 on F1/F2
   is not a mixture artefact, so the open question stands but with one more candidate removed.

ATIME DISCLOSURE. Reading a parquet footer updates the file's atime under this filesystem's
`relatime` mount option, so the probes in §CJ, §CK and §CL did change the atime of the 12 cache files
they inspected (all six F1/F2 and all six F3/F4), setting them to 2026-09-18 20:17:44. Content is
untouched: no cache file has an mtime after 09-08, zero were modified after 09-09, and the row counts
and column names re-read here still match every previously published value. Nothing in the pipeline
depends on atime -- the extractor tests cache_path.exists() and calls pd.read_parquet, and with age
`-` the cleaner consults no timestamp at all -- but a safety declaration saying "read-only" should
state that reading files changes their atime, because that is a change this pack caused to files
outside its own directory. Recorded here rather than left implicit.

Reads: /usr/lib/tmpfiles.d/tmp.conf, systemd unit files, /proc, /tmp and dataset parquet FOOTERS
(metadata.num_rows, metadata.schema.names -- no row data), manifest.json text, m1_full_v1.log,
checkpoint directory names, and one existing §CE JSON. No repository module imported or executed, no
checkpoint.pt opened, no weight or tensor loaded, no database access, no network, no writes to any
artifact directory, no signal to any process.
"""
from __future__ import annotations

import glob
import json
import re
import statistics as st
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
CACHE = Path("/tmp/walk_forward_cache_v1")
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "measurements" / "tmp_cache_policy_and_observability.json"

PATIENCE = 10
# the 12 cache files identified by §CL's validated key function (6 log-proven + 6 predicted)
CACHE_FILES = {
    "F1/train": "3a00d9a33131741a34cc4880b1c7b5e60b724d9af8f91e4beadde245fb9aaace",
    "F1/validation": "c258791e203f46bf5bdfdaca7825bf6295851afadc423274c7e5ded6c2b93e38",
    "F1/test": "0c14f9b161aabfa8d62d7c46fd2d6c206c05e43f39af03e77249ed2f8ba8d03f",
    "F2/train": "2e64e81ea5877c5f92c14373870d4890a1e532230d3c9de1e735fab312407a25",
    "F2/validation": "60278a675f1e244ae103a79fe22f42c229a5eb2e2c46e6812b29f3d63b64bc93",
    "F2/test": "ae985cf2de4984e34cf22a422c31a7bb72d9a327ef6f802b7342b0b1bdf52efb",
    "F3/train": "1546b0cfb48b1038aba185c9abb40d2564215837de1d263abde636f5ecb5dc37",
    "F3/validation": "a3f645bcf8d914295403cc558228eb530ae7e4f61782a59f290cf96b348e807e",
    "F3/test": "6350b5bf7012f8499d284c561c9dbd4cdc9b4e0d48bed889fdbb0ef1ac66187a",
    "F4/train": "5ec22abc8b0bd0adc0d7a7a6a01dce0d7e2ae8e23188511f1eff4643a8cd098a",
    "F4/validation": "f7f96c8e66fd5198d398769879b48692e9da874d0b9505ab6163a4bf239911ac",
    "F4/test": "363c44fb164b07bb6889f9e604e57b74f2cfd75c6a28ddcda499f6076dad9667",
}
assert all(len(k) == 64 and all(c in "0123456789abcdef" for c in k)
           for k in CACHE_FILES.values()), (
    "a cache key is not a full 64-hex sha256; the first version of this probe carried a 50-character "
    "mangled key for F2/train, so that file was reported missing and -- because the aggregate check "
    "was all(... if exists) -- silently skipped, letting 'all rows match published' come out True on "
    "11 of 12 files. A vacuous truth is worse than a failure.")

EXPECTED_ROWS = {
    "F1/train": 2296573, "F1/validation": 696906, "F1/test": 804575,
    "F2/train": 2539884, "F2/validation": 783477, "F2/test": 882426,
    "F3/train": 2790995, "F3/validation": 859748, "F3/test": 932167,
    "F4/train": 3046399, "F4/validation": 908649, "F4/test": 938594,
}


def sh(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True).stdout.strip()


def tmp_policy() -> dict:
    """What clears /tmp, from the policy file, the unit files, and the observed mtime cutoff."""
    conf = Path("/usr/lib/tmpfiles.d/tmp.conf").read_text(encoding="utf-8")
    line = next(l for l in conf.splitlines()
                if l.strip() and not l.strip().startswith("#") and l.split()[1] == "/tmp")
    parts = line.split()
    # /proc/stat's btime, NOT `uptime -s`. The first version of this probe used `uptime -s`, which
    # is derived as (now - uptime_seconds) and therefore drifts by a second between calls: two runs
    # of this probe reported 15:41:07 and 15:41:08 for the same boot, which made the measurement
    # file non-reproducible and would have failed this pack's cwd-independence convention for a
    # reason that has nothing to do with cwd. btime is the kernel's recorded boot epoch and is
    # stable across calls (verified: three consecutive reads all return 1786434067).
    boot_epoch = int(re.search(r"^btime (\d+)", Path("/proc/stat").read_text(), re.M).group(1))
    boot_dt = datetime.fromtimestamp(boot_epoch)
    boot = boot_dt.isoformat(sep=" ")
    before = sh("find", "/tmp", "-maxdepth", "1", "!", "-newermt", boot)
    hour_start = boot_dt.replace(minute=0, second=0, microsecond=0)
    hour_end = hour_start.replace(minute=59, second=59)
    in_boot_hour = sh("find", "/tmp", "-maxdepth", "1", "-newermt", hour_start.isoformat(sep=" "),
                      "!", "-newermt", hour_end.isoformat(sep=" "))
    clean_exec = sh("systemctl", "cat", "systemd-tmpfiles-clean.service")
    setup_exec = sh("systemctl", "cat", "systemd-tmpfiles-setup.service")
    timer = sh("systemctl", "cat", "systemd-tmpfiles-clean.timer")
    return {
        "policy_line": line,
        "type": parts[0],
        "policy_line_fields": {
            "type": parts[0], "path": parts[1], "mode": parts[2], "user": parts[3],
            "group": parts[4], "age": parts[5] if len(parts) > 5 else None,
            "note": ("tmpfiles.d columns are Type Path Mode User Group Age Argument, so the age field "
                     "is index 5 and NOT index 4. The first version of this probe read parts[4] and "
                     "published age='root', which is the group column -- a wrong value in the one "
                     "field the whole conclusion turns on, since age '-' is what makes the daily "
                     "--clean a no-op here."),
        },
        "type_man_gloss_D": "directory to create and REMOVE",
        "type_man_gloss_d": "directory to create and CLEANUP",
        "age_field": parts[5] if len(parts) > 5 else None,
        "age_field_meaning": ("'-' means no age-based cleanup, so systemd-tmpfiles --clean deletes "
                              "nothing here on account of age"),
        "clean_unit_execstart": next((l.split("=", 1)[1] for l in clean_exec.splitlines()
                                     if l.startswith("ExecStart=")), None),
        "setup_unit_execstart": next((l.split("=", 1)[1] for l in setup_exec.splitlines()
                                     if l.startswith("ExecStart=")), None),
        "timer_schedule": [l.split("=", 1)[1] for l in timer.splitlines()
                           if l.startswith(("OnBootSec=", "OnUnitActiveSec="))],
        "last_boot": boot,
        "last_boot_source": "/proc/stat btime (stable across calls; `uptime -s` is not)",
        "last_boot_epoch": boot_epoch,
        "tmp_entries_older_than_last_boot": len([x for x in before.split("\n") if x]),
        "tmp_entries_created_in_boot_hour": len([x for x in in_boot_hour.split("\n") if x]),
        "reading": ("a sharp cutoff at the boot instant is what an empty-at-boot policy produces; "
                    "zero survivors older than boot is the evidence, and the daily --clean timer "
                    "cannot be what clears /tmp because the age field is '-'"),
    }


def cache_survival() -> dict:
    """How long the F3/F4 cache files have survived the daily cleaner, and their atime state."""
    now = datetime.now()
    rows = {}
    for name, key in CACHE_FILES.items():
        p = CACHE / f"{key}.parquet"
        if not p.is_file():
            rows[name] = {"exists": False}
            continue
        s = p.stat()
        mt = datetime.fromtimestamp(s.st_mtime)
        at = datetime.fromtimestamp(s.st_atime)
        rows[name] = {
            "exists": True,
            "rows": pq.ParquetFile(p).metadata.num_rows,
            "rows_match_published": pq.ParquetFile(p).metadata.num_rows == EXPECTED_ROWS[name],
            "mtime": mt.isoformat(timespec="seconds"),
            "atime": at.isoformat(timespec="seconds"),
            "days_survived": round((now - mt).total_seconds() / 86400, 2),
            "daily_clean_cycles_survived": int((now - mt).total_seconds() // 86400),
            "atime_changed_by_this_pack_probes": at.date() == datetime(2026, 9, 18).date(),
        }
    n_mod = len([f for f in glob.glob(str(CACHE / "*.parquet"))
                 if datetime.fromtimestamp(Path(f).stat().st_mtime) > datetime(2026, 9, 9)])
    return {
        "per_file": rows,
        "n_cache_files_total": len(glob.glob(str(CACHE / "*.parquet"))),
        "cache_bytes_total": sum(f.stat().st_size for f in CACHE.glob("*.parquet")),
        "n_files_modified_after_2026_09_09": n_mod,
        # NOT `all(... if exists)`: a missing file must make this False, not be skipped. That
        # conditional form is exactly what let the mangled key above pass as "all match".
        "all_expected_rows_still_match": (
            len(rows) == len(CACHE_FILES)
            and all(r.get("exists") and r.get("rows_match_published") for r in rows.values())),
        "n_files_found": sum(1 for r in rows.values() if r.get("exists")),
        "n_files_expected": len(CACHE_FILES),
        "atime_disclosure": ("reading a parquet footer updates atime under this filesystem's relatime "
                             "mount option, so the §CJ/§CK/§CL probes changed the atime of the 12 "
                             "files they inspected. mtime is unchanged on all 46 files and none was "
                             "modified after 09-09, so content is intact; the extractor depends only "
                             "on path existence and file content, and with age '-' the cleaner "
                             "consults no timestamp, so nothing in the pipeline reads atime."),
        "mount_options_root": sh("findmnt", "-no", "OPTIONS", "/"),
    }


def peaks(ws: str, car: str) -> dict:
    run_re = re.compile(r"^(?P<run>.+)-epoch-(?P<epoch>\d+)$")
    runs: dict[str, list[int]] = defaultdict(list)
    for d in glob.glob(str(ART / ws / "checkpoints" / "*-epoch-*")):
        m = run_re.match(Path(d).name)
        if m:
            runs[m.group("run")].append(int(m.group("epoch")))
    out: dict[str, list[int]] = defaultdict(list)
    for r, e in runs.items():
        g = re.match(rf"^{car}-(f\d)-c\d+-\d+-seed-\d+$", r)
        if g:
            out[g.group(1)].append(max(e))
    return {k: sorted(v) for k, v in sorted(out.items())}


def peaks_and_checkpoint_counts(ws: str, car: str) -> dict:
    """{fold: [(peak_epoch, n_checkpoints_saved), ...]} for grid runs.

    The two quantities are NOT the same, and conflating them was the second error in the first
    version of this probe. A checkpoint is written only on strict improvement
    (trainer.py:110-116), so the saved epochs form a SUBSET of 1..peak; n_checkpoints <= peak, with
    equality only when every epoch up to the peak improved. The observable epochs are the saved ones,
    so observability is sum(n_checkpoints) / sum(peak + patience) -- using peak in the numerator
    instead inflates it badly (49.7% reported versus 24.4% actual on M1).
    """
    run_re = re.compile(r"^(?P<run>.+)-epoch-(?P<epoch>\d+)$")
    runs: dict[str, list[int]] = defaultdict(list)
    for d in glob.glob(str(ART / ws / "checkpoints" / "*-epoch-*")):
        m = run_re.match(Path(d).name)
        if m:
            runs[m.group("run")].append(int(m.group("epoch")))
    out: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for r, e in runs.items():
        g = re.match(rf"^{car}-(f\d)-c\d+-\d+-seed-\d+$", r)
        if g:
            out[g.group(1)].append((max(e), len(set(e))))
    return {k: sorted(v) for k, v in sorted(out.items())}


def observability() -> dict:
    doc = {}
    for label, ws, car in (("M1", "m1_full_v1", "m1"), ("M2", "m2_v1", "m2")):
        pk = peaks_and_checkpoint_counts(ws, car)
        per_fold = {}
        allp: list[tuple[int, int]] = []
        for f, pairs in pk.items():
            v = [p for p, _ in pairs]
            allp.extend(pairs)
            nck = [n for _, n in pairs]
            per_fold[f] = {
                "n_runs": len(v),
                "mean_peak": round(st.mean(v), 4),
                "median_peak": st.median(v),
                "p25": v[len(v) // 4], "p75": v[3 * len(v) // 4],
                "p90": v[min(len(v) - 1, int(0.90 * len(v)))],
                "max_peak": max(v),
                "trimmed_mean_10pct": round(st.mean(v[len(v) // 10: len(v) - len(v) // 10]), 4),
                "trimmed_mean_25pct": round(st.mean(v[len(v) // 4: len(v) - len(v) // 4]), 4),
                "n_peak_ge_20": sum(1 for x in v if x >= 20),
                "bimodal_signature": (
                    "both tails heavy" if (sum(1 for x in v if x <= 2) >= len(v) * 0.25
                                           and sum(1 for x in v if x >= 20) >= len(v) * 0.10)
                    else "upper tail heavy" if sum(1 for x in v if x >= 20) >= len(v) * 0.10
                    else "lower tail heavy" if sum(1 for x in v if x <= 2) >= len(v) * 0.25
                    else "neither tail dominant"),
                "n_peak_eq_1": sum(1 for x in v if x == 1),
                "share_peak_eq_1_pct": round(100 * sum(1 for x in v if x == 1) / len(v), 1),
                "n_peak_le_2": sum(1 for x in v if x <= 2),
                "share_peak_le_2_pct": round(100 * sum(1 for x in v if x <= 2) / len(v), 1),
                "mean_checkpoints_saved": round(st.mean(nck), 4),
                "n_checkpoints_equals_peak_in_runs": sum(1 for p, n in pairs if p == n),
                "executed_epochs_lower_bound": sum(x + PATIENCE for x in v),
                "observable_epochs": sum(nck),
                "observability_pct": round(100 * sum(nck) / sum(x + PATIENCE for x in v), 1),
            }
        doc[label] = {
            "per_fold": per_fold,
            "scope": "grid runs only (final-refit runs excluded, matching §CJ's phase split)",
            "n_runs": len(allp),
            "observability_pct": round(
                100 * sum(n for _, n in allp) / sum(p + PATIENCE for p, _ in allp), 1),
            "n_runs_with_exactly_one_checkpoint": sum(1 for _, n in allp if n == 1),
            "n_runs_where_peak_eq_1": sum(1 for p, _ in allp if p == 1),
        }
    doc["reading"] = (
        "a checkpoint is written only on strict improvement (trainer.py:110-116), so every epoch a run "
        "executed AFTER its peak leaves no trace. The observable fraction of executed epochs is this "
        "low, and runs with exactly one checkpoint have nothing but epoch 1 on the record. Every "
        "training-dynamics statistic in this pack (§BO.9's peak spread, §CA/§CB's peak-basis variance "
        "decomposition, §CI's epochs/run trend, §CJ's cap scan, §CK's four-fold pattern) is computed "
        "from peaks, which is the right quantity for 'where did it converge' but carries no "
        "information about the shape of the decline after the peak.")
    return doc


def negative_controls_and_invariants() -> dict:
    """Every instrument here must be able to say NO, per this pack's convention.

    Three checks, two of which are invariants the trainer's own logic guarantees and one of which is
    a synthetic case the instrument must reject:

      (i)   n_checkpoints <= peak for EVERY run. The trainer saves a checkpoint only on strict
            improvement (trainer.py:110-116), so the saved epochs are a subset of 1..peak. A run
            violating this would mean the checkpoint set is not what this pack assumes it is.
      (ii)  peak == 1 if and only if the run has exactly one checkpoint. Necessarily true under the
            same logic: epoch 1 always improves on best_value = -inf, and if peak is 1 then no later
            epoch improved, so nothing else was saved. This is a free consistency check between two
            independently computed quantities.
      (iii) the instrument must be able to REJECT an impossible input rather than quietly reporting a
            number for it. A synthetic run claiming peak 5 with 9 saved checkpoints is impossible under
            the trainer's logic (9 > 5), and the guard flags it. Note what this does and does not show:
            the resulting "observability" would read 60%, a perfectly ordinary-looking percentage, NOT
            something over 100% -- so an out-of-range value would not have betrayed the bad input.
            That is precisely why the guard is on n_ck <= peak rather than on the ratio.

    Without (iii) the observability percentages would be indistinguishable from an instrument that
    divides by the wrong denominator -- which is exactly the bug the first version of this probe had,
    when it used peak as the numerator and reported 49.7% instead of 24.4% on M1. That bug also
    produced no error and no implausible-looking number; it was caught only by comparing against an
    independently computed figure.

    A THIRD bug of the same family was also caught and is guarded against: the first version carried a
    50-character mangled key for F2/train, so that one file read as missing, and because the aggregate
    check was written all(... if exists) the missing file was SKIPPED, letting "all rows match
    published" come out True on 11 of 12 files. A vacuous truth is worse than a failure. Two guards
    now close it: an assert that every cache key is a full 64-hex sha256, and the aggregate rewritten
    so a missing file makes it False rather than being skipped. Both guards were validated in the
    direction that proves they can fail -- injecting the truncated key aborts on the assert, and
    pointing one key at a well-formed but nonexistent hash drops the count to 11/12 and flips the
    aggregate to False -- before either guard was trusted.
    """
    viol_i: list[str] = []
    viol_ii: list[str] = []
    n_runs = 0
    for label, ws, car in (("M1", "m1_full_v1", "m1"), ("M2", "m2_v1", "m2")):
        for fold, pairs in peaks_and_checkpoint_counts(ws, car).items():
            for peak, nck in pairs:
                n_runs += 1
                if nck > peak:
                    viol_i.append(f"{label}/{fold} peak={peak} n_ck={nck}")
                if (peak == 1) != (nck == 1):
                    viol_ii.append(f"{label}/{fold} peak={peak} n_ck={nck}")

    # (iii) synthetic impossible case
    synthetic = {"peak": 5, "n_checkpoints": 9}
    rejected = synthetic["n_checkpoints"] > synthetic["peak"]
    # and the observable fraction it would imply, showing why the guard matters
    impossible_observability_pct = round(100 * synthetic["n_checkpoints"]
                                         / (synthetic["peak"] + PATIENCE), 1)

    return {
        "invariant_i_n_checkpoints_le_peak": {
            "violations": viol_i, "n_runs_checked": n_runs, "holds": not viol_i},
        "invariant_ii_peak_eq_1_iff_one_checkpoint": {
            "violations": viol_ii, "n_runs_checked": n_runs, "holds": not viol_ii},
        "synthetic_impossible_run": {
            "peak": synthetic["peak"], "n_checkpoints": synthetic["n_checkpoints"],
            "rejected_by_guard": rejected,
            "observability_it_would_have_reported_pct": impossible_observability_pct,
            "reading": ("the guard is on n_ck <= peak, NOT on the ratio: this impossible input would "
                        "have produced 60%, an ordinary-looking number, so an out-of-range check would "
                        "not have caught it. The ratio itself is uninformative about the bug."),
        },
        "ALL_CONTROLS_BEHAVE_AS_EXPECTED": (not viol_i) and (not viol_ii) and rejected,
    }


def fourth_candidate_excluded() -> dict:
    """Is M1's F2->F4 peak-mean drop an influx of degenerate peak==1 runs?"""
    pk = peaks("m1_full_v1", "m1")
    s = {f: {"mean": st.mean(v), "share1": sum(1 for x in v if x == 1) / len(v),
             "mean_ex1": st.mean([x for x in v if x > 1]), "median": st.median(v)}
         for f, v in pk.items()}
    base_share = s["f1"]["share1"]
    cf = {f: base_share * 1 + (1 - base_share) * d["mean_ex1"] for f, d in s.items()}
    total_drop = s["f2"]["mean"] - s["f4"]["mean"]
    cf_drop = cf["f2"] - cf["f4"]
    return {
        "per_fold": {f: {"mean_peak": round(d["mean"], 4),
                         "share_peak_eq_1_pct": round(d["share1"] * 100, 1),
                         "mean_peak_excluding_peak_eq_1": round(d["mean_ex1"], 4),
                         "median_peak": d["median"],
                         "counterfactual_mean_at_F1_peak1_share": round(cf[f], 4)}
                     for f, d in s.items()},
        "F1_share_peak_eq_1_pct": round(base_share * 100, 2),
        "total_F2_to_F4_drop": round(total_drop, 4),
        "drop_with_peak1_share_held_at_F1_level": round(cf_drop, 4),
        "share_of_drop_explained_by_peak1_influx_pct": round(
            (total_drop - cf_drop) / total_drop * 100, 1),
        "median_collapse": {"f1": s["f1"]["median"], "f2": s["f2"]["median"],
                            "f3": s["f3"]["median"], "f4": s["f4"]["median"]},
        "order_flip_between_mean_and_central_mass": {
            "what": ("does F3 or F4 have the longer TYPICAL run? The mean and the central mass "
                     "disagree, so any comparison of these two folds must state its basis"),
            "by_mean": "F3 > F4",
            "by_median": "F3 < F4",
            "by_trimmed_25pct": "F3 < F4",
            "flipped": (s["f3"]["mean"] > s["f4"]["mean"]) != (s["f3"]["median"] > s["f4"]["median"]),
            "means": {"f3": round(s["f3"]["mean"], 4), "f4": round(s["f4"]["mean"], 4)},
            "medians": {"f3": s["f3"]["median"], "f4": s["f4"]["median"]},
            "trimmed_25pct": {
                "f1": round(st.mean(sorted(pk["f1"])[len(pk["f1"]) // 4:
                                                      len(pk["f1"]) - len(pk["f1"]) // 4]), 4),
                "f2": round(st.mean(sorted(pk["f2"])[len(pk["f2"]) // 4:
                                                      len(pk["f2"]) - len(pk["f2"]) // 4]), 4),
                "f3": round(st.mean(sorted(pk["f3"])[len(pk["f3"]) // 4:
                                                      len(pk["f3"]) - len(pk["f3"]) // 4]), 4),
                "f4": round(st.mean(sorted(pk["f4"])[len(pk["f4"]) // 4:
                                                      len(pk["f4"]) - len(pk["f4"]) // 4]), 4)},
            "cause": ("F3 is bimodal (30 runs at peak<=2 AND 13 at peak>=20, p90 31, max 33) while "
                      "F4 is uniformly low (23 at peak<=2 but only 2 at peak>=20, p90 15, max 20), "
                      "so F3's mean is lifted by its upper tail while its central mass sits below "
                      "F4's"),
            "consequence": ("§CI's 1.647x growth and §CK.2's 'non-monotone, peaking at F2' are both "
                            "mean-basis statements and both hold on that basis, but neither discloses "
                            "that F3 and F4 swap places on the median basis"),
        },
        "verdict": ("REJECTED as the explanation: holding the peak==1 share at F1's 3.7% removes only "
                    "7.9% of the drop. The real change is a whole-distribution leftward shift -- the "
                    "median falls 15 -> 3 -> 4 while the 25%-trimmed mean falls only 12.76 -> 4.76 -> "
                    "5.49 and the raw mean 13.27 -> 9.09 -> 6.96. So §CK.8's open question loses a "
                    "fourth candidate (degenerate-run influx) and keeps three excluded (config mix, "
                    "panel size, validation signal strength), with no cause established."),
    }


def version_retention_evidence() -> dict:
    """Does the database retain old versions (so a cold re-assembly under pinned versions still works)?"""
    p = ART / "opt1_feasibility" / "universe_versions_origin_20days.json"
    d = json.load(open(p, encoding="utf-8"))
    nv = Counter(day["n_versions"] for day in d["per_day"])
    newest_is_lexmax = sum(1 for day in d["per_day"] if day["rule_picks_the_newest"])
    return {
        "source": str(p.relative_to(ART)),
        "table_examined": "universe_daily",
        "days_examined": len(d["per_day"]),
        "versions_coexisting_per_day": dict(sorted(nv.items())),
        "max_coexisting": max(nv),
        "days_where_lexicographic_max_is_newest": newest_is_lexmax,
        "reading": ("the database retains old versions rather than replacing them, so a cold "
                    "re-assembly inside the SAME process -- which re-filters by the SAME pinned "
                    "versions (filter_current, walk_forward_common.py:150-155) -- would reproduce the "
                    "same rows provided those versions are still present"),
        "EVIDENCE_LIMIT": ("this covers universe_daily on 20 days (2024-12-20 .. 2025-01-17) only. It "
                           "does NOT evidence retention for factor_daily or label_daily, nor for the "
                           "2017-2025 training years. So 'a cache deletion without a restart is "
                           "recoverable' is supported for one of three tables on a 20-day window and "
                           "must not be cited as a general guarantee."),
        "restart_is_the_dangerous_case": ("current_versions sorts by published_at and lets later rows "
                                          "overwrite earlier ones, i.e. it picks the NEWEST published "
                                          "version per partition. A restarted process therefore pins "
                                          "whatever is newest at the new start time, which need not "
                                          "match 09-13's snapshot -- and --resume restores search "
                                          "records, not that in-memory snapshot."),
    }


def main() -> None:
    doc: dict[str, object] = {
        "question": ("what actually clears the fold cache that §CL's conclusion depends on, how long "
                     "have the F3/F4 cache files already survived it, what fraction of executed "
                     "epochs is observable at all, and is M1's fold-wise epoch drop an artefact of "
                     "degenerate runs?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "tmp_policy": tmp_policy(),
        "cache_survival": cache_survival(),
        "version_retention": version_retention_evidence(),
        "observability": observability(),
        "fourth_candidate_excluded": fourth_candidate_excluded(),
        "negative_controls_and_invariants": negative_controls_and_invariants(),
    }
    tp = doc["tmp_policy"]
    cs = doc["cache_survival"]
    f34 = [v["daily_clean_cycles_survived"] for k, v in cs["per_file"].items()
           if k.startswith(("F3", "F4")) and v.get("exists")]
    doc["conclusion"] = {
        "what_clears_the_cache": ("a BOOT, not elapsed time and not the daily cleaner: tmp.conf uses "
                                  f"type {tp['type']} (create-and-REMOVE) with age '{tp['age_field']}' "
                                  "(no age-based cleanup), and the boot unit runs --create --remove "
                                  "--boot while the daily timer runs only --clean"),
        "evidence_of_boot_clearing": (
            f"{tp['tmp_entries_older_than_last_boot']} /tmp entries predate the last boot "
            f"({tp['last_boot']}), with {tp['tmp_entries_created_in_boot_hour']} created inside the "
            "boot hour -- a sharp cutoff"),
        "clean_cycles_the_F3F4_files_survived": min(f34) if f34 else None,
        "all_12_cache_rows_still_match_published": cs["all_expected_rows_still_match"],
        "no_cache_file_modified_after_2026_09_09": cs["n_files_modified_after_2026_09_09"] == 0,
        "so_CL7b_should_read": ("'the cache is deleted' means 'the machine reboots' (or a manual rm), "
                                "not 'time passes' or 'the daily cleaner runs'. Over the ~2-3 weeks "
                                "M2 still needs, the operative question is whether this host reboots, "
                                "which is a much more specific thing to watch than /tmp hygiene."),
        "observability_of_executed_epochs_pct": {
            "M1": doc["observability"]["M1"]["observability_pct"],
            "M2": doc["observability"]["M2"]["observability_pct"]},
        "runs_with_a_single_checkpoint": {
            "M1": doc["observability"]["M1"]["n_runs_with_exactly_one_checkpoint"],
            "M2": doc["observability"]["M2"]["n_runs_with_exactly_one_checkpoint"]},
        "fourth_candidate_for_CK8": "rejected (explains only "
            f"{doc['fourth_candidate_excluded']['share_of_drop_explained_by_peak1_influx_pct']}% "
            "of the F2->F4 drop)",
    }
    doc.update({"read_only": True, "database_access": False, "weights_loaded": False,
                "checkpoint_pt_opened": False, "parquet_row_data_loaded": False,
                "training_performed": False, "repository_module_imported_or_executed": False,
                "processes_signalled": [],
                "side_effect_disclosed": ("atime of the 12 inspected cache files was updated to "
                                          "2026-09-18 20:17:44 by the §CJ/§CK/§CL probes reading "
                                          "their footers; mtime and content unchanged")})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("=== /tmp policy ===")
    print(f"  line: {tp['policy_line']}")
    print(f"  type {tp['type']} ({tp['type_man_gloss_D']}) vs d ({tp['type_man_gloss_d']}); "
          f"age field {tp['age_field']!r}")
    print(f"  boot unit: {tp['setup_unit_execstart']}")
    print(f"  daily unit: {tp['clean_unit_execstart']}  schedule {tp['timer_schedule']}")
    print(f"  last boot {tp['last_boot']}: entries older = "
          f"{tp['tmp_entries_older_than_last_boot']}, created in boot hour = "
          f"{tp['tmp_entries_created_in_boot_hour']}")
    print("\n=== cache survival ===")
    print(f"  {cs['n_cache_files_total']} files, {cs['cache_bytes_total']/2**30:.2f} GiB, "
          f"modified after 09-09: {cs['n_files_modified_after_2026_09_09']}")
    print(f"  mount options for /: {cs['mount_options_root']}")
    for name in ("F1/train", "F3/train", "F4/test"):
        v = cs["per_file"][name]
        print(f"  {name:16s} rows={v['rows']:>9,} match={v['rows_match_published']} "
              f"mtime={v['mtime'][:16]} atime={v['atime'][:16]} survived={v['days_survived']} d "
              f"({v['daily_clean_cycles_survived']} clean cycles)")
    print(f"  all 12 rows still match published: {cs['all_expected_rows_still_match']}")
    print("\n=== observability (executed epochs that left a trace) ===")
    for label in ("M1", "M2"):
        o = doc["observability"][label]
        print(f"  {label}: {o['n_runs']} runs, observability {o['observability_pct']}%, "
              f"runs with exactly 1 checkpoint: {o['n_runs_with_exactly_one_checkpoint']}")
        for f, d in o["per_fold"].items():
            print(f"     {f}: mean {d['mean_peak']:6.2f} median {d['median_peak']:4.1f} "
                  f"max {d['max_peak']:3d} peak==1 {d['share_peak_eq_1_pct']:4.1f}% "
                  f"peak<=2 {d['share_peak_le_2_pct']:4.1f}% obs {d['observability_pct']:4.1f}%")
    print("\n=== fourth candidate for §CK.8's open question ===")
    fc = doc["fourth_candidate_excluded"]
    print(f"  F2->F4 drop {fc['total_F2_to_F4_drop']} ; with peak==1 share held at F1's "
          f"{fc['F1_share_peak_eq_1_pct']}% it is {fc['drop_with_peak1_share_held_at_F1_level']}"
          f" -> explained {fc['share_of_drop_explained_by_peak1_influx_pct']}%")
    print(f"  medians: {fc['median_collapse']}")
    of = fc["order_flip_between_mean_and_central_mass"]
    print(f"  ORDER FLIP F3 vs F4: by mean {of['by_mean']} ({of['means']}), by median "
          f"{of['by_median']} ({of['medians']}), by 25% trim {of['by_trimmed_25pct']} "
          f"({of['trimmed_25pct']}) -> flipped={of['flipped']}")
    print("\nconclusion:", json.dumps(doc["conclusion"], ensure_ascii=False, indent=1))
    nc = doc["negative_controls_and_invariants"]
    print("\n=== negative controls and invariants ===")
    i = nc["invariant_i_n_checkpoints_le_peak"]
    ii = nc["invariant_ii_peak_eq_1_iff_one_checkpoint"]
    print(f"  (i)   n_checkpoints <= peak over {i['n_runs_checked']} runs: "
          f"violations {len(i['violations'])} -> {'holds' if i['holds'] else 'VIOLATED'}")
    print(f"  (ii)  peak==1 iff exactly one checkpoint: violations {len(ii['violations'])} "
          f"-> {'holds' if ii['holds'] else 'VIOLATED'}")
    sy = nc["synthetic_impossible_run"]
    print(f"  (iii) synthetic peak=5 with 9 checkpoints rejected: {sy['rejected_by_guard']} "
          f"(it would have implied {sy['observability_it_would_have_reported_pct']}% observable)")
    print(f"  ALL_CONTROLS_BEHAVE_AS_EXPECTED: {nc['ALL_CONTROLS_BEHAVE_AS_EXPECTED']}")
    assert nc["ALL_CONTROLS_BEHAVE_AS_EXPECTED"], "a control or invariant failed"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
