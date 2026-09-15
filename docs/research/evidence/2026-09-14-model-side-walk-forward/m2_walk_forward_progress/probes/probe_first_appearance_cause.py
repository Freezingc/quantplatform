#!/usr/bin/env python3
"""Read-only probe: WHY does the count of securities first appearing in the panel fall 468->395->257->109?

§BW.1 measured the monotone drop in "securities whose earliest panel date falls inside the test window"
(F1 468, F2 395, F3 257, F4 109) and explicitly refused to claim a cause (§BW.5: "没有查...外部成因").
Two candidate causes were named: (a) market IPO pace, (b) frozen-universe construction.

This probe separates them from the panel alone, with one external table (the BSE old->new code map,
embedded below, transcribed from tushare bse_mapping on 2026-09-16). The mechanism it tests:

  The Beijing Stock Exchange migrated every listed security from an 8xxxxx/4xxxxx code to a 920xxx code
  during 2023-2024. A frozen panel assembled in 2026 carries the CURRENT (920xxx) code. If the panel's
  920xxx series starts at the migration date rather than being backfilled to the entity's true IPO date,
  then one economic entity produces TWO "first appearances" in the panel: the 8xx code early, and the
  920xxx code in the migration year. That inflates the first-appearance count in exactly 2023-2024.

So for each fold this probe reports, among the securities first appearing inside the test window:
  - the board composition (STAR 688 / ChiNext 300,301 / SH-main 60x / SZ-main 00x / BSE-new 920 / BSE-old 8xx,4xx)
  - for every 920xxx first-appearance, whether its 8xx/4xx twin is ALSO in the panel and first appears
    EARLIER than the test window  => "migration re-entry" (universe-construction artifact), versus
    twin absent-or-not-earlier  => "genuine new BSE IPO".

The F4/2025 answer follows: if 2025's first appearances contain no migration re-entries (the migration
was complete by 2025, so every 920xxx is a genuine 2025 IPO) and the count matches the market's 2025 IPO
count, then the F4 shortfall is market IPO pace, not a universe artifact.

Safety: reads parquet artifacts in place (signal_date, ts_code only). No database access, no weight
loading, no training, no repository module imported or executed, nothing written into any artifact
directory, no process signalled. Writes only the small JSON named by --out into the pack.
"""
from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
FOLD_DATASET = {
    "F1": "226a8dfc497c2de6b07516711f058f26a9270675916c9e67dbf7f087a07d2538",
    "F2": "eb9bba1b5a0bd66d0ad6c82536b295de1fc079aec125cb27bb68491bdb21a234",
    "F3": "9766e69f342511c5bda30188df15a7b9b17d936bdc26b9b60ba244e8e7e0c9e7",
    "F4": "6f9478d7645c8c745a73f00c03b6122e17d0c06c05cc159ece531fb00df9fb9d",
}
LABEL_WORKSPACE = "m1_full_v1"

# BSE old(8xxxxx/4xxxxx)->new(920xxx) code map, transcribed from tushare `bse_mapping` 2026-09-16.
# Stored as "old:new" pairs; inverted below to new->old.
_BSE_PAIRS = """
430017:920017 430047:920047 430090:920090 430139:920139 430198:920198 430300:920300 430418:920418
430425:920425 430476:920476 430478:920478 430489:920489 430510:920510 430556:920556 430564:920564
430685:920685 430718:920718 830779:920779 830799:920799 830809:920809 830832:920832 830839:920839
830879:920879 830896:920896 830946:920946 830964:920964 830974:920974 831010:920010 831039:920039
831087:920087 831152:920152 831167:920267 831175:920375 831195:920195 831278:920378 831304:920304
831305:920405 831370:920370 831396:920496 831445:920445 831526:920526 831627:920627 831641:920641
831689:920689 831726:920726 831768:920768 831832:920932 831834:920834 831855:920855 831856:920856
831906:920906 831961:920961 832000:920000 832023:920023 832089:920089 832110:920110 832145:920145
832149:920249 832171:920171 832175:920175 832225:920225 832278:920278 832419:920419 832469:920469
832471:920471 832491:920491 832522:920522 832566:920566 832651:920651 832662:920662 832735:920735
832786:920786 832802:920802 832876:920876 832885:920885 832978:920978 832982:920982 833030:920130
833075:920075 833171:920571 833230:920230 833266:920266 833284:920284 833346:920346 833394:920394
833427:920427 833429:920429 833454:920454 833455:920455 833509:920509 833523:920523 833533:920533
833575:920575 833580:920580 833751:920751 833781:920781 833819:920819 833873:920873 833914:920914
833943:920943 834014:920014 834021:920021 834033:920033 834058:920058 834062:920062 834261:920261
834407:920407 834415:920415 834475:920475 834599:920599 834639:920639 834682:920682 834765:920765
834770:920770 834950:920950 835174:920174 835179:920179 835184:920184 835185:920185 835207:920207
835237:920237 835305:920305 835368:920368 835438:920438 835508:920508 835579:920579 835640:920640
835670:920670 835857:920057 835892:920992 835985:920985 836077:920077 836149:920149 836208:920208
836221:920221 836239:920239 836247:920247 836260:920260 836263:920263 836270:920270 836395:920395
836414:920414 836419:920519 836422:920422 836433:920433 836504:920504 836547:920547 836675:920675
836699:920699 836717:920717 836720:920720 836807:920807 836826:920826 836871:920871 836892:920892
836942:920942 836957:920957 836961:920061 837006:920006 837023:920123 837046:920046 837092:920092
837174:920274 837212:920212 837242:920242 837344:920344 837403:920403 837592:920592 837663:920663
837748:920748 837821:920821 838030:920030 838163:920163 838171:920271 838227:920227 838262:920262
838275:920275 838402:920402 838670:920870 838701:920701 838810:920810 838837:920837 838924:920924
838971:920971 839167:920167 839273:920273 839371:920371 839493:920493 839680:920680 839719:920719
839725:920725 839729:920729 839790:920790 839792:920792 839946:920146 870199:920199 870204:920204
870299:920299 870357:920357 870436:920436 870508:920608 870656:920656 870726:920926 870866:920866
870976:920976 871245:920245 871263:920363 871396:920396 871478:920578 871553:920553 871634:920634
871642:920642 871694:920694 871753:920753 871857:920857 871970:920970 871981:920981 872190:920190
872351:920351 872374:920374 872392:920392 872541:920541 872808:920808 872895:920895 872925:920925
872931:920931 872953:920953 873001:920001 873122:920122 873132:920132 873152:920252 873167:920367
873169:920169 873223:920223 873305:920505 873339:920339 873527:920527 873570:920570 873576:920576
873593:920593 873665:920665 873679:920679 873690:920690 873693:920693 873703:920703 873706:920706
873726:920026 873806:920806 873833:920833
"""
NEW_TO_OLD = {}
OLD_TO_NEW = {}
for _p in _BSE_PAIRS.split():
    _o, _n = _p.split(":")
    NEW_TO_OLD[_n + ".BJ"] = _o + ".BJ"
    OLD_TO_NEW[_o + ".BJ"] = _n + ".BJ"


def board_of(code: str) -> str:
    num, _, ex = code.partition(".")
    if ex == "BJ":
        return "BSE-new(920)" if num.startswith("920") else "BSE-old(8xx/4xx)"
    if ex == "SH":
        if num.startswith("688"):
            return "STAR(688)"
        if num.startswith(("600", "601", "603", "605")):
            return "SH-main(60x)"
        return "SH-other"
    if ex == "SZ":
        if num.startswith(("300", "301")):
            return "ChiNext(30x)"
        if num.startswith(("000", "001", "002", "003")):
            return "SZ-main(00x)"
        return "SZ-other"
    return "other"


def load_first_seen(fold: str):
    parts = []
    for split in ("train", "validation", "test"):
        paths = sorted(glob.glob(str(ART / LABEL_WORKSPACE / "datasets" / FOLD_DATASET[fold]
                                     / f"split={split}" / "*" / "*.parquet")))
        f = pd.concat([pd.read_parquet(p, columns=["signal_date", "ts_code"]) for p in paths],
                      ignore_index=True)
        f["split"] = split
        parts.append(f)
    p = pd.concat(parts, ignore_index=True)
    p["signal_date"] = pd.to_datetime(p["signal_date"], errors="raise").dt.date
    first = p.groupby("ts_code")["signal_date"].min()
    t = p.loc[p["split"] == "test"]
    test_start, test_end = min(t["signal_date"]), max(t["signal_date"])
    return first, test_start, test_end, set(p["ts_code"].unique())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    per_fold = {}
    for fold in ("F1", "F2", "F3", "F4"):
        first, test_start, test_end, all_codes = load_first_seen(fold)
        new_codes = sorted(first[(first >= test_start) & (first <= test_end)].index)
        boards = {}
        for c in new_codes:
            boards[board_of(c)] = boards.get(board_of(c), 0) + 1
        # classify every 920xxx first-appearance: migration re-entry vs genuine new BSE IPO
        reentry, genuine_new, twin_not_earlier = [], [], []
        for c in new_codes:
            if board_of(c) != "BSE-new(920)":
                continue
            o = NEW_TO_OLD.get(c)
            if o is not None and o in first and first[o] < test_start:
                reentry.append({"new": c, "old": o, "old_first_seen": str(first[o]),
                                "new_first_seen": str(first[c])})
            elif o is not None and o in first:
                twin_not_earlier.append({"new": c, "old": o, "old_first_seen": str(first[o])})
            else:
                genuine_new.append(c)
        per_fold[fold] = {
            "test_window": [str(test_start), str(test_end)],
            "distinct_panel_securities": int(first.size),
            "first_appearing_in_test_window": int(len(new_codes)),
            "board_composition": boards,
            "bse_new_920_total": int(boards.get("BSE-new(920)", 0)),
            "bse_920_migration_reentry": len(reentry),
            "bse_920_genuine_new": len(genuine_new),
            "bse_920_twin_present_not_earlier": len(twin_not_earlier),
            "reentry_examples": reentry[:12],
            "first_appearing_codes": new_codes,
        }
        print(f"[{fold}] window {test_start}..{test_end}  panel {first.size}  "
              f"first-in-window {len(new_codes)}  boards {boards}", flush=True)
        print(f"       920xxx: {boards.get('BSE-new(920)',0)} total -> reentry {len(reentry)}, "
              f"genuine-new {len(genuine_new)}, twin-not-earlier {len(twin_not_earlier)}", flush=True)

    result = {
        "question": ("Why does the count of securities first appearing inside the test window fall "
                     "468->395->257->109 across F1..F4: market IPO pace, or frozen-universe construction "
                     "(the BSE 8xx->920xxx code migration splitting one entity into two panel series)?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True, "database_access": False, "weights_loaded": False, "training_performed": False,
        "repository_module_imported_or_executed": False,
        "external_table": "tushare bse_mapping (o_code->n_code), transcribed 2026-09-16, %d pairs" % len(NEW_TO_OLD),
        "per_fold": per_fold,
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                        encoding="utf-8")
    print(f"\nwrote {args.out}")
    print("\n=== first-appearance cause, side by side ===")
    print(f"  {'fold':<5}{'window':<26}{'first-in-win':>13}{'920xxx':>8}{'reentry':>9}{'genuine':>9}")
    for f in ("F1", "F2", "F3", "F4"):
        e = per_fold[f]
        print(f"  {f:<5}{e['test_window'][0]+'..'+e['test_window'][1]:<26}"
              f"{e['first_appearing_in_test_window']:>13}{e['bse_new_920_total']:>8}"
              f"{e['bse_920_migration_reentry']:>9}{e['bse_920_genuine_new']:>9}")


if __name__ == "__main__":
    main()
