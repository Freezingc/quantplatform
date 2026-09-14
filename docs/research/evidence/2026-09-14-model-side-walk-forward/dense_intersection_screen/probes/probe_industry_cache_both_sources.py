"""Both halves of the PIT industry cache, measured: SW2014 (pre-cutover) and SW2021 (post-cutover).

`VersionedPITIndustryProvider` serves SW2014 before `PIT_CUTOVER = 2021-12-10` and SW2021 on/after
it, and the two halves are stored in DIFFERENT files: `pit_industry_members_*_SI.json` (31 files, the
SW2021 side) and `pit_industry_sw2014_members_*_SI.json` (28 files, the SW2014 side). The earlier
probe read only the first family, which was the right scope for the STAR-Market finding -- the whole
2022-2023 gap is after the cutover -- but it means the pre-2022 side of the replay's industry input
had never been measured at all.

That matters because the two factors needing an industry label (yeslab_357, yeslab_471) are thin on
exactly the same 31 trading days at the very start of the published factor_daily range
(2014-01-02..2014-02-20, coverage down to 0.6031 and 0.5756) and then fine for the rest of 2014.
yeslab_357's own replay lookback is 1 day, so its own formula cannot explain a 31-day ramp; the
shared input can. This measures whether the SW2014 membership file has a left edge at the same place.

Reads local JSON only. No database, no weights, no writes outside --out.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import pandas as pd

CACHE_DEFAULT = Path(
    "/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/artifacts/tushare_cache"
)
OUT_DEFAULT = Path("/data/zhangyudong/qrv2_research_artifacts/industry_cache_both_sources.json")
SOURCES = {
    # label -> (member file glob, classification file)
    "SW2021": ("pit_industry_members_*_SI.json", "pit_industry_sw2021_l1_classification.json"),
    "SW2014": (
        "pit_industry_sw2014_members_*_SI.json",
        "pit_industry_sw2014_l1_classification.json",
    ),
}
STAR_PREFIXES = ("688", "689")
AS_OF = (
    "2014-01-02",
    "2014-01-31",
    "2014-02-20",
    "2014-02-21",
    "2014-03-03",
    "2014-12-31",
    "2015-07-05",
    "2020-01-02",
    "2021-12-09",
    "2021-12-10",
    "2023-07-05",
    "2023-09-25",
    "2023-10-09",
)


def load(cache: Path, pattern: str) -> tuple[pd.DataFrame, dict]:
    files = sorted(glob.glob(str(cache / pattern)))
    if not files:
        raise SystemExit(f"no files match {pattern} under {cache}")
    digest = hashlib.sha256()
    frames = []
    for path in files:
        raw = Path(path).read_bytes()
        digest.update(hashlib.sha256(raw).digest())
        frame = pd.read_json(Path(path), orient="table")
        frame["source_file"] = Path(path).name
        frames.append(frame)
    members = pd.concat(frames, ignore_index=True)
    members["con_code"] = members["con_code"].astype(str)
    members["in_date"] = pd.to_datetime(members["in_date"], errors="coerce")
    members["out_parsed"] = pd.to_datetime(members["out_date"], errors="coerce", format="%Y%m%d")
    if members["out_parsed"].isna().all():
        members["out_parsed"] = pd.to_datetime(members["out_date"], errors="coerce")
    provenance = {
        "pattern": pattern,
        "files": len(files),
        "rows": int(len(members)),
        "combined_sha256": digest.hexdigest(),
    }
    return members, provenance


def active(members: pd.DataFrame, as_of: str) -> dict:
    stamp = pd.Timestamp(as_of)
    live = members.loc[members["in_date"].notna() & (members["in_date"] <= stamp)]
    live = live.loc[live["out_parsed"].isna() | (live["out_parsed"] > stamp)]
    codes = live["con_code"]
    star = codes[codes.str.startswith(STAR_PREFIXES)]
    return {
        "as_of": as_of,
        "active_members": int(codes.nunique()),
        "active_star_members": int(star.nunique()),
        "industries": int(live["index_code"].nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()

    result: dict = {"cache_dir": str(args.cache), "sources": {}}
    for label, (pattern, classification) in SOURCES.items():
        members, provenance = load(args.cache, pattern)
        star = members.loc[members["con_code"].str.startswith(STAR_PREFIXES)]
        non_star = members.loc[~members["con_code"].str.startswith(STAR_PREFIXES)]
        classification_path = args.cache / classification
        entry = {
            "provenance": provenance
            | {
                "classification_file": classification,
                "classification_exists": classification_path.is_file(),
                "classification_sha256": (
                    hashlib.sha256(classification_path.read_bytes()).hexdigest()
                    if classification_path.is_file()
                    else None
                ),
            },
            "boards": {
                "star": {
                    "rows": int(len(star)),
                    "codes": int(star["con_code"].nunique()),
                    "earliest_in_date": str(star["in_date"].min().date()) if len(star) else None,
                },
                "non_star": {
                    "rows": int(len(non_star)),
                    "codes": int(non_star["con_code"].nunique()),
                    "earliest_in_date": (
                        str(non_star["in_date"].min().date()) if len(non_star) else None
                    ),
                },
            },
            "in_date_by_month_first_six": {
                str(month): int(count)
                for month, count in list(
                    non_star.groupby(non_star["in_date"].dt.strftime("%Y-%m")).size().items()
                )[:6]
            },
            "active_membership_as_of": [active(members, day) for day in AS_OF],
        }
        result["sources"][label] = entry
        print(f"\n=== {label} ({provenance['files']} files, {provenance['rows']} rows) ===")
        print(f"  classification: {classification} exists={entry['provenance']['classification_exists']}")
        for board, stats in entry["boards"].items():
            print(f"  {board}: {stats}")
        print("  earliest non-STAR in_date months:", entry["in_date_by_month_first_six"])
        print("  active membership:")
        for row in entry["active_membership_as_of"]:
            print(
                f"    {row['as_of']}: members={row['active_members']:>5} "
                f"star={row['active_star_members']:>4} industries={row['industries']:>3}"
            )

    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
