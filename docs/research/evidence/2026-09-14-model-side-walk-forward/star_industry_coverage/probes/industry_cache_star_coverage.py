"""The shared dependency, measured: STAR-Market membership in the cached SW2021 industry file.

`materialize_frozen_factor_daily.py` replays the frozen factors from Tushare market data instead of
copying YesLab's stored values, and it singles out exactly three factors as needing a point-in-time
industry label:

    PIT_FACTORS = {"yeslab_357", "yeslab_363", "yeslab_471"}
    PIT_CUTOVER = date(2021, 12, 10)
    provider = VersionedPITIndustryProvider.from_cache(CACHE, cutover=PIT_CUTOVER)
    prov = provider if fid in PIT_FACTORS else None

Two of those three are the two factors the dense-coverage analysis found missing pairwise for
STAR-Market securities, and the third is not in the frozen set. That is the shared dependency the
nesting (357's missing set is a subset of 471's) implied, and it explains why two factors on
different YesLab tables with different owners lost the SAME months with IDENTICAL row counts while
every control factor on both tables kept full STAR coverage: the controls do not ask for an industry
label, so a hole in the industry file cannot reach them.

`VersionedPITIndustryProvider` serves SW2014 before the cutover and SW2021 on/after it. The whole
gap (2022-01 .. 2023-09) is after 2021-12-10, so the SW2021 side is the only one that matters here.
This reads the cached membership intervals that side is built from and reports, per board, the
earliest `in_date` on record and how many STAR names are active as of the days the panel drops.

Reads local JSON files only. No database, no weights, no writes outside --out.
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
OUT_DEFAULT = Path("/data/zhangyudong/qrv2_research_artifacts/industry_cache_star_coverage.json")
STAR_PREFIXES = ("688", "689")
# The days the published panel drops, and the month the gap ends, as measured elsewhere in this pack.
AS_OF_DAYS = ("2022-07-29", "2023-07-05", "2023-09-01", "2023-09-25", "2023-10-09")


def load_members(cache: Path) -> tuple[pd.DataFrame, dict]:
    files = sorted(glob.glob(str(cache / "pit_industry_members_*_SI.json")))
    if not files:
        raise SystemExit(f"no pit_industry_members_*_SI.json under {cache}")
    frames = []
    digest = hashlib.sha256()
    for path in files:
        raw = Path(path).read_bytes()
        digest.update(hashlib.sha256(raw).digest())
        frame = pd.read_json(Path(path), orient="table")
        frame["source_file"] = Path(path).name
        frames.append(frame)
    members = pd.concat(frames, ignore_index=True)
    members["con_code"] = members["con_code"].astype(str)
    members["in_date"] = pd.to_datetime(members["in_date"], errors="coerce")
    members["out_date_parsed"] = pd.to_datetime(members["out_date"], errors="coerce", format="%Y%m%d")
    if members["out_date_parsed"].isna().all() and "out_date" in members:
        members["out_date_parsed"] = pd.to_datetime(members["out_date"], errors="coerce")
    provenance = {
        "cache_dir": str(cache),
        "member_files": len(files),
        "member_rows": int(len(members)),
        "combined_sha256_of_member_files": digest.hexdigest(),
        "columns": sorted(set(members.columns) - {"out_date_parsed", "source_file"}),
    }
    return members, provenance


def active_as_of(members: pd.DataFrame, as_of: str) -> dict:
    stamp = pd.Timestamp(as_of)
    active = members.loc[members["in_date"].notna() & (members["in_date"] <= stamp)]
    active = active.loc[active["out_date_parsed"].isna() | (active["out_date_parsed"] > stamp)]
    codes = active["con_code"]
    star = codes[codes.str.startswith(STAR_PREFIXES)]
    return {
        "as_of": as_of,
        "active_members": int(codes.nunique()),
        "active_star_members": int(star.nunique()),
        "industries_represented": int(active["index_code"].nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=CACHE_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()

    members, provenance = load_members(args.cache)
    star = members.loc[members["con_code"].str.startswith(STAR_PREFIXES)]
    non_star = members.loc[~members["con_code"].str.startswith(STAR_PREFIXES)]

    result = {
        "provenance": provenance,
        "boards": {
            "star_688_689": {
                "rows": int(len(star)),
                "distinct_codes": int(star["con_code"].nunique()),
                "earliest_in_date": str(star["in_date"].min().date()) if len(star) else None,
                "latest_in_date": str(star["in_date"].max().date()) if len(star) else None,
            },
            "non_star": {
                "rows": int(len(non_star)),
                "distinct_codes": int(non_star["con_code"].nunique()),
                "earliest_in_date": str(non_star["in_date"].min().date()) if len(non_star) else None,
                "latest_in_date": str(non_star["in_date"].max().date()) if len(non_star) else None,
            },
        },
        "star_rows_by_in_date_month": {
            str(month): int(count)
            for month, count in star.groupby(star["in_date"].dt.strftime("%Y-%m")).size().items()
        },
        "active_membership_as_of": [active_as_of(members, day) for day in AS_OF_DAYS],
        "star_codes_named_in_the_panel_report": {},
    }
    # The panel named these three as the longest-missing securities; say whether the cache has them.
    for code in ("689009.SH", "688567.SH", "688002.SH", "688003.SH", "688005.SH"):
        rows = members.loc[members["con_code"] == code]
        result["star_codes_named_in_the_panel_report"][code] = {
            "present_in_cache": bool(len(rows)),
            "industries": sorted(rows["index_code"].astype(str).unique()),
            "in_dates": sorted({str(value.date()) for value in rows["in_date"] if pd.notna(value)}),
        }

    print(f"cache: {provenance['cache_dir']}")
    print(f"member files: {provenance['member_files']}, rows: {provenance['member_rows']}")
    for board, stats in result["boards"].items():
        print(f"  {board}: {stats}")
    print("  STAR rows by in_date month (first 6):")
    for month, count in list(result["star_rows_by_in_date_month"].items())[:6]:
        print(f"    {month}: {count}")
    print("  active membership as of:")
    for entry in result["active_membership_as_of"]:
        print(f"    {entry}")
    print("  codes named in the panel report:")
    for code, entry in result["star_codes_named_in_the_panel_report"].items():
        print(f"    {code}: {entry}")

    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
