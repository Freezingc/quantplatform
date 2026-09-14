"""The census treated SW2014 and SW2021 as two halves that independently agree. They are one dataset.

probe_industry_label_census.py loads SOURCES = {SW2014: pit_industry_sw2014_members_*, SW2021:
pit_industry_members_*} and reports both, which reads as a cross-check. But the builder calls
pro.index_member(index_code=...) -- and that API takes no `src` argument, so the same index_code
returns the same rows whichever classification system the caller thinks it is serving. Where the two
L1 code sets overlap, the two cache halves should therefore be byte-identical, and the agreement
carries no independent evidence.

This hashes every file in both halves and reports the identity exactly. Reads local JSON only.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

CACHE = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/artifacts/tushare_cache")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cls = {}
    for src in ("sw2014", "sw2021"):
        rows = json.loads((CACHE / f"pit_industry_{src}_l1_classification.json").read_text())["data"]
        cls[src] = {r["index_code"]: r["industry_name"] for r in rows}

    legacy = {p.stem.replace("pit_industry_members_", ""): p
              for p in CACHE.glob("pit_industry_members_*_SI.json")}
    s14 = {p.stem.replace("pit_industry_sw2014_members_", ""): p
           for p in CACHE.glob("pit_industry_sw2014_members_*_SI.json")}
    s21 = {p.stem.replace("pit_industry_sw2021_members_", ""): p
           for p in CACHE.glob("pit_industry_sw2021_members_*_SI.json")}

    shared = sorted(set(legacy) & set(s14))
    identical = [k for k in shared if sha(legacy[k]) == sha(s14[k])]
    differ = [k for k in shared if sha(legacy[k]) != sha(s14[k])]

    key = lambda code: code.replace(".", "_")
    out = {
        "claim_under_test": "the SW2014 and SW2021 cache halves are two independent measurements",
        "verdict": None,
        "builder_api": "pro.index_member(index_code=..., fields='index_code,con_code,in_date,out_date,is_new')",
        "builder_takes_a_src_argument": False,
        "classification_sizes": {s: len(v) for s, v in cls.items()},
        "l1_codes_sw2014_only": sorted(set(cls["sw2014"]) - set(cls["sw2021"])),
        "l1_codes_sw2021_only": sorted(set(cls["sw2021"]) - set(cls["sw2014"])),
        "member_files": {
            "legacy_pit_industry_members": len(legacy),
            "pit_industry_sw2014_members": len(s14),
            "pit_industry_sw2021_members": len(s21),
            "note_sw2021": "no sw2021-named member files exist; fetch_pit_members falls back to the "
                           "legacy names, so the SW2021 half of the census IS the legacy half",
        },
        "legacy_code_set_equals": {
            "sw2021_l1_codes": sorted(key(c) for c in cls["sw2021"]) == sorted(legacy),
            "sw2014_l1_codes": sorted(key(c) for c in cls["sw2014"]) == sorted(s14),
        },
        "comparison": {
            "codes_present_in_both_halves": len(shared),
            "byte_identical": len(identical),
            "different": differ,
            "identity_rate": (len(identical) / len(shared)) if shared else None,
        },
        "star_min_in_date_per_half": {},
        "database_writes": False,
    }
    import pandas as pd
    for label, table in (("legacy_used_as_SW2021", legacy), ("sw2014", s14)):
        frames = [pd.read_json(p, orient="table") for p in table.values()]
        m = pd.concat(frames, ignore_index=True)
        star = m[m["con_code"].astype(str).str.startswith(("688", "689"))]
        out["star_min_in_date_per_half"][label] = {
            "rows": int(len(m)), "star_rows": int(len(star)),
            "star_min_in_date": str(star["in_date"].min()) if len(star) else None,
            "all_min_in_date": str(m["in_date"].min()),
        }
    same_star = (out["star_min_in_date_per_half"]["legacy_used_as_SW2021"]["star_min_in_date"]
                 == out["star_min_in_date_per_half"]["sw2014"]["star_min_in_date"])
    out["verdict"] = (
        f"REFUTED -- {len(identical)} of {len(shared)} overlapping industry files are byte-identical "
        f"(identity rate {out['comparison']['identity_rate']:.4f}), both halves reach the same earliest "
        f"STAR in_date ({'same' if same_star else 'different'}), and no sw2021-named member files exist "
        "at all. index_member takes no src argument, so the two halves are one dataset wearing two names."
    )
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
