"""Where exactly the STAR rows are lost: upstream -> factor_daily -> published pin -> parquet.

The first probe (probe_star_upstream_coverage.py) falsified the producer-gap hypothesis: on
2022-07-29 and 2023-07-05 the upstream YesLab tables DO carry 688/689 rows with non-null values for
both `行业内收益标准化绝对值` (yeslab_357) and `AttentionSpillover` (yeslab_471), and their counts
are indistinguishable from the near-complete controls on the same tables. So the rows are lost
somewhere on our side of the chain, and "somewhere" is not good enough to hand to anyone.

The chain has three candidate drop points, and this separates them on the same day:

  1. factor_daily has no such rows at all            -> lost in materialization/cleaning
  2. rows exist, but under a data_version that the
     published pin does not point at                 -> stale/partial publish, filter_current drops them
  3. rows exist under the pinned version             -> lost in dataset assembly (builder join)

Case 2 is the one worth suspecting a priori: `filter_current` keeps only rows whose data_version
equals the version published for that logical partition, so a repair that landed in the table
without a republish is invisible to every run, silently.

SAFETY: SELECT-only. Reads published dataset parquet already on disk plus read-only queries against
dfs://YesLab and dfs://QuantResearchV2Test. No writes, no weights, no artifact mutation, and it
does not import or edit anything the M2 driver executes. Run from research_v2/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from walk_forward_common import TARGET_DATABASE, new_holder, query  # noqa: E402

ARTIFACTS = Path("/data/zhangyudong/qrv2_research_artifacts")
OUT = ARTIFACTS / "star_missing_chain_probe.json"
WORKSPACE = ARTIFACTS / "m0r_v2"

# column -> factor_id, from the pushed dense-coverage evidence (not retyped from memory)
EVIDENCE = Path(
    "/data/zhangyudong/qrv2_worktrees/research-conditional-factor-review/docs/research/evidence/"
    "2026-09-14-model-side-walk-forward/dense_factor_coverage/m0r_v2.json"
)

# (day, test-year split, upstream table, upstream factor_name)
DAYS = (
    ("2022-07-29", 2022, "factor_data8", "行业内收益标准化绝对值"),
    ("2023-07-05", 2023, "factor_data8", "行业内收益标准化绝对值"),
)
TARGETS = ("yeslab_357", "yeslab_471")
STAR_PREFIXES = ("688", "689")


def dataset_dir(year: int) -> Path:
    matches = sorted(WORKSPACE.glob(f"datasets/*/split=test/year={year}"))
    if not matches:
        raise SystemExit(f"no test/year={year} split under {WORKSPACE}")
    return matches[0].parent.parent  # the dataset root, holding manifest.json


def panel_rows(root: Path, year: int, day: str) -> pd.DataFrame:
    shards = sorted((root / f"split=test/year={year}").glob("*.parquet"))
    frame = pd.concat([pd.read_parquet(path) for path in shards], ignore_index=True)
    frame["day"] = pd.to_datetime(frame["signal_date"]).dt.strftime("%Y-%m-%d")
    label_valid = frame["label_valid"].astype("boolean").fillna(False).astype(bool)
    finite = np.isfinite(pd.to_numeric(frame["raw_return_5d"], errors="coerce"))
    return frame.loc[(frame["day"] == day) & label_valid & finite]


def manifest_versions(root: Path) -> dict[str, str]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return manifest["metadata"]["data_versions"].get("factor_daily", {})


def main() -> None:
    columns = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    col_to_factor = dict(zip(columns["dense_columns"], columns["dense_factors"], strict=True))
    target_cols = [c for c, f in col_to_factor.items() if f in TARGETS]
    control_col = next(c for c, f in col_to_factor.items() if f == "yeslab_381")
    print(f"dense mapping: {col_to_factor}", flush=True)

    holder = new_holder()
    result: dict = {"days": {}, "published_pins": {}}

    _, pins = query(
        holder,
        "select logical_partition, data_version, published_at "
        f"from loadTable('{TARGET_DATABASE}','published_version') "
        "where object_name='factor_daily'",
    )
    pin_map: dict[str, tuple[str, str]] = {}
    if isinstance(pins, pd.DataFrame) and not pins.empty:
        pins = pins.sort_values("published_at", kind="mergesort")
        for row in pins.itertuples(index=False):
            pin_map[str(row.logical_partition)] = (str(row.data_version), str(row.published_at))
    result["published_pins"] = {
        partition: {"data_version": v[0], "published_at": v[1]}
        for partition, v in sorted(pin_map.items())
        if partition.startswith(("2022-07", "2023-07", "2023-09", "2023-10"))
    }
    print(f"published factor_daily partitions: {len(pin_map)}", flush=True)

    for day, year, table, upstream_name in DAYS:
        root = dataset_dir(year)
        scored = panel_rows(root, year, day)
        partition = day[:7]
        consumed = manifest_versions(root).get(partition)
        pinned = pin_map.get(partition, (None, None))[0]
        print(
            f"\n=== {day} (test/year={year}, fold dataset {root.name[:12]}) ===\n"
            f"  panel valid-label rows: {len(scored)}; "
            f"manifest factor_daily[{partition}]={consumed}; published pin={pinned}; "
            f"same={consumed == pinned}",
            flush=True,
        )
        entry: dict = {
            "year": year,
            "dataset": root.name,
            "partition": partition,
            "manifest_data_version": consumed,
            "published_pin_data_version": pinned,
            "pin_matches_manifest": consumed == pinned,
            "panel_valid_label_rows": int(len(scored)),
            "factors": {},
        }

        missing_sets: dict[str, set[str]] = {}
        for column in [*target_cols, control_col]:
            factor_id = col_to_factor[column]
            codes = scored.loc[scored[column].isna(), "ts_code"].astype(str)
            missing_sets[factor_id] = set(codes)
            star_missing = sorted(c for c in codes if c.startswith(STAR_PREFIXES))
            entry["factors"][factor_id] = {
                "panel_column": column,
                "panel_missing_rows": int(len(codes)),
                "panel_missing_star": len(star_missing),
                "panel_missing_star_sample": star_missing[:5],
            }
            print(
                f"  panel  {factor_id:<11} ({column}) missing={len(codes):>4} "
                f"of which STAR={len(star_missing):>4}",
                flush=True,
            )

        for factor_id in [*TARGETS, "yeslab_381"]:
            _, rows = query(
                holder,
                "select signal_date, ts_code, factor_id, factor_version, value, data_version "
                f"from loadTable('{TARGET_DATABASE}','factor_daily') "
                f"where signal_date between date('{day.replace('-', '.')}') "
                f"and date('{day.replace('-', '.')}') "
                f"and factor_id='{factor_id}'",
            )
            if not isinstance(rows, pd.DataFrame) or rows.empty:
                entry["factors"][factor_id]["factor_daily"] = {"rows": 0}
                print(f"  f_daily {factor_id:<11} NO ROWS AT ALL", flush=True)
                continue
            code = rows["ts_code"].astype(str)
            value = pd.to_numeric(rows["value"], errors="coerce")
            version = rows["data_version"].astype(str)
            is_star = code.str.startswith(STAR_PREFIXES)
            per_version = {}
            for ver, group in rows.assign(_star=is_star, _val=value.notna()).groupby(version):
                per_version[str(ver)] = {
                    "rows": int(len(group)),
                    "codes": int(group["ts_code"].nunique()),
                    "star_codes": int(group.loc[group["_star"], "ts_code"].nunique()),
                    "star_codes_with_value": int(
                        group.loc[group["_star"] & group["_val"], "ts_code"].nunique()
                    ),
                    "is_pinned": str(ver) == pinned,
                    "is_consumed": str(ver) == consumed,
                }
            star_codes_in_table = set(code[is_star & value.notna()])
            missing = missing_sets.get(factor_id, set())
            entry["factors"][factor_id]["factor_daily"] = {
                "rows": int(len(rows)),
                "data_versions": sorted(per_version),
                "per_data_version": per_version,
                "star_codes_with_value_any_version": len(star_codes_in_table),
                "panel_missing_star_recovered_by_table": len(
                    {c for c in missing if c.startswith(STAR_PREFIXES)} & star_codes_in_table
                ),
                "panel_missing_star_total": len(
                    [c for c in missing if c.startswith(STAR_PREFIXES)]
                ),
            }
            print(
                f"  f_daily {factor_id:<11} rows={len(rows):>6} "
                f"versions={sorted(per_version)} "
                f"star_with_value={len(star_codes_in_table):>4} "
                f"panel-missing-Star-recovered={entry['factors'][factor_id]['factor_daily']['panel_missing_star_recovered_by_table']}"
                f"/{entry['factors'][factor_id]['factor_daily']['panel_missing_star_total']}",
                flush=True,
            )
            for ver, stats in sorted(per_version.items()):
                print(
                    f"          version {ver[:16]} rows={stats['rows']:>6} "
                    f"star_codes={stats['star_codes']:>4} "
                    f"star_with_value={stats['star_codes_with_value']:>4} "
                    f"pinned={stats['is_pinned']} consumed={stats['is_consumed']}",
                    flush=True,
                )
            del rows

        # Upstream, restricted to exactly the codes the panel says are missing for yeslab_357.
        missing = sorted(missing_sets["yeslab_357"])
        _, up = query(
            holder,
            "select ts_code, trade_date, value from "
            f"loadTable('dfs://YesLab', '{table}') where factor_name = '{upstream_name}' "
            f"and trade_date between date('{day.replace('-', '.')}') "
            f"and date('{day.replace('-', '.')}')",
        )
        if isinstance(up, pd.DataFrame) and not up.empty:
            up_code = up["ts_code"].astype(str)
            up_val = pd.to_numeric(up["value"], errors="coerce")
            upstream_with_value = set(up_code[up_val.notna()])
            wanted = set(missing)
            entry["upstream"] = {
                "table": table,
                "factor_name": upstream_name,
                "panel_missing_for_yeslab_357": len(wanted),
                "present_upstream_with_value": len(wanted & upstream_with_value),
                "present_upstream_star_with_value": len(
                    {c for c in wanted & upstream_with_value if c.startswith(STAR_PREFIXES)}
                ),
                "duplicate_rows_in_table": int(len(up)) - int(up_code.nunique()),
            }
            print(
                f"  upstream {table}: of the {len(wanted)} codes the panel says are missing "
                f"for yeslab_357, {entry['upstream']['present_upstream_with_value']} have a "
                f"non-null value upstream on that same day "
                f"(STAR: {entry['upstream']['present_upstream_star_with_value']}); "
                f"duplicate rows in table={entry['upstream']['duplicate_rows_in_table']}",
                flush=True,
            )
        else:
            entry["upstream"] = {"table": table, "rows": 0}
        result["days"][day] = entry

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
