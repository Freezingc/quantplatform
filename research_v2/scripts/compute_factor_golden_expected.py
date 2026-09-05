"""Build the C8.3 factor golden fixture and its independently derived expectations.

The expectations here are computed by a hand-written reference that deliberately shares no
code with ``research_v2.factors``: deduplication is a dict-of-groups loop, availability is
the point-in-time rule transcribed from the C4 spec, and nothing is imported from the
package under test except the contract enums needed to describe a factor. A golden sample
produced by running the implementation and snapshotting its output would be circular -- it
would pin whatever the code happens to do, including a defect. This mirrors the B plan's
``compute_pit_golden_expected.py``.

The fixture is the same 50 securities x 60 signal dates shape the B plan uses, extended with
the eight cases C8.3 requires. The 60 dates are two blocks of 30 business days, January and
September 2025, because a single contiguous 60-day span cannot reach the 180-day staleness
bound; the gap is what makes the stale case arise naturally rather than by lowering a
threshold to fit the fixture.

Usage:
  compute_factor_golden_expected.py            # regenerate fixtures + README
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pandas as pd

GOLDEN_DATA = Path(__file__).resolve().parents[1] / "tests" / "golden" / "data"
GOLDEN_README = Path(__file__).resolve().parents[1] / "tests" / "golden" / "README_factor_cases.md"

SECURITIES = tuple(f"{index:06d}.SZ" for index in range(50))

# Two blocks of 30 business days. The January block carries the announcement and its
# revision; the September block is where that revision has gone stale.
SIGNAL_DATES: tuple[date, ...] = (
    tuple(ts.date() for ts in pd.bdate_range("2025-01-02", periods=30))
    + tuple(ts.date() for ts in pd.bdate_range("2025-09-01", periods=30))
)

CALENDAR: tuple[date, ...] = tuple(
    ts.date() for ts in pd.bdate_range("2025-01-01", "2025-12-31")
)

TECHNICAL_FACTORS: tuple[tuple[str, str], ...] = (
    ("yeslab_901", "alpha_dense"),
    ("yeslab_902", "alpha_dup"),
    ("yeslab_903", "alpha_single"),
    ("yeslab_904", "alpha_conflict"),
    ("yeslab_905", "alpha_mirror"),
    ("yeslab_906", "alpha_copy"),
)
FUNDAMENTAL_FACTOR_ID = "yeslab_910"
MAX_STALENESS_DAYS = 180

# Case anchors, expressed as (security index range, day index range) so the README can name
# the exact cells each case occupies.
MISSING_SECODIES = range(10, 15)
MISSING_DAYS = range(20, 25)
DUPLICATE_SECODIES = range(0, 5)
DUPLICATE_DAYS = range(0, 5)
SINGLE_SECODIES = range(20, 25)
SINGLE_DAYS = range(10, 15)
CONFLICT_SECODIES = range(30, 35)
CONFLICT_DAYS = range(30, 35)

REPORT_PERIOD = date(2024, 12, 31)
FIRST_ANNOUNCEMENT = date(2025, 1, 3)   # a Friday; available the following Monday
# Also a Friday, and deliberately inside the January signal block: a revision announced in
# February would become available on 2025-02-17 and expire on 2025-08-16, and with signal
# dates only in January and September that window would contain no signal date at all -- so
# the revision would never be published and case 4 would silently go unexercised. Measured
# before this correction: expected fundamental rows carried only revision r1.
REVISION_ANNOUNCEMENT = date(2025, 1, 17)


def base_value(day_index: int, security_index: int) -> float:
    """A deterministic value with cross-sectional variation and no ties within a day.

    No RNG: a golden fixture must regenerate byte-identically on any machine, and a seeded
    generator still leaves the reader unable to predict a cell by inspection.
    """
    return round(math.sin(security_index) * 100.0 + day_index, 6)


def build_technical_source() -> pd.DataFrame:
    """The raw YesLab-shaped extract: ts_code / trade_date / factor_name / value."""
    rows: list[dict[str, object]] = []

    def emit(factor_name: str, day_index: int, security_index: int, value: float | None) -> None:
        rows.append(
            {
                "ts_code": SECURITIES[security_index],
                "trade_date": SIGNAL_DATES[day_index],
                "factor_name": factor_name,
                "value": value,
            }
        )

    for day_index in range(len(SIGNAL_DATES)):
        for security_index in range(len(SECURITIES)):
            value = base_value(day_index, security_index)
            # Case 1: a daily technical block simply absent from the source. It must stay
            # absent -- never forward-filled from the previous day.
            dense_present = not (
                security_index in MISSING_SECODIES and day_index in MISSING_DAYS
            )
            if dense_present:
                emit("alpha_dense", day_index, security_index, value)
                emit("alpha_copy", day_index, security_index, value)
                emit("alpha_mirror", day_index, security_index, -value)
            emit("alpha_dup", day_index, security_index, value)
            if security_index in DUPLICATE_SECODIES and day_index in DUPLICATE_DAYS:
                # Case 5: a byte-identical second row, as YesLab daily extracts really do
                # carry. Must collapse to one published row, not conflict.
                emit("alpha_dup", day_index, security_index, value)
            emit("alpha_single", day_index, security_index, value)
            if security_index in SINGLE_SECODIES and day_index in SINGLE_DAYS:
                # Case 6: one finite value plus a null for the same key. The finite one is
                # published and the group is labelled SINGLE_NONNULL.
                emit("alpha_single", day_index, security_index, None)
            if security_index in CONFLICT_SECODIES and day_index in CONFLICT_DAYS:
                # Case 7: two distinct finite values for one key. The whole group is
                # quarantined; nothing is published and no side is chosen.
                emit("alpha_conflict", day_index, security_index, value)
                emit("alpha_conflict", day_index, security_index, value + 0.5)
            else:
                emit("alpha_conflict", day_index, security_index, value)
    return pd.DataFrame(rows)


def build_fundamental_events() -> pd.DataFrame:
    """One report period, announced once and later restated, for every security."""
    rows: list[dict[str, object]] = []
    for security_index, ts_code in enumerate(SECURITIES):
        rows.append(
            {
                "ts_code": ts_code,
                "report_period": REPORT_PERIOD,
                "ann_date": FIRST_ANNOUNCEMENT,
                "raw_value": round(10.0 + security_index, 6),
                "source_revision_id": "r1",
            }
        )
        rows.append(
            {
                "ts_code": ts_code,
                "report_period": REPORT_PERIOD,
                "ann_date": REVISION_ANNOUNCEMENT,
                "raw_value": round(20.0 + security_index, 6),
                "source_revision_id": "r2",
            }
        )
    return pd.DataFrame(rows)


def build_universe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"signal_date": day, "ts_code": code}
            for day in SIGNAL_DATES
            for code in SECURITIES
        ]
    )


# --- independent reference implementation -------------------------------------


def reference_next_trading_day(ann_date: date) -> date:
    """The first calendar trading day strictly after the announcement."""
    for day in CALENDAR:
        if day > ann_date:
            return day
    raise ValueError(f"announcement {ann_date} is beyond the calendar")


def reference_dedup(
    factor_id: str, rows: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Classify each business-key group by hand, without touching the package under test.

    The published value is unambiguous in every class, which is what makes an independent
    reference possible at all: an all-identical group has one value, a single-nonnull group
    has one finite value, and a conflicting group publishes nothing. So this does not need
    to reproduce the source_row_id hashing the implementation uses to break ties.
    """
    groups: dict[tuple[date, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (row["trade_date"], str(row["ts_code"]))
        groups.setdefault(key, []).append(row)

    clean: list[dict[str, object]] = []
    quarantine: list[dict[str, object]] = []
    for (observation_date, ts_code), group in groups.items():
        finite = [
            float(row["value"])
            for row in group
            if row["value"] is not None and math.isfinite(float(row["value"]))
        ]
        distinct = sorted(set(finite))
        if len(distinct) > 1:
            quarantine.append(
                {
                    "observation_date": observation_date,
                    "ts_code": ts_code,
                    "factor_id": factor_id,
                    "factor_version": 1,
                    "reason": "CONFLICT_QUARANTINE",
                    "distinct_values_json": str(distinct),
                }
            )
            continue
        if not finite:
            quarantine.append(
                {
                    "observation_date": observation_date,
                    "ts_code": ts_code,
                    "factor_id": factor_id,
                    "factor_version": 1,
                    "reason": "NO_FINITE_VALUE",
                    "distinct_values_json": str(distinct),
                }
            )
            continue
        resolution = "IDENTICAL_COLLAPSE" if len(finite) == len(group) else "SINGLE_NONNULL"
        clean.append(
            {
                "signal_date": observation_date,
                "ts_code": ts_code,
                "factor_id": factor_id,
                "factor_version": 1,
                "value": distinct[0],
                "dedup_resolution": resolution,
            }
        )
    return clean, quarantine


def reference_expected_technical(
    source: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Expected clean and quarantine rows for every technical factor.

    A group of one is UNIQUE rather than IDENTICAL_COLLAPSE, matching the implementation's
    distinction between a key that was never duplicated and one that was.
    """
    clean: list[dict[str, object]] = []
    quarantine: list[dict[str, object]] = []
    for factor_id, factor_name in TECHNICAL_FACTORS:
        rows = source.loc[source["factor_name"] == factor_name].to_dict(orient="records")
        groups: dict[tuple[date, str], int] = {}
        for row in rows:
            key = (row["trade_date"], str(row["ts_code"]))
            groups[key] = groups.get(key, 0) + 1
        factor_clean, factor_quarantine = reference_dedup(factor_id, rows)
        for record in factor_clean:
            key = (record["signal_date"], str(record["ts_code"]))
            if groups[key] == 1:
                record["dedup_resolution"] = "UNIQUE"
        clean.extend(factor_clean)
        quarantine.extend(factor_quarantine)
    return clean, quarantine


def reference_expected_fundamental(
    events: pd.DataFrame, universe: pd.DataFrame
) -> list[dict[str, object]]:
    """Expected expansion, transcribed from the C4.3/C4.4 rules rather than from the code."""
    versions = [
        {
            "ts_code": str(row["ts_code"]),
            "report_period": row["report_period"],
            "ann_date": row["ann_date"],
            "available_date": reference_next_trading_day(row["ann_date"]),
            "raw_value": float(row["raw_value"]),
            "source_revision_id": str(row["source_revision_id"]),
        }
        for row in events.to_dict(orient="records")
    ]
    expected: list[dict[str, object]] = []
    for row in universe.to_dict(orient="records"):
        signal_date = row["signal_date"]
        ts_code = str(row["ts_code"])
        known = [
            version
            for version in versions
            if version["ts_code"] == ts_code and version["available_date"] <= signal_date
        ]
        if not known:
            continue
        latest = max(
            known,
            key=lambda item: (
                item["available_date"],
                item["report_period"],
                item["source_revision_id"],
            ),
        )
        stale_days = (signal_date - latest["available_date"]).days
        if stale_days > MAX_STALENESS_DAYS:
            continue
        expected.append(
            {
                "signal_date": signal_date,
                "ts_code": ts_code,
                "raw_value": latest["raw_value"],
                "report_period": latest["report_period"],
                "ann_date": latest["ann_date"],
                "available_date": latest["available_date"],
                "source_revision_id": latest["source_revision_id"],
                "stale_days": stale_days,
            }
        )
    return expected


def main() -> int:
    source = build_technical_source()
    events = build_fundamental_events()
    universe = build_universe()

    clean, quarantine = reference_expected_technical(source)
    fundamental = reference_expected_fundamental(events, universe)

    clean_frame = pd.DataFrame(clean).sort_values(
        ["factor_id", "signal_date", "ts_code"], ignore_index=True
    )
    quarantine_frame = pd.DataFrame(quarantine).sort_values(
        ["factor_id", "observation_date", "ts_code"], ignore_index=True
    )
    fundamental_frame = pd.DataFrame(fundamental).sort_values(
        ["signal_date", "ts_code"], ignore_index=True
    )

    GOLDEN_DATA.mkdir(parents=True, exist_ok=True)
    source.to_parquet(GOLDEN_DATA / "factors_50x60.parquet", index=False)
    events.to_parquet(GOLDEN_DATA / "fundamental_events_50x60.parquet", index=False)
    universe.to_parquet(GOLDEN_DATA / "universe_50x60.parquet", index=False)
    clean_frame.to_parquet(GOLDEN_DATA / "expected_factor_observations.parquet", index=False)
    quarantine_frame.to_parquet(GOLDEN_DATA / "expected_factor_quarantine.parquet", index=False)
    fundamental_frame.to_parquet(
        GOLDEN_DATA / "expected_fundamental_expansion.parquet", index=False
    )

    resolutions = clean_frame["dedup_resolution"].value_counts().to_dict()
    reasons = quarantine_frame["reason"].value_counts().to_dict()
    print(f"technical source rows      : {len(source)}")
    print(f"expected clean rows        : {len(clean_frame)}  resolutions={resolutions}")
    print(f"expected quarantine rows   : {len(quarantine_frame)}  reasons={reasons}")
    print(f"expected fundamental rows  : {len(fundamental_frame)}")
    print(
        "fundamental stale_days     : "
        f"min={int(fundamental_frame['stale_days'].min())} "
        f"max={int(fundamental_frame['stale_days'].max())}"
    )
    print(
        "fundamental revisions seen : "
        f"{sorted(set(fundamental_frame['source_revision_id']))}"
    )
    _write_readme(len(source), clean_frame, quarantine_frame, fundamental_frame)
    print(f"wrote fixtures to {GOLDEN_DATA}")
    return 0


def _write_readme(
    source_rows: int,
    clean: pd.DataFrame,
    quarantine: pd.DataFrame,
    fundamental: pd.DataFrame,
) -> None:
    """Per-case documentation, as C8.3 requires: every case named with its expected effect."""
    resolutions = clean["dedup_resolution"].value_counts().to_dict()
    reasons = quarantine["reason"].value_counts().to_dict()
    revisions = fundamental["source_revision_id"].value_counts().to_dict()
    lines = [
        "# Factor golden sample: the eight cases C8.3 requires",
        "",
        "Regenerate with `python scripts/compute_factor_golden_expected.py`. The expectations are",
        "produced by the hand-written reference in that script, which imports nothing from",
        "`research_v2.factors`, so the golden test compares the production pipeline against an",
        "independent derivation rather than against a snapshot of itself.",
        "",
        f"Universe: {len(SECURITIES)} securities x {len(SIGNAL_DATES)} signal dates",
        f"({SIGNAL_DATES[0]} .. {SIGNAL_DATES[-1]}, two blocks of 30 business days).",
        "The gap between the blocks is deliberate: a contiguous 60-day span cannot reach the",
        f"{MAX_STALENESS_DAYS}-day staleness bound, so the stale case would otherwise require",
        "lowering a threshold to fit the fixture.",
        "",
        f"Technical source rows: {source_rows}. Expected clean: {len(clean)}.",
        f"Expected quarantine: {len(quarantine)}.",
        "",
        "| # | case | where | expected effect |",
        "|---|---|---|---|",
        f"| 1 | daily technical missing | `alpha_dense`/`alpha_copy`/`alpha_mirror`, securities"
        f" {MISSING_SECODIES.start}-{MISSING_SECODIES.stop - 1}, day indices"
        f" {MISSING_DAYS.start}-{MISSING_DAYS.stop - 1} | those cells are absent from clean;"
        " they are **not** forward-filled from the previous day |",
        f"| 2 | fundamental announced on a Friday | ann_date {FIRST_ANNOUNCEMENT} |"
        f" available_date {reference_next_trading_day(FIRST_ANNOUNCEMENT)}, the next open day |",
        f"| 3 | 180-day stale | September block | those signal dates publish **nothing** for"
        f" {FUNDAMENTAL_FACTOR_ID}, because the latest version exceeds {MAX_STALENESS_DAYS} days |",
        f"| 4 | revision announcement | ann_date {REVISION_ANNOUNCEMENT}, revision `r2` |"
        " signal dates from its available date onward see `r2`; earlier dates keep `r1` |",
        f"| 5 | exact duplicate | `alpha_dup`, securities {DUPLICATE_SECODIES.start}-"
        f"{DUPLICATE_SECODIES.stop - 1}, day indices {DUPLICATE_DAYS.start}-"
        f"{DUPLICATE_DAYS.stop - 1} | one published row, `dedup_resolution=IDENTICAL_COLLAPSE` |",
        f"| 6 | single-nonnull | `alpha_single`, securities {SINGLE_SECODIES.start}-"
        f"{SINGLE_SECODIES.stop - 1}, day indices {SINGLE_DAYS.start}-{SINGLE_DAYS.stop - 1} |"
        " the finite value is published, `dedup_resolution=SINGLE_NONNULL` |",
        f"| 7 | conflict | `alpha_conflict`, securities {CONFLICT_SECODIES.start}-"
        f"{CONFLICT_SECODIES.stop - 1}, day indices {CONFLICT_DAYS.start}-"
        f"{CONFLICT_DAYS.stop - 1} | the whole group is quarantined as `CONFLICT_QUARANTINE`"
        " and nothing is published; no side is chosen |",
        "| 8 | identical and sign-opposite factors | `alpha_copy` = `alpha_dense`,"
        " `alpha_mirror` = -`alpha_dense` | both reach |Spearman| of 1 against `alpha_dense`"
        " (the mirror measures 0.9999999999999999, since a rank correlation over 50 names"
        " accumulates float error), so no threshold the pipeline would use can separate"
        " either from its origin |",
        "",
        "## Expected counts",
        "",
        f"- clean `dedup_resolution`: {resolutions}",
        f"- quarantine `reason`: {reasons}",
        f"- fundamental `source_revision_id`: {revisions}",
        f"- fundamental `stale_days`: min {int(fundamental['stale_days'].min())},"
        f" max {int(fundamental['stale_days'].max())}",
        "",
        "Columns compared by the golden test are the business-meaningful ones only. `created_at`"
        " is a wall-clock timestamp and `record_hash` derives from it, so neither is"
        " reproducible and both are excluded, exactly as the B plan's market golden test"
        " excludes its own non-deterministic fields.",
        "",
    ]
    GOLDEN_README.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
