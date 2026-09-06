"""Build C3.4's deduplication regression fixture from a measured duplicate-key census.

C3.4 asks for anonymized GP_017 and GP_077 duplicate-key samples and is explicit that the
fixture stores only the necessary fields plus a source hash, not bulk source data. So this
script copies nothing: it reads the aggregate census produced by
``census_duplicate_key_shapes.py`` and emits a small sample whose *shape* is the measured
one -- how many keys fall in each of the six classes the deduplicator distinguishes, how
many rows each duplicated key carries, and how many of those rows hold a usable value.
Business keys and values are synthetic. What is real is the distribution, and every number
in the output traces to a recorded measurement plus the rounding rule stated beside it.

Two things make the output checkable rather than merely plausible. The allocation is
largest-remainder rounding of the measured ratios, so it is exact, sums to the sample size,
and can be recomputed independently by a test from the metrics stored alongside it. And
the script expands its own fixture before writing it, refusing to emit a sample whose
class histogram is not the one it claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

# The six classes are exactly the six things ``deduplicate_factor_observations`` does with a
# business key, and they are the six cells the census counts, so the mapping is one to one.
CLASSES: tuple[str, ...] = (
    "unique_usable",
    "unique_unusable",
    "duplicated_identical",
    "duplicated_single_nonnull",
    "duplicated_conflict",
    "duplicated_no_usable_value",
)

FACTOR_LABELS: dict[str, str] = {"GP_017": "yeslab_386", "GP_077": "yeslab_446"}

_FIRST_DATE = date(2020, 1, 2)


def allocate(ratios: Mapping[str, float], total: int) -> dict[str, int]:
    """Split ``total`` keys across the classes in proportion to ``ratios``.

    Largest remainder, so the parts always sum to ``total`` exactly and no class is
    silently rounded away. Ties break on the class name rather than on dict order, because
    a fixture that depends on iteration order is not reproducible.
    """
    names = sorted(ratios)
    exact = {name: ratios[name] * total for name in names}
    allocated = {name: int(exact[name] // 1) for name in names}
    leftover = total - sum(allocated.values())
    ranked = sorted(names, key=lambda name: (-(exact[name] - allocated[name]), name))
    for name in ranked[:leftover]:
        allocated[name] += 1
    return allocated


def class_ratios(metrics: Mapping[str, int]) -> dict[str, float]:
    """Each class as a share of all keys, from the census's own partition."""
    keys = metrics["key_count"]
    if keys <= 0:
        raise ValueError("a census with no keys cannot be sampled")
    return {name: metrics[f"{name}_key_count"] / keys for name in CLASSES}


def duplicated_multiplicity(metrics: Mapping[str, int]) -> int:
    """Rows per duplicated key, which the census shows is uniform for both factors.

    Refusing to invent a spread is the point: if the average is not an integer then the
    source mixes multiplicities, and a fixture built on one number would be wrong in a way
    nothing downstream could see. That needs a multiplicity histogram in the census first.
    """
    duplicated = metrics["duplicated_key_count"]
    if duplicated == 0:
        raise ValueError("this factor has no duplicated keys to sample")
    rows = metrics["row_count"] - metrics["unique_key_count"]
    average = rows / duplicated
    if abs(average - round(average)) > 1e-9:
        raise ValueError(
            f"duplicated keys average {average} rows each, which is not uniform; the census "
            "needs a multiplicity histogram before a fixture can claim one shape"
        )
    multiplicity = int(round(average))
    if multiplicity != metrics["max_multiplicity"]:
        raise ValueError(
            f"uniform multiplicity {multiplicity} disagrees with max_multiplicity "
            f"{metrics['max_multiplicity']}"
        )
    return multiplicity


def _values_for(
    generator: random.Random,
    class_name: str,
    multiplicity: int,
    usable_rows: int,
    distinct_values: int,
) -> list[float | None]:
    """One synthetic group of the given class: ``usable_rows`` values, ``distinct`` of them.

    Values are drawn from a fixed range and rounded, so the fixture is readable and holds no
    source data. Only the *count* of usable rows and of distinct values carries meaning.
    """
    if class_name in {"unique_unusable", "duplicated_no_usable_value"}:
        return [None] * multiplicity
    distinct = [round(generator.uniform(-3.0, 3.0), 6) for _ in range(distinct_values)]
    usable = [distinct[index % distinct_values] for index in range(usable_rows)]
    return usable + [None] * (multiplicity - usable_rows)


def build_groups(
    generator: random.Random,
    metrics: Mapping[str, int],
    allocated: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Emit one group per sampled key, in class order, with the measured within-group shape."""
    multiplicity = duplicated_multiplicity(metrics)
    single_nonnull = metrics["duplicated_single_nonnull_key_count"]
    conflicts = metrics["duplicated_conflict_key_count"]
    usable_per_single_nonnull = (
        1
        if single_nonnull == 0
        else max(
            1,
            min(
                multiplicity - 1,
                round(metrics["single_nonnull_usable_row_count"] / single_nonnull),
            ),
        )
    )
    usable_per_conflict = (
        multiplicity
        if conflicts == 0
        else max(
            2, min(multiplicity, round(metrics["conflict_usable_row_count"] / conflicts))
        )
    )
    distinct_per_conflict = (
        2
        if conflicts == 0
        else max(
            2,
            min(
                usable_per_conflict,
                round(metrics["conflict_distinct_value_total"] / conflicts),
            ),
        )
    )
    shapes = {
        "unique_usable": (1, 1, 1),
        "unique_unusable": (1, 0, 0),
        "duplicated_identical": (multiplicity, multiplicity, 1),
        "duplicated_single_nonnull": (multiplicity, usable_per_single_nonnull, 1),
        "duplicated_conflict": (multiplicity, usable_per_conflict, distinct_per_conflict),
        "duplicated_no_usable_value": (multiplicity, 0, 0),
    }
    groups: list[dict[str, Any]] = []
    for class_name in CLASSES:
        size, usable_rows, distinct_values = shapes[class_name]
        for _ in range(allocated[class_name]):
            groups.append(
                {
                    "class": class_name,
                    "values": _values_for(
                        generator, class_name, size, usable_rows, distinct_values
                    ),
                }
            )
    return groups


def expand_rows(
    groups: Sequence[Mapping[str, Any]],
    *,
    factor_id: str,
    codes_per_date: int = 50,
) -> pd.DataFrame:
    """Turn fixture groups into the frame ``deduplicate_factor_observations`` accepts.

    Keys are synthetic and derived from the group's position: one ``(date, code)`` pair per
    group, so groups can never collide, and ``source_row_id`` increases within a group in
    the order the values are written -- which is what makes the surviving representative
    deterministic and checkable.
    """
    rows: list[dict[str, Any]] = []
    row_index = 0
    for position, group in enumerate(groups):
        observation_date = _FIRST_DATE + timedelta(days=position // codes_per_date)
        ts_code = f"S{position % codes_per_date:04d}"
        for value in group["values"]:
            rows.append(
                {
                    "observation_date": observation_date,
                    "ts_code": ts_code,
                    "factor_id": factor_id,
                    "factor_version": 1,
                    "source_row_id": f"r{row_index:06d}",
                    "raw_value": value,
                }
            )
            row_index += 1
    return pd.DataFrame(rows)


def histogram(groups: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Class counts of an already-built sample, recomputed from the groups themselves."""
    counts = {name: 0 for name in CLASSES}
    for group in groups:
        counts[str(group["class"])] += 1
    return counts


def build_factor_sample(
    label: str, measured: Mapping[str, int], *, sample_keys: int, seed: int
) -> dict[str, Any]:
    """One factor's anonymized sample, with the measurement it was scaled from beside it."""
    ratios = class_ratios(measured)
    allocated = allocate(ratios, sample_keys)
    generator = random.Random(f"{seed}:{label}")
    groups = build_groups(generator, measured, allocated)
    return {
        "factor_label": label,
        "source_factor_id": FACTOR_LABELS[label],
        "measured": dict(measured),
        "measured_ratios": {name: round(value, 9) for name, value in sorted(ratios.items())},
        "sample_keys": sample_keys,
        "allocated": dict(sorted(allocated.items())),
        "duplicated_multiplicity": duplicated_multiplicity(measured),
        "groups": groups,
    }


def build_fixture(census: Mapping[str, Any], *, sample_keys: int, seed: int) -> dict[str, Any]:
    crossval = census["crossval"]["factors"]
    samples = {
        label: build_factor_sample(
            label, crossval[label]["measured"], sample_keys=sample_keys, seed=seed
        )
        for label in sorted(FACTOR_LABELS)
        if label in crossval
    }
    missing = sorted(set(FACTOR_LABELS) - set(samples))
    if missing:
        raise ValueError(f"the census does not measure {', '.join(missing)}")
    return {
        "fixture_id": "DEDUP_REGRESSION_GP017_GP077_V1",
        "purpose": (
            "C3.4: prove duplicate rows are counted and that a conflicting key contributes "
            "nothing to clean, on the shapes the real source was measured to have"
        ),
        "anonymization": (
            "business keys and values are synthetic; only the measured class distribution, "
            "multiplicity and within-group usable-row pattern are carried over from the source"
        ),
        "seed": seed,
        "factors": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-keys", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()

    census_text = args.census.read_bytes()
    census = json.loads(census_text)
    if census.get("self_test", {}).get("status") != "PASS":
        raise ValueError("refusing to build a fixture from a census whose self-test did not pass")
    if census.get("crossval", {}).get("status") != "PASS":
        raise ValueError("refusing to build a fixture from a census that did not reproduce itself")

    fixture = build_fixture(census, sample_keys=args.sample_keys, seed=args.seed)
    fixture["provenance"] = {
        "census_artifact": str(args.census),
        "census_sha256": hashlib.sha256(census_text).hexdigest(),
        "census_script": "research_v2/scripts/census_duplicate_key_shapes.py",
        "census_script_sha256": hashlib.sha256(
            (Path(__file__).resolve().parent / "census_duplicate_key_shapes.py").read_bytes()
        ).hexdigest(),
        "generator_script": "research_v2/scripts/build_dedup_regression_fixture.py",
        "generator_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
        "measured_on": census["as_of_date"],
        "source_tables": {
            label: crossval["source_table"]
            for label, crossval in census["crossval"]["factors"].items()
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    document = json.dumps(fixture, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    args.output.write_text(document, encoding="utf-8")

    # Verify what was written, not what was built. The in-memory sample has the allocated
    # histogram by construction -- build_groups emits exactly allocated[class] groups per
    # class -- so checking it there would be a check that cannot fail. What can genuinely
    # go wrong is the round trip: a null that becomes a string, a float that loses a bit, a
    # group that stops expanding to the rows it describes. So the file is read back and
    # re-expanded, and the result has to be the same frame.
    reread = json.loads(args.output.read_text(encoding="utf-8"))
    for label, sample in sorted(reread["factors"].items()):
        written = histogram(sample["groups"])
        if written != sample["allocated"]:
            raise ValueError(f"{label}: the written sample has the wrong shape: {written}")
        pd.testing.assert_frame_equal(
            expand_rows(sample["groups"], factor_id=label),
            expand_rows(fixture["factors"][label]["groups"], factor_id=label),
        )
    for label, sample in sorted(fixture["factors"].items()):
        rows = sum(len(group["values"]) for group in sample["groups"])
        print(
            f"{label}: {sample['sample_keys']} keys, {len(sample['groups'])} groups, "
            f"{rows} rows, multiplicity {sample['duplicated_multiplicity']}"
        )
        print(f"{label}: re-read from disk and re-expanded identically")
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
