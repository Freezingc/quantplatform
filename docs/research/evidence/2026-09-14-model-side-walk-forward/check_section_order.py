#!/usr/bin/env python3
"""Maintenance check: is problems_found.md still in chronological section order?

This exists because the mistake it catches has already happened TWICE, and both times the cause was
the same: appending a new section with apply_patch, using the previous section's closing
"safety declaration + driver sha256" lines as the context anchor. Those two lines are boilerplate that
ends nearly every section, so the anchor is NOT unique, apply_patch matches the FIRST occurrence, and
the new section silently lands in the middle of the file. It happened with §CC (which landed before
§CB) and again with §CD (which landed between §CB and §CC), and in both cases nothing else looked
wrong -- the file applied cleanly and only the section order revealed it.

Run this after every append to problems_found.md. It checks three things:
  1. every ## <LETTERS>. heading appears in the expected alphabetical/chronological sequence;
  2. there are no duplicate section headings;
  3. the ### subsections of each section are numbered contiguously, starting at .0 or .1 -- §BR starts
     at BR.0 by design ("validate the projection before pricing with it"), so a 0 start is legal and
     only gaps or duplicates are failures.

It reads only this pack's own markdown. No database, no network, no repository module, nothing written.
Exit status is 0 when the file is in order and 1 when it is not, so it can gate a commit.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "problems_found.md"
SEC = re.compile(r"^## ([A-Z]{1,3})\. ")
SUB = re.compile(r"^### ([A-Z]{1,3})\.(\d+) ")


def section_order(n: int) -> str:
    """0 -> 'A', 25 -> 'Z', 26 -> 'AA', matching the AA..CD scheme this file uses."""
    out = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        out = chr(65 + r) + out
    return out


def main() -> int:
    lines = TARGET.read_text(encoding="utf-8").split("\n")
    found = [SEC.match(l).group(1) for l in lines if SEC.match(l)]
    problems = []

    dupes = {s for s in found if found.count(s) > 1}
    if dupes:
        problems.append(f"duplicate section headings: {sorted(dupes)}")

    expected = [section_order(i) for i in range(len(found))]
    if found != expected:
        first_bad = next((i for i, (a, b) in enumerate(zip(found, expected)) if a != b), None)
        problems.append(
            "sections are out of chronological order\n"
            f"    found   : {' '.join(found)}\n"
            f"    expected: {' '.join(expected)}\n"
            f"    first mismatch at index {first_bad}: found {found[first_bad]!r}, "
            f"expected {expected[first_bad]!r}\n"
            "    most likely cause: an append anchored on the boilerplate 'safety declaration +\n"
            "    driver sha256' closing lines, which are not unique -- anchor on text unique to\n"
            "    the section being extended, then re-run the lossless reorder if it landed wrong."
        )

    subs: dict[str, list[int]] = {}
    for l in lines:
        m = SUB.match(l)
        if m:
            subs.setdefault(m.group(1), []).append(int(m.group(2)))
    for sec, nums in sorted(subs.items()):
        start = nums[0]
        if start not in (0, 1):
            problems.append(f"section {sec} subsections start at .{start}, expected .0 or .1: {nums}")
        elif nums != list(range(start, start + len(nums))):
            problems.append(f"section {sec} subsections are not contiguous from .{start}: {nums}")
        if len(nums) != len(set(nums)):
            problems.append(f"section {sec} has duplicate subsection numbers: {nums}")

    print(f"{TARGET.name}: {len(found)} sections ({found[0] if found else '-'}..{found[-1] if found else '-'}), "
          f"{sum(len(v) for v in subs.values())} subsections")
    if problems:
        print("FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("OK: chronological order, no duplicate headings, all subsections contiguous")
    return 0


if __name__ == "__main__":
    sys.exit(main())
