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
  4. no section contains a duplicated closing "safety declaration" (the "驱动 sha256 仍为 ... 折缓存
     ... 完好。" block). This catches a DIFFERENT failure from the one above, introduced across the
     §CO/§CP/§CQ/§CR/§CS appends: a new section was built as `body = anchor + new_text` where `anchor`
     was the previous section's closing declaration, and then written as `file + body` while `file`
     ALREADY ended with `anchor` -- so the previous section's declaration was emitted twice, once as the
     tail of the old section and once as the head of the new. The two copies are byte-identical, so no
     number is wrong and the file reads fine; check (1) does not catch it (the duplicated text is not a
     heading) and neither does a `****` scan. It was found only by counting declarations per section.
     The check flags any section with two identical adjacent declaration blocks, or a trailing sentence
     of a declaration repeated immediately after itself.
  5. every markdown TABLE row (a line starting with "|") closes its bold runs within the line, i.e. the
     count of "**" outside code spans is even. A table row is a single physical line, so an unclosed "**"
     in one leaks into the next cell and garbles the render. This catches a real defect that shipped in
     ledger row A1: a stray "**" before a closing parenthesis paired with the next cell's "**", bolding
     "） | §BZ.3 / §BZ.4". Prose paragraphs and blockquotes are deliberately EXCLUDED -- markdown allows a
     bold run to span soft line breaks there, so requiring per-line balance would flag hundreds of
     legitimate multi-line bold passages. Only single-line elements (table rows) must balance.
  6. inside a table row, a "|" written literally in a code span must be escaped as "\\|", because GFM
     splits cells on an unescaped pipe even when it is inside backticks. An unescaped one silently splits
     the row into too many cells. This was found by writing the very row that documents it: ledger C10's
     text quoted `） | ` and `|` inside code spans, which gave that row 8 pipes instead of 6.

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


DECL_HEAD = "驱动 sha256 仍为"
DECL_TAIL = "折缓存 46 个文件完好。"


def _declarations(text: str) -> list[str]:
    """Every full closing safety-declaration block, as the exact substring from head to tail."""
    out = []
    i = 0
    while True:
        h = text.find(DECL_HEAD, i)
        if h < 0:
            break
        e = text.find(DECL_TAIL, h)
        if e < 0:
            break
        out.append(text[h:e + len(DECL_TAIL)])
        i = e + len(DECL_TAIL)
    return out


def find_duplicate_declarations(text: str) -> list[str]:
    """Flag duplicated closing declarations: two identical blocks adjacent, or a tail sentence repeated.

    Returns a list of human-readable problems (empty when clean).
    """
    problems = []
    decls = _declarations(text)
    for a, b in zip(decls, decls[1:]):
        if a == b:
            # they must be separated by only whitespace to count as the append-duplication bug
            ia = text.index(a)
            ib = text.index(b, ia + len(a))
            between = text[ia + len(a):ib]
            if between.strip() == "":
                problems.append(
                    f"duplicated closing safety declaration (two identical blocks separated only by "
                    f"whitespace); tail='{a[-40:]}' -- cause: an append built as `anchor + body` where "
                    f"the file already ended with `anchor`; delete one copy")
    # partial tail duplication: the final sentence repeated immediately after itself
    tail_line = f"停滞看门狗"
    for m in re.finditer(re.escape(DECL_TAIL) + r"\n(停滞看门狗[^\n]*?" + re.escape(DECL_TAIL) + r")", text):
        problems.append(
            f"partial duplicated declaration tail: '{DECL_TAIL}' immediately followed by a repeated "
            f"'停滞看门狗 ... {DECL_TAIL}' fragment")
    return problems


PACK_FILES = ("problems_found.md", "corrections_ledger.md", "README.md")


def find_unbalanced_table_rows(text: str, filename: str) -> list[str]:
    """Flag table rows whose bold runs do not close within the line (code spans excluded).

    Only lines starting with "|" are checked. Bold may legitimately span lines inside prose and
    blockquotes, so those are not checked; a table row cannot span lines, so its "**" count outside
    `code spans` must be even. Runs over every pack markdown file -- the defect that motivated this check
    (ledger row A1) lives in corrections_ledger.md, which the section-ordering part of this gate never
    looks at, so scoping the table check to problems_found.md alone would have missed it. Returns
    human-readable problems, empty when clean.
    """
    problems = []
    for n, line in enumerate(text.split("\n"), 1):
        if not line.strip().startswith("|"):
            continue
        for span in re.finditer(r"`[^`]*`", line):
            body = span.group(0)[1:-1]
            if "|" in body and "\\|" not in body:
                problems.append(
                    f"{filename} line {n}: table row has an unescaped pipe inside a code span "
                    f"({span.group(0)!r}) -- GFM splits cells on it, so the row gains a phantom column; "
                    f"write it as \\| instead")
        no_code = re.sub(r"`[^`]*`", "", line)
        if no_code.count("**") % 2 != 0:
            problems.append(
                f"{filename} line {n}: table row has an unclosed bold run ({no_code.count('**')} '**' "
                f"outside code spans, odd) -- a stray '**' in a table row leaks into the next cell and "
                f"garbles the render; row starts: {line.strip()[:70]!r}")
        if no_code.count("|") != 6 and filename == "corrections_ledger.md" \
                and re.match(r"^\| \*\*[ABCD]\d+\*\*", line.strip()):
            problems.append(
                f"{filename} line {n}: ledger row has {no_code.count('|')} pipes outside code spans, "
                f"expected 6 (4 cells + 2 delimiters); row starts: {line.strip()[:70]!r}")
    return problems


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

    text = TARGET.read_text(encoding="utf-8")
    problems.extend(find_duplicate_declarations(text))
    for fname in PACK_FILES:
        fpath = HERE / fname
        if fpath.is_file():
            problems.extend(find_unbalanced_table_rows(
                fpath.read_text(encoding="utf-8"), fname))

    print(f"{TARGET.name}: {len(found)} sections ({found[0] if found else '-'}..{found[-1] if found else '-'}), "
          f"{sum(len(v) for v in subs.values())} subsections, "
          f"{len(_declarations(text))} safety declarations")
    if problems:
        print("FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("OK: chronological order, no duplicate headings, all subsections contiguous, no duplicated "
          "declarations, all table-row bold runs balanced, no unescaped pipes in table code spans")
    return 0


if __name__ == "__main__":
    sys.exit(main())
