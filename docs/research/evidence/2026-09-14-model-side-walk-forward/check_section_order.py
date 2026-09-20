#!/usr/bin/env python3
"""Maintenance check: is problems_found.md still in chronological section order?

This exists because the mistake it catches has already happened TWICE, and both times the cause was
the same: appending a new section with apply_patch, using the previous section's closing
"safety declaration + driver sha256" lines as the context anchor. Those two lines are boilerplate that
ends nearly every section, so the anchor is NOT unique, apply_patch matches the FIRST occurrence, and
the new section silently lands in the middle of the file. It happened with §CC (which landed before
§CB) and again with §CD (which landed between §CB and §CC), and in both cases nothing else looked
wrong -- the file applied cleanly and only the section order revealed it.

Run this after every append to problems_found.md. It checks nine things:
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
  7. every non-blank line of corrections_ledger.md must have an even count of "**" outside code spans.
     The ledger has no legitimate multi-line bold -- each A/B/C row is one physical line, and a D rule's
     in-place correction markers are indented ">" lines of their own -- so an orphan "**" left by a marker
     leaks bold into the next line. Checks 5 and 6 cover table rows only, which is why two such defects
     (D13 and D14, both introduced while adding markers in one commit) shipped undetected. An earlier
     version of check 7 matched only lines starting with "<n>. **" and so still missed them, because the
     markers sit on their own indented lines; its negative test passed the defective file and forced the
     wider scope.
  8. no line of any pack markdown file may contain "****" outside a code span. Four adjacent asterisks are
     what a bold run closing immediately against another bold run opening produces; markdown does not
     render them as intended and the emphasis leaks. This recurred about eight times while adding in-place
     correction markers this session (B26, A16, D9, D14, DA.3, the DC heading, C10, B34) and every time it
     was found only by counting asterisks, never by reading. Two instances had already shipped and sat in
     the file for days: the BU section heading and a BZ revision paragraph. Checks 5 and 7 cannot catch it,
     because "****" is an EVEN count of "**" -- the line still balances, it just renders wrong. The fix is
     always the same: put a space between the two bold runs, which leaves the marker count unchanged and so
     preserves every downstream pairing.
  9. RATCHET on CommonMark emphasis leaks, measured with a real parser (markdown-it-py) rather than by
     counting asterisks. The mechanism, verified case by case against the parser: a "**" run is
     RIGHT-flanking (and so able to close) only if it is not preceded by whitespace AND either (a) not
     preceded by punctuation or (b) preceded by punctuation and followed by whitespace or punctuation. So a
     run PRECEDED BY PUNCTUATION and FOLLOWED BY a letter or digit can only OPEN, never close, and when no
     other closer is available its literal "**" reaches the HTML. The pack's idiom
     "**篡改**」**——**理由是" does this: the second run follows a full-width bracket and precedes a CJK
     letter. Contrast "**篡改**——**理由是**", which renders correctly because that closer is preceded by a
     letter. Checks 5, 7 and 8 cannot see any of this: they count asterisks, and the offending blocks have
     an EVEN count -- flanking, not parity, decides pairing. An earlier draft of this check's message
     described the cause as "a '**' followed directly by a CJK letter", which is wrong: "**篡改**理由"
     renders fine. The message was corrected by testing five minimal cases against the parser. A whole-document parse found 96 leaked "**" in
     problems_found.md and 66 in README.md, in 53 blocks, of which 44 predate this session. The 9 blocks in
     sections CX-DG were repaired (space insertion where that restores the intended bold, marker deletion
     otherwise, always with rendered text verified unchanged), bringing the total to LEAK_BUDGET below. The
     budgets are the MEASURED post-repair counts, not aspirational ones -- the ledger's was first written as
     6 by guessing and the gate immediately rejected the real file at 8, which is the ratchet working. The
     44 pre-existing blocks were deliberately NOT mass-repaired: an automated whole-file repair strips bold
     from 37 spans, 17 of which are polarity-bearing negations ("没有", "不是", "不可"), and in this pack bold
     marks the citable/not-citable distinction, so silently un-bolding a negation in text whose authoring
     intent cannot be verified here is worse than leaving a stray asterisk visible. The ratchet keeps the
     count from growing; fixing the remainder properly needs per-block human judgement.

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


def find_unbalanced_ledger_lines(text: str, filename: str) -> list[str]:
    """Flag any ledger line whose bold runs do not close within that line (code spans excluded).

    The ledger has no legitimate multi-line bold: every A/B/C row is one physical line, and a D rule's
    in-place correction markers are indented `> ` lines of their own. So EVERY non-blank line must carry an
    even count of "**" outside code spans.

    THE FIRST VERSION OF THIS CHECK WAS SCOPED WRONG, and its own negative test caught it. It matched only
    lines beginning with "<n>. **" -- i.e. the numbered D rule itself -- but a D rule's correction markers
    live on SEPARATE indented lines, so an orphan "**" left at the end of a marker (the exact defect that
    shipped in D13 and D14) was never inspected and the gate passed on the defective file. This is the same
    scope error as §CW.1 (comparing a 27-row statistic with an 81-row one) and §CZ.8 (applying a table-row
    check only to problems_found.md when the defect was in corrections_ledger.md): the check's scope must
    cover where the defect actually lives, and the only way to confirm that is to inject the defect and
    require the gate to fail.
    """
    problems = []
    for n, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        no_code = re.sub(r"`[^`]*`", "", line)
        if no_code.count("**") % 2 != 0:
            problems.append(
                f"{filename} line {n}: bold run does not close within the line ({no_code.count('**')} "
                f"'**' outside code spans, odd). The ledger has no legitimate multi-line bold, so an orphan "
                f"'**' (typically left by an in-place correction marker) leaks bold into the next line; "
                f"line starts: {line.strip()[:70]!r}")
    return problems


def find_bold_collisions(text: str, filename: str) -> list[str]:
    """Flag "****" outside code spans: a bold run closing directly against another one opening.

    Checks 5 and 7 cannot catch this, because four asterisks are an even count of "**" so the line still
    balances. It recurred about eight times while adding in-place markers this session, and two instances
    had already shipped, so it gets its own check.
    """
    problems = []
    for n, line in enumerate(text.split("\n"), 1):
        # Replace code spans with a PLACEHOLDER, not with the empty string. Deleting them would make two
        # bold markers that were separated by a code span (e.g. **`N/A`**) look adjacent and produce a
        # false "****". This bug was found on the check's first run, which flagged ~40 lines that all came
        # from bolded code spans rather than real collisions.
        no_code = re.sub(r"`[^`]*`", "\x00", line)
        if "****" in no_code:
            at = no_code.index("****")
            problems.append(
                f"{filename} line {n}: four adjacent asterisks (a bold run closing against another "
                f"opening) render wrong and leak emphasis; insert a space between the two runs, which "
                f"keeps the '**' count unchanged so downstream pairing is unaffected. Context: "
                f"{no_code[max(0, at - 40):at + 40]!r}")
    return problems


# Total leaked "**" in rendered HTML across the three pack files, measured by markdown-it-py.
# A ratchet, not a target of zero: the residual sits in blocks authored before this session, where an
# automated repair would strip bold from polarity-bearing negations. See check 9's docstring.
LEAK_BUDGET = {"problems_found.md": 78, "README.md": 66, "corrections_ledger.md": 8}


def count_emphasis_leaks(text: str) -> int:
    """Count literal '**' that survive into rendered HTML outside <code> spans."""
    try:
        from markdown_it import MarkdownIt
    except ImportError:
        return -1                      # parser unavailable: skip rather than guess
    html = MarkdownIt("commonmark").render(text)
    return len(re.findall(r"\*\*", re.sub(r"<code>.*?</code>", "", html, flags=re.S)))


def find_emphasis_leak_overage(files: dict[str, str]) -> list[str]:
    """Flag any file whose rendered-HTML '**' leak count exceeds its ratcheted budget."""
    problems = []
    for fname, text in files.items():
        budget = LEAK_BUDGET.get(fname)
        if budget is None:
            continue
        n = count_emphasis_leaks(text)
        if n < 0:
            problems.append(f"{fname}: markdown-it-py not installed, so the emphasis-leak ratchet could "
                            f"not be evaluated; install it or inspect the file manually")
            continue
        if n > budget:
            problems.append(
                f"{fname}: {n} literal '**' leak into rendered HTML, above the ratcheted budget of "
                f"{budget}. Cause: under CommonMark's flanking rules a '**' run that is PRECEDED BY "
                f"PUNCTUATION and FOLLOWED BY a non-whitespace non-punctuation character (a CJK letter or "
                f"digit) is left-flanking only, so it can OPEN emphasis but never CLOSE it; if no other "
                f"closer is available the run renders literally. The pack's idiom "
                f"'**篡改**」**——**理由是' triggers this: the second '**' follows a full-width bracket and "
                f"precedes 理, so it opens instead of closing. Note '**篡改**——**理由是**' is FINE (the "
                f"closer there is preceded by a letter), so the fix is contextual: insert a space or "
                f"re-punctuate so the closer is preceded by a letter, or remove the stray run. A parity "
                f"count cannot catch this -- the offending blocks have an EVEN '**' count. Do NOT raise "
                f"the budget without recording why in the ledger.")
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
    file_texts = {}
    for fname in PACK_FILES:
        fpath = HERE / fname
        if fpath.is_file():
            ftext = fpath.read_text(encoding="utf-8")
            file_texts[fname] = ftext
            problems.extend(find_unbalanced_table_rows(ftext, fname))
            problems.extend(find_bold_collisions(ftext, fname))
            if fname == "corrections_ledger.md":
                problems.extend(find_unbalanced_ledger_lines(ftext, fname))
    problems.extend(find_emphasis_leak_overage(file_texts))

    print(f"{TARGET.name}: {len(found)} sections ({found[0] if found else '-'}..{found[-1] if found else '-'}), "
          f"{sum(len(v) for v in subs.values())} subsections, "
          f"{len(_declarations(text))} safety declarations")
    if problems:
        print("FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("OK: chronological order, no duplicate headings, all subsections contiguous, no duplicated "
          "declarations, all table-row bold runs balanced, no unescaped pipes in table code spans, "
          "all ledger lines bold-balanced, no **** bold collisions, emphasis leaks within budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
