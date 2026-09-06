"""Every command in the reproduction runbook has to name a script and flags that exist.

Task 4.2 of the frozen-set plan is ticked on the strength of this runbook, and nothing had
checked it: a runbook is followed under pressure, often at the end of a fifteen-hour run, and a
renamed flag or a moved script turns the last step into a debugging session at the worst possible
moment. Verified by hand once on 2026-09-06 -- 17 (script, flag) pairs, all present -- and made
permanent here so the next edit to either side is checked too.

Flags are collected from the scripts' ASTs rather than by running ``--help``: a string literal
can appear in a docstring or an error message without being an option, and eleven subprocesses
that each import dolphindb cost more than parsing eleven files.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

# tests/unit/scripts/<this file> -> parents[3] is the research_v2 package root.
_RESEARCH_V2 = Path(__file__).resolve().parents[3]
_RUNBOOK = _RESEARCH_V2 / "runbooks" / "real-walk-forward-reproduction.md"
_SCRIPTS = _RESEARCH_V2 / "scripts"
_SCRIPT_REFERENCE = re.compile(r"scripts/([A-Za-z0-9_]+\.py)")
_FLAG = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")
_CODE_BLOCK = re.compile(r"```[a-z]*\n(.*?)```", re.DOTALL)


def _command_lines() -> list[str]:
    """The lines someone will copy out of the runbook, with continuations joined."""
    text = _RUNBOOK.read_text(encoding="utf-8")
    lines: list[str] = []
    for block in _CODE_BLOCK.findall(text):
        lines.extend(block.replace("\\\n", " ").splitlines())
    return lines


def _referenced_pairs() -> set[tuple[str, str]]:
    """(script, flag) pairs, attributing each flag to the script named before it."""
    pairs: set[tuple[str, str]] = set()
    for line in _command_lines():
        current: str | None = None
        for part in re.split(r"(scripts/[A-Za-z0-9_]+\.py)", line):
            if part.startswith("scripts/"):
                current = part.split("/")[-1]
            elif current:
                pairs.update((current, flag) for flag in _FLAG.findall(part))
    return pairs


def _declared_flags(script: str) -> set[str]:
    """Flags the script actually registers, read from its ``add_argument`` calls."""
    tree = ast.parse((_SCRIPTS / script).read_text(encoding="utf-8"))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func
        attribute = name.attr if isinstance(name, ast.Attribute) else getattr(name, "id", "")
        if attribute != "add_argument":
            continue
        for argument in node.args:
            if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
                continue
            if argument.value.startswith("--"):
                flags.add(argument.value)
    return flags


def test_the_runbook_exists_and_still_carries_commands() -> None:
    """Guards the guard: a regex that stops matching would make every other test here vacuous."""
    assert _RUNBOOK.exists()
    lines = _command_lines()
    assert len(lines) > 10, "the runbook's code blocks no longer parse as commands"
    assert len(_referenced_pairs()) >= 15, "flag extraction stopped finding the documented flags"


def test_every_script_the_runbook_names_exists() -> None:
    scripts = {script for script, _ in _referenced_pairs()}

    assert scripts, "no scripts were extracted, so this test would pass vacuously"
    assert sorted(script for script in scripts if not (_SCRIPTS / script).exists()) == []


def test_every_flag_the_runbook_passes_is_one_the_script_registers() -> None:
    """The check that matters: a flag in prose is a flag someone will type."""
    unknown = sorted(
        (script, flag)
        for script, flag in _referenced_pairs()
        if flag not in _declared_flags(script)
    )

    assert unknown == []


def test_the_flags_documented_in_prose_are_registered_too() -> None:
    """Two flags the runbook explains in prose rather than in a code block.

    ``--factor-set`` selects what the coverage gate measures (``frozen17`` versus
    ``c62pass11``) and ``--minimum-factors`` goes with it; both are documented in sentences, so
    the code-block extraction above never sees them. Listed explicitly rather than extracted
    heuristically, because guessing which prose dashes are flags would be worse than naming them.
    """
    declared = _declared_flags("verify_frozen_fold_coverage.py")

    assert {"--factor-set", "--minimum-factors"} <= declared


def test_the_flag_extraction_sees_a_flag_that_is_not_registered() -> None:
    """Control: the unknown-flag computation reports at all.

    Narrower than it first looks, and stated narrowly because a first version of this
    docstring claimed it would catch a collector that "returned every flag". It would not: a
    collector that pooled the flags of all eleven scripts would still not contain a made-up
    flag, so this control would pass while the real check went vacuous. The test below is the
    one that catches that.
    """
    pairs = _referenced_pairs() | {("run_real_walk_forward.py", "--definitely-not-a-flag")}
    unknown = sorted(
        (script, flag) for script, flag in pairs if flag not in _declared_flags(script)
    )

    assert ("run_real_walk_forward.py", "--definitely-not-a-flag") in unknown


def test_flags_are_attributed_to_the_script_that_declares_them_not_pooled() -> None:
    """The control that actually matters: a pooled collector makes the real check vacuous.

    If ``_declared_flags`` ignored which script it was asked about and returned every flag any
    runbook script registers, then every flag the runbook mentions would look registered and
    ``test_every_flag_the_runbook_passes_is_one_the_script_registers`` could never fail -- while
    still passing its own control above. Cross-attribution is the realistic way this breaks,
    because the runbook names eleven scripts and a per-line attribution bug is easy to write.
    """
    publisher = _declared_flags("publish_factor_daily_versions.py")
    walker = _declared_flags("run_real_walk_forward.py")

    assert "--months" in publisher
    assert "--months" not in walker, "a flag of one script must not be credited to another"
    assert "--folds" in walker
    assert "--folds" not in publisher
    assert publisher != walker
