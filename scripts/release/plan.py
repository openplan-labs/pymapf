"""Decide what a merge to main releases, from its Conventional Commits.

Reads the commits since the last ``v*`` tag (every merge releases, so in the
usual case that is exactly the merged pull request) and prints the part of
the version to bump, or nothing:

=====================================================  =========
commits                                                 bump
=====================================================  =========
``type!: ...`` or a ``BREAKING CHANGE:`` footer          ``major``
``feat: ...``                                           ``minor``
``fix: ...``, ``perf: ...``, ``revert: ...``            ``patch``
only ``docs/test/ci/chore/refactor/build/style``        (nothing)
=====================================================  =========

Commits that are not Conventional Commits (old history, a merge commit, the
formatter's "Apply formatting changes") count for nothing. Whether ``major``
really bumps the major version is :mod:`bump`'s call: below 1.0 it does not.

Usage::

    python scripts/release/plan.py            # commits since the last tag
    python scripts/release/plan.py v0.9.0     # commits since a given ref
"""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Iterable, Optional

SUBJECT = re.compile(r"^(?P<type>\w+)(?:\([^)]*\))?(?P<breaking>!)?: \S")
BREAKING_FOOTER = re.compile(r"^BREAKING[ -]CHANGE: ", re.MULTILINE)
RELEASING = {"feat": "minor", "fix": "patch", "perf": "patch", "revert": "patch"}
ORDER = ["", "patch", "minor", "major"]


def bump_for(message: str) -> str:
    """The bump one commit message asks for (``""`` for none)."""
    subject = message.strip().split("\n", 1)[0]
    match = SUBJECT.match(subject)
    if not match:
        return ""
    if match.group("breaking") or BREAKING_FOOTER.search(message):
        return "major"
    return RELEASING.get(match.group("type"), "")


def plan(messages: Iterable[str]) -> str:
    """The largest bump any of ``messages`` asks for."""
    return max((bump_for(m) for m in messages), key=ORDER.index, default="")


def last_tag() -> Optional[str]:
    result = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0", "--match", "v*"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def messages_since(ref: Optional[str]) -> list:
    span = "%s..HEAD" % ref if ref else "HEAD"
    out = subprocess.run(
        ["git", "log", "--no-merges", "--format=%B%x00", span],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [m.strip() for m in out.split("\x00") if m.strip()]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ref = argv[0] if argv else last_tag()
    print(plan(messages_since(ref)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
