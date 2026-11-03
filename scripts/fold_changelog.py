"""Fold the hand-written ``## [Unreleased]`` notes into a release-please entry.

PyMAPF's changelog has two authors. Contributors write the *why* -- measured
numbers, what was wrong, what a reader should do differently -- under
``## [Unreleased]``, in Keep a Changelog sections. release-please writes the
*what*: one line per conventional commit, under a heading for the new version.

release-please inserts its entry above the first version-like heading, and
``## [Unreleased]`` is one, so a release PR's changelog comes out as::

    ## [0.10.0](...) (2026-11-03)      <- release-please, the commit list
    ## [Unreleased]                     <- the notes this release is made of
    ## [0.9.0]

This script runs on the release PR's branch and turns that into::

    ## [Unreleased]

    ## [0.10.0](...) (2026-11-03)
    <the hand-written notes>
    <the commit list>
    ## [0.9.0]

It is idempotent: on a changelog with no release entry above
``## [Unreleased]``, or nothing written under it, it changes nothing.

Usage: ``python scripts/fold_changelog.py [CHANGELOG.md]``. Exits 0 either
way and says whether it changed the file.
"""

from __future__ import annotations

import re
import sys

UNRELEASED = "## [Unreleased]"
VERSION_HEADING = re.compile(r"^## \[\d+\.\d+\.\d+[^\]]*\]")


def fold(text: str) -> str:
    """Return ``text`` with the Unreleased notes moved into the release above."""
    lines = text.split("\n")
    try:
        unreleased = lines.index(UNRELEASED)
    except ValueError:
        return text

    release = next(
        (i for i in range(unreleased) if VERSION_HEADING.match(lines[i])), None
    )
    if release is None:
        return text  # no release entry above the notes: nothing to fold into

    end = next(
        (i for i in range(unreleased + 1, len(lines)) if lines[i].startswith("## ")),
        len(lines),
    )
    notes = "\n".join(lines[unreleased + 1 : end]).strip("\n")
    generated = "\n".join(lines[release + 1 : unreleased]).strip("\n")

    entry = [lines[release], ""]
    if notes:
        entry += [notes, ""]
    if generated:
        entry += [generated, ""]

    head = "\n".join(lines[:release]).rstrip("\n")
    tail = "\n".join(lines[end:])
    return "%s\n\n%s\n\n%s%s" % (head, UNRELEASED, "\n".join(entry), tail)


def main(argv=None) -> int:
    path = (argv or sys.argv[1:] or ["CHANGELOG.md"])[0]
    with open(path, encoding="utf-8") as handle:
        before = handle.read()
    after = fold(before)
    if after == before:
        print("%s: nothing to fold" % path)
        return 0
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(after)
    print("%s: folded the [Unreleased] notes into the new release entry" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
