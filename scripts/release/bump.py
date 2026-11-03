"""Bump the version and date the changelog for a release.

``python scripts/release/bump.py patch|minor|major`` updates, in place:

- ``pymapf/__init__.py``: ``__version__``, the single source of truth that
  ``pyproject.toml`` reads;
- ``CITATION.cff``: ``version``;
- ``CHANGELOG.md``: the notes under ``## [Unreleased]`` become
  ``## [X.Y.Z] - YYYY-MM-DD``, and an empty ``## [Unreleased]`` is left above
  them for the next change. If nobody wrote notes, the release's commit
  subjects are listed instead, so no release has an empty entry.

and prints the new version. Below 1.0, a ``major`` bump moves the minor
version: semantic versioning reserves 0.x for an API that may still change,
and leaving 0.x is a decision for a person, not for a commit message.
"""

from __future__ import annotations

import datetime
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VERSION_LINE = re.compile(r'^__version__ = "(\d+)\.(\d+)\.(\d+)"$', re.MULTILINE)
CITATION_LINE = re.compile(r"^version: .*$", re.MULTILINE)
UNRELEASED = "## [Unreleased]"


def next_version(current: str, part: str) -> str:
    major, minor, patch = (int(x) for x in current.split("."))
    if part == "major" and major == 0:
        part = "minor"
    if part == "major":
        return "%d.0.0" % (major + 1)
    if part == "minor":
        return "%d.%d.0" % (major, minor + 1)
    if part == "patch":
        return "%d.%d.%d" % (major, minor, patch + 1)
    raise ValueError("part must be patch, minor or major, not %r" % part)


def date_changelog(text: str, version: str, date: str, fallback: str = "") -> str:
    """Move the Unreleased notes under a heading for ``version``."""
    lines = text.split("\n")
    start = lines.index(UNRELEASED)
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")),
        len(lines),
    )
    notes = "\n".join(lines[start + 1 : end]).strip("\n") or fallback.strip("\n")
    entry = [UNRELEASED, "", "## [%s] - %s" % (version, date), ""]
    if notes:
        entry += [notes, ""]
    return "\n".join(lines[:start] + entry + lines[end:])


def commit_list(messages) -> str:
    """Releasable commit subjects as a bullet list, for an entry with no notes."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plan import bump_for

    subjects = [m.split("\n", 1)[0] for m in messages if bump_for(m)]
    return "\n".join("- %s" % s for s in subjects)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    part = argv[0]

    init = os.path.join(ROOT, "pymapf", "__init__.py")
    with open(init, encoding="utf-8") as handle:
        source = handle.read()
    match = VERSION_LINE.search(source)
    if match is None:
        raise SystemExit('no __version__ = "X.Y.Z" line in %s' % init)
    current = ".".join(match.groups())
    version = next_version(current, part)
    with open(init, "w", encoding="utf-8") as handle:
        handle.write(VERSION_LINE.sub('__version__ = "%s"' % version, source))

    citation = os.path.join(ROOT, "CITATION.cff")
    with open(citation, encoding="utf-8") as handle:
        text = handle.read()
    with open(citation, "w", encoding="utf-8") as handle:
        handle.write(CITATION_LINE.sub("version: %s" % version, text, count=1))

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plan import last_tag, messages_since

    changelog = os.path.join(ROOT, "CHANGELOG.md")
    with open(changelog, encoding="utf-8") as handle:
        text = handle.read()
    fallback = commit_list(messages_since(last_tag()))
    today = datetime.date.today().isoformat()
    with open(changelog, "w", encoding="utf-8") as handle:
        handle.write(date_changelog(text, version, today, fallback))

    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
