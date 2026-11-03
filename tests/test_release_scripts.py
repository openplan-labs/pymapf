"""The release scripts: what a merge releases, and what the bump writes."""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts", "release"))

from bump import date_changelog, next_version  # noqa: E402
from plan import bump_for, plan  # noqa: E402


@pytest.mark.parametrize(
    "message, expected",
    [
        ("fix(cbs): reopen improved nodes", "patch"),
        ("perf: cache distance tables", "patch"),
        ("revert: undo the heuristic change", "patch"),
        ("feat(rl): lifelong benchmark", "minor"),
        ("feat!: drop Python 3.10", "major"),
        ("fix(swarm)!: reject mixed composite outputs", "major"),
        ("fix(core): x\n\nBREAKING CHANGE: agents may not share a goal", "major"),
        ("docs(readme): typo", ""),
        ("chore(deps): update numpy", ""),
        ("ci: cache pip", ""),
        ("Merge pull request #12 from someone/branch", ""),
        ("Apply formatting changes", ""),
        ("Fix the heuristic", ""),  # not a Conventional Commit: counts for nothing
    ],
)
def test_each_commit_asks_for_the_bump_its_type_implies(message, expected):
    assert bump_for(message) == expected


def test_a_merge_releases_the_largest_bump_among_its_commits():
    assert plan(["docs: a", "fix: b", "feat: c"]) == "minor"
    assert plan(["fix: a", "fix(x)!: b"]) == "major"
    assert plan(["docs: a", "ci: b"]) == ""
    assert plan([]) == ""


@pytest.mark.parametrize(
    "current, part, expected",
    [
        ("0.9.0", "patch", "0.9.1"),
        ("0.9.3", "minor", "0.10.0"),
        ("0.9.3", "major", "0.10.0"),  # below 1.0 a breaking change bumps the minor
        ("1.4.2", "major", "2.0.0"),
        ("1.4.2", "minor", "1.5.0"),
    ],
)
def test_next_version(current, part, expected):
    assert next_version(current, part) == expected


def test_next_version_rejects_an_unknown_part():
    with pytest.raises(ValueError):
        next_version("0.9.0", "huge")


CHANGELOG = (
    "# Changelog\n\nIntro.\n\n## [Unreleased]\n\n### Fixed\n\n- a thing\n\n"
    "## [0.9.0]\n\nOld.\n"
)


def test_the_unreleased_notes_become_the_release_entry():
    dated = date_changelog(CHANGELOG, "0.10.0", "2026-11-03")
    assert dated == (
        "# Changelog\n\nIntro.\n\n## [Unreleased]\n\n## [0.10.0] - 2026-11-03\n\n"
        "### Fixed\n\n- a thing\n\n## [0.9.0]\n\nOld.\n"
    )


def test_a_release_without_notes_lists_its_commits():
    empty = "# Changelog\n\n## [Unreleased]\n\n## [0.9.0]\n\nOld.\n"
    dated = date_changelog(empty, "0.9.1", "2026-11-03", "- fix: a")
    assert "## [0.9.1] - 2026-11-03\n\n- fix: a\n\n## [0.9.0]" in dated


def test_the_real_files_are_in_the_shape_the_bump_expects():
    root = os.path.dirname(HERE)
    with open(os.path.join(root, "CHANGELOG.md"), encoding="utf-8") as handle:
        assert "\n## [Unreleased]\n" in handle.read()
    with open(os.path.join(root, "pymapf", "__init__.py"), encoding="utf-8") as h:
        assert '\n__version__ = "' in h.read()
    with open(os.path.join(root, "CITATION.cff"), encoding="utf-8") as handle:
        assert "\nversion: " in handle.read()
