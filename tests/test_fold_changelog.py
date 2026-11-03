"""The release PR's changelog keeps the hand-written notes, under the release."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

from fold_changelog import fold  # noqa: E402

HEADER = "# Changelog\n\nIntro text.\n\n"

AS_RELEASE_PLEASE_WRITES_IT = (
    HEADER + "## [0.10.0](https://example.org/compare/v0.9.0...v0.10.0) (2026-11-03)\n"
    "\n"
    "\n"
    "### Bug Fixes\n"
    "\n"
    "* **rl:** stop the goal bonus being farmed ([abc1234](https://example.org))\n"
    "\n"
    "## [Unreleased]\n"
    "\n"
    "### Fixed\n"
    "\n"
    "- **The goal bonus could be farmed.** +8.6 against 0 over twenty steps.\n"
    "\n"
    "## [0.9.0]\n"
    "\n"
    "Old notes.\n"
)


def test_the_notes_move_under_the_new_release_above_the_commit_list():
    folded = fold(AS_RELEASE_PLEASE_WRITES_IT)
    assert folded.startswith(HEADER + "## [Unreleased]\n\n## [0.10.0](")
    release = folded.index("## [0.10.0]")
    notes = folded.index("- **The goal bonus could be farmed.**")
    commits = folded.index("### Bug Fixes")
    older = folded.index("## [0.9.0]")
    assert release < notes < commits < older
    # The Unreleased section is left empty, ready for the next release.
    unreleased = folded.index("## [Unreleased]")
    assert folded[unreleased:release].strip() == "## [Unreleased]"
    assert folded.endswith("## [0.9.0]\n\nOld notes.\n")


def test_folding_twice_changes_nothing():
    once = fold(AS_RELEASE_PLEASE_WRITES_IT)
    assert fold(once) == once


def test_a_changelog_between_releases_is_left_alone():
    between = HEADER + "## [Unreleased]\n\n- something\n\n## [0.9.0]\n\nOld.\n"
    assert fold(between) == between


def test_an_empty_unreleased_section_keeps_the_generated_entry():
    text = (
        HEADER
        + "## [0.9.1](https://example.org) (2026-12-01)\n\n### Bug Fixes\n\n* x\n\n"
        "## [Unreleased]\n\n## [0.9.0]\n\nOld.\n"
    )
    folded = fold(text)
    assert folded.index("## [Unreleased]") < folded.index("## [0.9.1]")
    assert "### Bug Fixes\n\n* x" in folded
    assert folded.endswith("## [0.9.0]\n\nOld.\n")


def test_the_real_changelog_is_in_the_shape_this_expects():
    path = os.path.join(os.path.dirname(HERE), "CHANGELOG.md")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    assert text.startswith("# Changelog\n")
    assert "\n## [Unreleased]\n" in text
