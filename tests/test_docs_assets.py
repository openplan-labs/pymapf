"""The published site's data files must be readable by a browser.

`.docs/` is uploaded verbatim to GitHub Pages and the playground reads these
files at runtime, so a file that Python can write but `JSON.parse` cannot read
is a broken page rather than a failing build -- and nothing else catches it.

The specific trap: `json.dump` writes a bare ``NaN`` token for a non-finite
float. Python's own loader accepts it, so a round-trip in Python looks fine.
It is not JSON, and every browser rejects the whole document. `strict=False` is
therefore *not* what these tests want -- `parse_constant` is how you refuse it.
"""

import json
import os

import pytest

DOCS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".docs"
)
ASSETS = os.path.join(DOCS, "assets")


def published_json():
    """Every .json under .docs/, which is exactly what Pages uploads."""
    found = []
    for directory, _, names in os.walk(DOCS):
        found.extend(
            os.path.join(directory, name) for name in names if name.endswith(".json")
        )
    return sorted(found)


def reject_constant(name):
    raise ValueError("%s is not valid JSON" % name)


@pytest.mark.parametrize("path", published_json(), ids=os.path.basename)
def test_published_json_is_valid_json(path):
    with open(path) as handle:
        # parse_constant fires for NaN, Infinity and -Infinity -- the three
        # things Python emits happily and no JSON parser accepts.
        json.load(handle, parse_constant=reject_constant)


def test_the_benchmark_the_learning_section_reads_is_present_and_shaped():
    path = os.path.join(ASSETS, "rl-benchmark.json")
    with open(path) as handle:
        report = json.load(handle, parse_constant=reject_constant)

    assert report["settings"], "the learning section renders from this list"
    for setting in report["settings"]:
        assert setting["family"] and setting["rows"]
        methods = {row["method"] for row in setting["rows"]}
        # The page's whole argument is learned-versus-optimal on the same
        # instances, so both sides have to be in the artifact.
        assert "cbs" in methods
        assert any(method.startswith("ippo") for method in methods)
        for row in setting["rows"]:
            assert 0.0 <= row["success_rate"] <= 1.0
            # A method that solved nothing has no cost: null, never NaN.
            for key in ("mean_cost", "suboptimality"):
                assert row[key] is None or row[key] == row[key]


def test_every_asset_the_page_references_exists():
    with open(os.path.join(DOCS, "index.html")) as handle:
        page = handle.read()

    referenced = set()
    for attribute in ('src="', 'href="', 'poster="'):
        start = 0
        while True:
            found = page.find(attribute, start)
            if found < 0:
                break
            start = found + len(attribute)
            value = page[start : page.index('"', start)]
            if value.startswith(("http", "#", "data:", "mailto")):
                continue
            referenced.add(value.split("#")[0])

    missing = [
        value
        for value in sorted(referenced)
        if value and not os.path.exists(os.path.join(DOCS, value))
    ]
    assert not missing, "index.html links files that will 404 on Pages: %s" % missing
