"""The browser playground's bundle must import on a bare standard library.

The page writes ``.docs/pymapf-bundle.json`` into Pyodide and runs
``import pymapf``; a module missing from the bundle is a ``ModuleNotFoundError``
in the browser that no other test would see. So build a fresh bundle, unpack
it, and import it in an interpreter with no site-packages at all.
"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_IMPORT_AND_SOLVE = """
import sys
sys.path.insert(0, sys.argv[1])
import pymapf
from pymapf.benchmark import _supported
from pymapf.core import Agent, GridMap, MAPFProblem
assert pymapf.__file__.startswith(sys.argv[1]), pymapf.__file__
problem = MAPFProblem(
    GridMap([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    [Agent("a", (0, 0), (2, 2)), Agent("b", (2, 2), (0, 0))],
)
for name in pymapf.available_solvers():
    solution = pymapf.solve(problem, name, **_supported(name, {"time_limit": 0.2}))
    assert solution is not None and solution.is_valid(), name
print("ok")
"""


def test_a_freshly_built_bundle_imports_and_solves_without_site_packages(tmp_path):
    bundle_path = tmp_path / "bundle.json"
    subprocess.run(
        [
            sys.executable,
            os.path.join(ROOT, "scripts", "build_web_bundle.py"),
            "--output",
            str(bundle_path),
        ],
        check=True,
        capture_output=True,
    )
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    site = tmp_path / "site"
    for relative, source in bundle["files"].items():
        target = site.joinpath(*relative.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")

    # -I: ignore PYTHONPATH and the cwd; -S: no site-packages (so no numpy,
    # and no installed pymapf to fall back on) -- what Pyodide provides.
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", _IMPORT_AND_SOLVE, str(site)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_the_bundle_builder_refuses_a_third_party_import(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        import build_web_bundle
    finally:
        sys.path.pop(0)
    package = tmp_path / "pymapf"
    package.mkdir()
    (package / "__init__.py").write_text("from . import fast\n")
    (package / "fast.py").write_text("import numpy\n")
    monkeypatch.setattr(build_web_bundle, "ROOT", str(tmp_path))
    try:
        build_web_bundle.collect_modules(["pymapf/__init__.py"])
    except SystemExit as error:
        assert "pymapf/fast.py imports numpy" in str(error)
    else:
        raise AssertionError("a numpy import slipped into the bundle")
