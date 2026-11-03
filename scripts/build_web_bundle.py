#!/usr/bin/env python3
"""Bundle the pure-Python part of PyMAPF for the browser playground.

The web playground runs the *real* library under Pyodide rather than a
JavaScript re-implementation, which is only possible because the solvers depend
on nothing but the standard library. This script collects those modules into a
single JSON file that the page writes into Pyodide's virtual filesystem.

Modules that need matplotlib/numpy (``pymapf.viz``, ``pymapf.decentralized``)
are deliberately excluded: pulling them in would drag megabytes of wheels into
the page for code the playground never calls.

The module list is not hand-maintained. It is the import graph reachable from
the entry points the page uses (``import pymapf`` and ``pymapf.benchmark``),
found by reading each module's ``import`` statements -- so a new module that
``pymapf/__init__.py`` starts importing is bundled automatically, instead of
breaking ``import pymapf`` in the browser. Every bundled module is then checked
to import nothing outside ``pymapf`` and the standard library.

    python scripts/build_web_bundle.py [--output .docs/pymapf-bundle.json] [--check]
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# What the playground imports (see .docs/worker.js); everything they reach is
# bundled.
ENTRY_POINTS = ["pymapf/__init__.py", "pymapf/benchmark.py"]


def _module_file(module: str):
    """``pymapf.core.grid`` -> its repository-relative path, or ``None``."""
    base = os.path.join(*module.split("."))
    for candidate in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.isfile(os.path.join(ROOT, candidate)):
            return candidate.replace(os.sep, "/")
    return None


def _package_of(relative: str) -> str:
    """The package a module's relative imports are resolved against."""
    parts = relative[: -len(".py")].split("/")
    return ".".join(parts[:-1])  # ``__init__`` and plain modules alike


def _imports(relative: str):
    """Yield ``(module, is_internal)`` for every import statement in a file.

    Function-level imports count too: they run as soon as the function is
    called, so the module they name must be in the bundle all the same.
    """
    with open(os.path.join(ROOT, relative), encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), relative)
    package = _package_of(relative)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, alias.name.split(".")[0] == "pymapf"
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                anchor = package.split(".")
                anchor = anchor[: len(anchor) - (node.level - 1)]
                base = ".".join(anchor + ([node.module] if node.module else []))
            else:
                base = node.module
            if base.split(".")[0] != "pymapf":
                yield base, False
                continue
            yield base, True
            # ``from . import algorithms`` imports a submodule, not a name.
            for alias in node.names:
                if _module_file(base + "." + alias.name):
                    yield base + "." + alias.name, True


def collect_modules(entry_points=ENTRY_POINTS) -> List[str]:
    """Every pymapf module reachable from ``entry_points``, parents included.

    Raises ``SystemExit`` if a reachable module imports a third-party package:
    the bundle runs under bare Pyodide, where only the standard library exists.
    """
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    seen = set()
    stack = list(entry_points)
    impure = []
    while stack:
        relative = stack.pop()
        if relative in seen:
            continue
        seen.add(relative)
        for module, internal in _imports(relative):
            if not internal:
                if module.split(".")[0] not in stdlib:
                    impure.append("%s imports %s" % (relative, module))
                continue
            # Importing ``a.b.c`` runs ``a/__init__``, ``a/b/__init__`` first.
            parts = module.split(".")
            for depth in range(1, len(parts) + 1):
                found = _module_file(".".join(parts[:depth]))
                if found is not None and found not in seen:
                    stack.append(found)
    if impure:
        raise SystemExit(
            "the web bundle must be pure standard library:\n  " + "\n  ".join(impure)
        )
    return sorted(seen)


def build(output: str) -> dict:
    files = {}
    for relative in collect_modules():
        path = os.path.join(ROOT, relative)
        with open(path, encoding="utf-8") as handle:
            files[relative] = handle.read()

    from pymapf import __version__

    payload = {
        "version": __version__,
        "digest": hashlib.sha256(
            "".join(files[key] for key in sorted(files)).encode("utf-8")
        ).hexdigest()[:12],
        "files": files,
    }

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=0, sort_keys=True)

    size = os.path.getsize(output)
    print(
        "wrote %s  (%d modules, %.1f kB, pymapf %s, digest %s)"
        % (output, len(files), size / 1024, payload["version"], payload["digest"])
    )
    return payload


# Run with ``python -I -S``: no site-packages, no cwd, no PYTHONPATH -- the
# bundle alone plus the standard library, which is what Pyodide gives it.
_SMOKE_TEST = """
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
print("bundle ok: pymapf %s, %d solvers" % (pymapf.__version__,
      len(pymapf.available_solvers())))
"""


def check(bundle_path: str) -> None:
    """Unpack a bundle into a temporary directory and import it in a clean
    interpreter (standard library only), solving a tiny instance with every
    registered solver. Raises ``SystemExit`` on failure."""
    import subprocess
    import tempfile

    with open(bundle_path, encoding="utf-8") as handle:
        bundle = json.load(handle)
    with tempfile.TemporaryDirectory() as root:
        for relative, source in bundle["files"].items():
            path = os.path.join(root, *relative.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(source)
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", _SMOKE_TEST, root],
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        raise SystemExit(
            "bundle %s does not import cleanly:\n%s" % (bundle_path, result.stderr)
        )
    print(result.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, ".docs", "pymapf-bundle.json"),
        help="destination JSON file (default: .docs/pymapf-bundle.json)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="after building, import the bundle in a clean interpreter and "
        "solve a tiny instance with every solver",
    )
    args = parser.parse_args()
    build(args.output)
    if args.check:
        check(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
