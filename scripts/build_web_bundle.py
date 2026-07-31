#!/usr/bin/env python3
"""Bundle the pure-Python part of PyMAPF for the browser playground.

The web playground runs the *real* library under Pyodide rather than a
JavaScript re-implementation, which is only possible because the solvers depend
on nothing but the standard library. This script collects those modules into a
single JSON file that the page writes into Pyodide's virtual filesystem.

Modules that need matplotlib/numpy (``pymapf.viz``, ``pymapf.decentralized``)
are deliberately excluded: pulling them in would drag megabytes of wheels into
the page for code the playground never calls.

    python scripts/build_web_bundle.py [--output .docs/pymapf-bundle.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Every module the playground can reach, in import order.
MODULES = [
    "pymapf/__init__.py",
    "pymapf/core/__init__.py",
    "pymapf/core/grid.py",
    "pymapf/core/graph.py",
    "pymapf/core/heuristics.py",
    "pymapf/core/trace.py",
    "pymapf/core/solver.py",
    "pymapf/algorithms/__init__.py",
    "pymapf/algorithms/search.py",
    "pymapf/algorithms/space_time_astar.py",
    "pymapf/algorithms/sipp.py",
    "pymapf/algorithms/pibt.py",
    "pymapf/algorithms/lacam.py",
    "pymapf/algorithms/lns.py",
    "pymapf/algorithms/prioritized_planning.py",
    "pymapf/algorithms/cbs.py",
    "pymapf/algorithms/weighted_cbs.py",
    "pymapf/scenarios.py",
    "pymapf/benchmark.py",
]


def build(output: str) -> dict:
    files = {}
    for relative in MODULES:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, ".docs", "pymapf-bundle.json"),
        help="destination JSON file (default: .docs/pymapf-bundle.json)",
    )
    args = parser.parse_args()
    build(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
