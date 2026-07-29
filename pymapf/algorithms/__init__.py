"""Concrete MAPF algorithms built on the :mod:`pymapf.core` framework.

Every solver registers itself by name, so ``pymapf.solve(problem, "lacam")``
works without importing anything from this package directly. See
``REFERENCES.md`` for the citation behind each one.
"""

from .cbs import ConflictBasedSearch
from .lacam import LaCAM
from .lns import LargeNeighborhoodSearch
from .pibt import PIBT, pibt_step
from .prioritized_planning import PrioritizedPlanning
from .search import astar, dijkstra, distance_table, focal_astar, weighted_astar
from .sipp import safe_intervals, sipp
from .space_time_astar import space_time_astar
from .weighted_cbs import WeightedCBS

__all__ = [
    # multi-agent solvers
    "ConflictBasedSearch",
    "WeightedCBS",
    "PrioritizedPlanning",
    "PIBT",
    "LaCAM",
    "LargeNeighborhoodSearch",
    # single-agent primitives
    "space_time_astar",
    "sipp",
    "safe_intervals",
    "pibt_step",
    "astar",
    "dijkstra",
    "weighted_astar",
    "focal_astar",
    "distance_table",
]
