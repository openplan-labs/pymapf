"""Pluggable, admissible grid heuristics.

All heuristics operate on integer cells of any dimension -- ``(row, col)`` on
a :class:`~pymapf.core.grid.GridMap`, ``(layer, row, col)`` on a
:class:`~pymapf.core.voxel.VoxelGrid`, and anything longer -- and return a
float estimate of the cost to travel between them. They are registered by name
so that solvers can be configured with a string (e.g. ``"manhattan"``) instead
of hard-coding a single global heuristic as the legacy ``common.HEURISTIC``
flag did.

Coordinate convention across the framework: a cell is ``(row, col)`` which maps
to ``grid[row][col]`` (i.e. ``(y, x)``); a voxel prepends the layer.

Which one is admissible depends on the move set and on the unit edge cost the
solvers use: ``manhattan`` for face-adjacent moves (4 on a plane, 6 in a
volume), ``chebyshev`` when diagonals are allowed (8 or 26), ``euclidean``
always. ``octile`` prices a planar diagonal at ``sqrt 2`` and so overestimates
under unit cost; it is kept for the classic grid-search convention.

Solvers default to ``heuristic=None``, which :func:`resolve_heuristic` turns
into the admissible choice for the map and move set (:func:`default_heuristic`)
-- so ``allow_diagonals=True`` no longer silently pairs diagonal moves with
Manhattan distance, and a general graph gets exact distances instead of a
coordinate formula it cannot evaluate.
"""

from __future__ import annotations

import weakref
from math import sqrt
from typing import Callable, Dict, Optional, Tuple

Cell = Tuple[int, ...]
Heuristic = Callable[[Cell, Cell], float]

_SQRT2 = sqrt(2)


def manhattan(a: Cell, b: Cell) -> float:
    """L1 distance. Admissible for face-adjacent moves in any dimension."""
    return float(sum(abs(x - y) for x, y in zip(a, b)))


def euclidean(a: Cell, b: Cell) -> float:
    """Straight-line distance. Admissible for any move set."""
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def chebyshev(a: Cell, b: Cell) -> float:
    """L-infinity distance. Admissible with diagonals under unit move cost."""
    return float(max(abs(x - y) for x, y in zip(a, b)))


def octile(a: Cell, b: Cell) -> float:
    """Octile distance: the planar formula (diagonals cost ``sqrt 2``, straight
    moves ``1``) on the two largest axis deltas, plus the rest linearly."""
    deltas = sorted((abs(x - y) for x, y in zip(a, b)), reverse=True)
    if len(deltas) < 2:
        return float(sum(deltas))
    first, second = deltas[0], deltas[1]
    return (first + second) + (_SQRT2 - 2) * min(first, second) + sum(deltas[2:])


HEURISTICS: Dict[str, Heuristic] = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "chebyshev": chebyshev,
    "octile": octile,
}


def true_distance(
    graph, goal: Optional[Cell] = None, allow_diagonals: bool = False
) -> Heuristic:
    """An exact, admissible heuristic: the real distance to ``goal``.

    Runs one backward Dijkstra from the goal and closes over the resulting
    table, so it costs O(V log V) once and O(1) per query. Unlike the geometric
    heuristics it accounts for walls, which is what makes it the right default
    on mazes and the *only* option on a graph without coordinates
    (:class:`~pymapf.core.graph.ExplicitGraph`).

    With ``goal=None`` the heuristic serves *every* goal: the first query for
    a target runs its Dijkstra and caches the table, so one heuristic can be
    handed to a multi-agent solver whose agents all have different goals.

    Nodes from which the goal is unreachable get ``inf``, which prunes them
    from any search that uses this heuristic.

    References:
        The "true distance heuristic" is standard practice in MAPF
        implementations; see e.g. Sturtevant, N.; Felner, A.; Barrer, M.;
        Schaeffer, J.; and Burch, N. 2009. *Memory-based heuristics for
        explicit state spaces.* IJCAI 2009: 609-614.
    """
    from ..algorithms.search import distance_table

    if goal is None:
        tables: Dict[Cell, Dict] = {}

        def any_goal(node: Cell, target: Cell) -> float:
            table = tables.get(target)
            if table is None:
                table = distance_table(graph, target, allow_diagonals=allow_diagonals)
                tables[target] = table
            return float(table.get(node, float("inf")))

        any_goal.goal = None
        return any_goal

    table = distance_table(graph, goal, allow_diagonals=allow_diagonals)

    def heuristic(node: Cell, target: Cell) -> float:
        if target != goal:
            # The table is goal-specific; fall back rather than lie.
            raise ValueError(
                "true_distance heuristic was built for goal %r, queried for %r"
                % (goal, target)
            )
        return float(table.get(node, float("inf")))

    heuristic.goal = goal  # lets callers detect/rebuild a stale table
    return heuristic


# One cached all-goals table per graph, so the many low-level searches of one
# solve (and of repeated solves on the same map) share their Dijkstras. Weak
# keys: the cache never keeps a graph alive.
_GRAPH_DEFAULTS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def default_heuristic(graph, allow_diagonals: bool = False) -> Heuristic:
    """The admissible heuristic for ``graph`` and its move set.

    Every move in this library costs one timestep, diagonal or not, so the
    right geometric bound depends on the moves allowed. Manhattan distance is
    exact on an open 4-connected grid but *overestimates* once diagonals are
    allowed -- one diagonal step covers two Manhattan units -- and an
    overestimating heuristic quietly costs A*, and every solver built on it,
    its optimality guarantee. Chebyshev distance is the tight admissible bound
    for 8- (and 26-) connected unit-cost moves.

    A graph that is not a lattice has no geometry to exploit, so it gets the
    exact distance tables of :func:`true_distance` (one Dijkstra per goal,
    cached per graph).
    """
    from .grid import GridMap
    from .voxel import VoxelGrid

    if isinstance(graph, (GridMap, VoxelGrid)):
        return chebyshev if allow_diagonals else manhattan
    try:
        per_graph = _GRAPH_DEFAULTS.setdefault(graph, {})
    except TypeError:  # not weak-referenceable: just build one
        return true_distance(graph, None, allow_diagonals=allow_diagonals)
    if allow_diagonals not in per_graph:
        per_graph[allow_diagonals] = true_distance(
            graph, None, allow_diagonals=allow_diagonals
        )
    return per_graph[allow_diagonals]


def resolve_heuristic(heuristic, graph, allow_diagonals: bool = False) -> Heuristic:
    """:func:`get_heuristic`, except ``None`` means :func:`default_heuristic`.

    This is what every search entry point calls, so ``heuristic=None`` (the
    default everywhere) is always admissible for the map it runs on.
    """
    if heuristic is None:
        return default_heuristic(graph, allow_diagonals)
    return get_heuristic(heuristic)


def get_heuristic(heuristic) -> Heuristic:
    """Resolve a heuristic from a name or pass through a callable.

    Accepts either a registered name (``str``) or a callable with the
    ``(Cell, Cell) -> float`` signature, so callers can supply custom
    heuristics without registering them.
    """
    if callable(heuristic):
        return heuristic
    try:
        return HEURISTICS[heuristic]
    except KeyError as error:
        raise ValueError(
            "Unknown heuristic %r. Available: %s"
            % (heuristic, ", ".join(sorted(HEURISTICS)))
        ) from error
