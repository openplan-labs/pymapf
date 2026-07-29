"""Single-agent graph search: the primitives every MAPF solver stands on.

These operate on anything with the map interface -- a
:class:`~pymapf.core.grid.GridMap` or a
:class:`~pymapf.core.graph.ExplicitGraph` -- and none of them know about time
or other agents. The space-time versions live in
:mod:`pymapf.algorithms.space_time_astar` and :mod:`pymapf.algorithms.sipp`.

References
----------
* Dijkstra, E. W. 1959. *A note on two problems in connexion with graphs.*
  Numerische Mathematik 1(1): 269-271.
* Hart, P. E.; Nilsson, N. J.; and Raphael, B. 1968. *A formal basis for the
  heuristic determination of minimum cost paths.* IEEE Transactions on Systems
  Science and Cybernetics 4(2): 100-107.  (A*)
* Pohl, I. 1970. *Heuristic search viewed as path finding in a graph.*
  Artificial Intelligence 1(3-4): 193-204.  (weighted A*)
* Pearl, J.; and Kim, J. H. 1982. *Studies in semi-admissible heuristics.*
  IEEE Transactions on Pattern Analysis and Machine Intelligence 4(4): 392-399.
  (focal search / A*-epsilon)
* Harabor, D.; and Grastien, A. 2011. *Online graph pruning for pathfinding on
  grid maps.* AAAI 2011: 1114-1119.  (jump point search)
"""

from __future__ import annotations

import heapq
from itertools import count
from typing import Callable, Dict, List, Optional, Tuple

from ..core.heuristics import get_heuristic

__all__ = [
    "dijkstra",
    "astar",
    "weighted_astar",
    "focal_astar",
    "distance_table",
    "reconstruct",
]


def reconstruct(parents: Dict, node) -> List:
    """Walk ``parents`` back from ``node`` to the root, returning a path."""
    path = [node]
    while parents.get(node) is not None:
        node = parents[node]
        path.append(node)
    path.reverse()
    return path


def dijkstra(
    graph,
    start,
    goal=None,
    allow_diagonals: bool = False,
) -> Tuple[Dict, Dict]:
    """Uniform-cost search from ``start`` (Dijkstra 1959).

    Returns ``(distance, parent)`` over every node reachable from ``start``, or
    stops early once ``goal`` is settled. Unit edge costs, matching the MAPF
    convention that one move takes one timestep.
    """
    distance: Dict = {start: 0}
    parent: Dict = {start: None}
    heap = [(0, next(_TIE), start)]
    settled = set()

    while heap:
        d, _, node = heapq.heappop(heap)
        if node in settled:
            continue
        settled.add(node)
        if goal is not None and node == goal:
            break
        for neighbour in graph.neighbors(node, allow_diagonals):
            nd = d + 1
            if nd < distance.get(neighbour, float("inf")):
                distance[neighbour] = nd
                parent[neighbour] = node
                heapq.heappush(heap, (nd, next(_TIE), neighbour))
    return distance, parent


_TIE = count()  # stable tie-breaker shared by the heaps in this module


def astar(
    graph,
    start,
    goal,
    heuristic="manhattan",
    allow_diagonals: bool = False,
) -> Optional[List]:
    """A* (Hart, Nilsson and Raphael 1968). Optimal for an admissible heuristic."""
    return weighted_astar(
        graph,
        start,
        goal,
        heuristic=heuristic,
        weight=1.0,
        allow_diagonals=allow_diagonals,
    )


def weighted_astar(
    graph,
    start,
    goal,
    heuristic="manhattan",
    weight: float = 1.0,
    allow_diagonals: bool = False,
) -> Optional[List]:
    """Weighted A* (Pohl 1970): expand on ``g + w*h``.

    The returned path costs at most ``weight`` times the optimum -- the same
    bounded-suboptimality trade the multi-agent solvers make, one level down.
    """
    if weight < 1.0:
        raise ValueError("weight must be >= 1.0")
    h = get_heuristic(heuristic)

    g: Dict = {start: 0}
    parent: Dict = {start: None}
    heap = [(weight * h(start, goal), next(_TIE), start)]
    closed = set()

    while heap:
        _, _, node = heapq.heappop(heap)
        if node in closed:
            continue
        closed.add(node)
        if node == goal:
            return reconstruct(parent, node)
        for neighbour in graph.neighbors(node, allow_diagonals):
            ng = g[node] + 1
            if ng < g.get(neighbour, float("inf")):
                g[neighbour] = ng
                parent[neighbour] = node
                heapq.heappush(
                    heap, (ng + weight * h(neighbour, goal), next(_TIE), neighbour)
                )
    return None


def focal_astar(
    graph,
    start,
    goal,
    heuristic="manhattan",
    weight: float = 1.5,
    tie_breaker: Optional[Callable] = None,
    allow_diagonals: bool = False,
) -> Optional[List]:
    """Focal search / A*-epsilon (Pearl and Kim 1982).

    OPEN keeps the admissible lower bound; FOCAL holds the nodes within
    ``weight`` times that bound and is ordered by ``tie_breaker`` (default:
    fewest expansions to the goal, i.e. plain ``h``). The path is still within
    ``weight`` of optimal -- this is the single-agent shape of what ECBS and
    :class:`~pymapf.algorithms.weighted_cbs.WeightedCBS` do on the high level.
    """
    if weight < 1.0:
        raise ValueError("weight must be >= 1.0")
    h = get_heuristic(heuristic)
    secondary = tie_breaker or (lambda node, g_value: h(node, goal))

    g: Dict = {start: 0}
    parent: Dict = {start: None}
    open_heap = [(h(start, goal), next(_TIE), start)]
    focal = [(secondary(start, 0), h(start, goal), next(_TIE), start)]
    closed = set()

    while open_heap:
        while open_heap and open_heap[0][2] in closed:
            heapq.heappop(open_heap)
        if not open_heap:
            break
        bound = weight * open_heap[0][0]

        node = None
        while focal:
            candidate = heapq.heappop(focal)
            if candidate[3] in closed:
                continue
            if candidate[1] > bound:  # no longer eligible; put it back for later
                heapq.heappush(open_heap, (candidate[1], candidate[2], candidate[3]))
                continue
            node = candidate[3]
            break
        if node is None:
            node = heapq.heappop(open_heap)[2]

        if node in closed:
            continue
        closed.add(node)
        if node == goal:
            return reconstruct(parent, node)

        for neighbour in graph.neighbors(node, allow_diagonals):
            ng = g[node] + 1
            if ng < g.get(neighbour, float("inf")):
                g[neighbour] = ng
                parent[neighbour] = node
                f = ng + h(neighbour, goal)
                heapq.heappush(open_heap, (f, next(_TIE), neighbour))
                if f <= bound:
                    heapq.heappush(
                        focal, (secondary(neighbour, ng), f, next(_TIE), neighbour)
                    )
    return None


def distance_table(graph, goal, allow_diagonals: bool = False) -> Dict:
    """Exact distance from every node to ``goal`` (a backward Dijkstra).

    This is the *true distance heuristic* every serious MAPF implementation
    uses: it is exact, so the low-level search expands no state that cannot be
    on a shortest path, and on a maze it is worth orders of magnitude over
    Manhattan distance. It is also the only admissible heuristic available on a
    graph with no geometry.
    """
    distance, _ = dijkstra(graph, goal, allow_diagonals=allow_diagonals)
    return distance
