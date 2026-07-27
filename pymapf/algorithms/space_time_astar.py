"""Constraint-aware space-time A* (the low-level search for MAPF solvers).

This is the shared single-agent primitive used by both Prioritized Planning and
Conflict-Based Search. It searches over ``(cell, time)`` states, honouring
vertex/edge :class:`~pymapf.core.solver.Constraints`, allowing *wait* actions,
and terminating on a provable upper bound so it can never loop forever (a
concrete failure mode of the legacy ``AStar``).

References
----------
* Hart, P. E.; Nilsson, N. J.; and Raphael, B. 1968. *A formal basis for the
  heuristic determination of minimum cost paths.* IEEE Transactions on Systems
  Science and Cybernetics 4(2): 100-107.
* Silver, D. 2005. *Cooperative pathfinding.* AIIDE 2005: 117-122.  (searching
  over (cell, time) states against a reservation table)
* Phillips, M.; and Likhachev, M. 2011. *SIPP: Safe interval path planning for
  dynamic environments.* ICRA 2011: 5628-5635.  (the interval-based alternative
  in :mod:`pymapf.algorithms.sipp`)
"""

from __future__ import annotations

import heapq
from itertools import count
from typing import List, Optional

from ..core.grid import Cell, GridMap
from ..core.heuristics import get_heuristic
from ..core.solver import Constraints


def space_time_astar(
    grid: GridMap,
    start: Cell,
    goal: Cell,
    constraints: Optional[Constraints] = None,
    heuristic="manhattan",
    allow_diagonals: bool = False,
    max_timestep: Optional[int] = None,
) -> Optional[List[Cell]]:
    """Find a minimal-time path from ``start`` to ``goal`` respecting constraints.

    Returns a list of cells (index ``t`` is the position at time ``t``) ending
    on ``goal``, or ``None`` if no such path exists within the time bound.
    """
    constraints = constraints or Constraints()
    h = get_heuristic(heuristic)

    # An agent must be able to *stay* on its goal once it arrives, so it may
    # only settle after the last vertex constraint that touches the goal.
    settle_time = constraints.last_vertex_time(goal)

    # Upper bound on time: no shortest constrained path needs to be longer than
    # the number of free cells plus the last constraint time. Beyond that the
    # instance is unsolvable for this agent.
    last_constraint_t = max(
        [t for (_, t) in constraints.vertex]
        + [t for (_, _, t) in constraints.edge]
        + [0]
    )
    if max_timestep is None:
        max_timestep = grid.free_cells + last_constraint_t + 1

    counter = count()  # stable tie-breaker for the heap
    start_state = (start, 0)
    open_heap = [(h(start, goal), next(counter), start, 0)]
    # Every (cell, t) state has a fixed cost g == t, so the first time we pop it
    # is optimal; a visited set is enough to avoid re-expansion.
    visited = set()
    parent = {start_state: None}

    while open_heap:
        _, _, cell, t = heapq.heappop(open_heap)
        state = (cell, t)
        if state in visited:
            continue
        visited.add(state)

        if cell == goal and t > settle_time:
            return _reconstruct(parent, state)

        if t >= max_timestep:
            continue

        # Candidate moves: every free spatial neighbor plus a wait in place.
        for ncell in grid.neighbors(cell, allow_diagonals) + [cell]:
            nt = t + 1
            nstate = (ncell, nt)
            if nstate in visited:
                continue
            if constraints.blocks_vertex(ncell, nt):
                continue
            if constraints.blocks_edge(cell, ncell, nt):
                continue
            if nstate not in parent:
                parent[nstate] = state
            heapq.heappush(open_heap, (nt + h(ncell, goal), next(counter), ncell, nt))

    return None


def _reconstruct(parent, state) -> List[Cell]:
    path = []
    while state is not None:
        path.append(state[0])
        state = parent[state]
    path.reverse()
    return path
