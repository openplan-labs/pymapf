"""Safe Interval Path Planning (SIPP).

Space-time A* searches ``(vertex, timestep)`` states, so its state space grows
linearly with the planning horizon: an agent that must wait 200 steps for a
corridor to clear pays for 200 states per vertex. SIPP instead collapses time
into *safe intervals* -- maximal windows during which a vertex is free -- and
searches ``(vertex, interval)`` states. On a map where most vertices are free
for most of the time (which is every real MAPF instance), the number of
intervals per vertex is tiny and independent of the horizon.

This makes SIPP the low-level search of choice for the anytime and
large-neighbourhood methods in this package, where the same agent is re-planned
thousands of times against a dense set of reservations.

The path returned is a plain list of vertices indexed by timestep, identical in
shape to what :func:`~pymapf.algorithms.space_time_astar.space_time_astar`
returns, so the two are interchangeable.

References
----------
* Phillips, M.; and Likhachev, M. 2011. *SIPP: Safe interval path planning for
  dynamic environments.* ICRA 2011: 5628-5635.
* Li, J.; Chen, Z.; Harabor, D.; Stuckey, P. J.; and Koenig, S. 2022.
  *MAPF-LNS2: Fast repairing for multi-agent path finding via large
  neighborhood search.* AAAI 2022: 10256-10265.  (SIPPS, the soft-constraint
  variant this implementation's collision counting follows.)
"""

from __future__ import annotations

import heapq
from itertools import count
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.grid import Cell
from ..core.heuristics import get_heuristic
from ..core.solver import Constraints

__all__ = ["safe_intervals", "sipp"]

Interval = Tuple[int, float]  # [start, end) with end possibly inf


def safe_intervals(
    blocked_times: Sequence[int], horizon: Optional[int] = None
) -> List[Interval]:
    """Maximal windows a vertex is free, given the timesteps it is occupied.

    ``[(0, 3), (5, inf)]`` means "free at t = 0,1,2 and from t = 5 onwards".
    """
    if not blocked_times:
        return [(0, float("inf"))]

    blocked = sorted(set(blocked_times))
    intervals: List[Interval] = []
    cursor = 0
    for t in blocked:
        if t > cursor:
            intervals.append((cursor, t))
        cursor = max(cursor, t + 1)
    end = float("inf") if horizon is None else max(horizon + 1, cursor)
    if cursor <= end:
        intervals.append((cursor, end))
    return intervals


def _intervals_from_constraints(
    constraints: Constraints, cells: Sequence[Cell]
) -> Dict[Cell, List[Interval]]:
    """Turn vertex constraints into per-vertex safe intervals."""
    blocked: Dict[Cell, List[int]] = {}
    for cell, t in constraints.vertex:
        blocked.setdefault(cell, []).append(t)
    return {cell: safe_intervals(blocked.get(cell, [])) for cell in cells}


def sipp(
    grid,
    start: Cell,
    goal: Cell,
    constraints: Optional[Constraints] = None,
    heuristic="manhattan",
    allow_diagonals: bool = False,
    max_timestep: Optional[int] = None,
) -> Optional[List[Cell]]:
    """Plan a single agent through ``constraints`` using safe intervals.

    Returns the timestep-indexed path, or ``None`` if the agent cannot reach
    (and remain at) its goal. Waiting is implicit: moving to a later interval of
    the same vertex *is* the wait action, which is why the state space stays
    small.
    """
    constraints = constraints or Constraints()
    h = get_heuristic(heuristic)
    horizon_bound = max_timestep if max_timestep is not None else _default_horizon(
        grid, constraints
    )

    interval_cache: Dict[Cell, List[Interval]] = {}

    def intervals_of(cell: Cell) -> List[Interval]:
        cached = interval_cache.get(cell)
        if cached is None:
            times = [t for (c, t) in constraints.vertex if c == cell]
            cached = safe_intervals(times)
            interval_cache[cell] = cached
        return cached

    def interval_index(cell: Cell, t: int) -> Optional[int]:
        for index, (lo, hi) in enumerate(intervals_of(cell)):
            if lo <= t < hi:
                return index
        return None

    start_index = interval_index(start, 0)
    if start_index is None:
        return None  # the agent is standing somewhere it is forbidden to be

    # The agent may only settle on its goal once nothing can push it off again.
    settle_time = constraints.last_vertex_time(goal)

    State = Tuple[Cell, int]  # (vertex, interval index)
    # `earliest[state]` is final once the state is closed (the heap is ordered
    # by arrival + h with unit costs), so it doubles as the path's timestamps.
    earliest: Dict[State, int] = {(start, start_index): 0}
    parent: Dict[State, Optional[State]] = {(start, start_index): None}
    tie = count()
    heap = [(h(start, goal), next(tie), 0, start, start_index)]
    closed = set()

    while heap:
        _, _, t, cell, index = heapq.heappop(heap)
        state = (cell, index)
        if state in closed:
            continue
        closed.add(state)

        lo, hi = intervals_of(cell)[index]
        if cell == goal and t > settle_time and hi == float("inf"):
            return _unroll(parent, earliest, state)

        for neighbour in list(grid.neighbors(cell, allow_diagonals)):
            for n_index, (n_lo, n_hi) in enumerate(intervals_of(neighbour)):
                # Earliest arrival at `neighbour` that leaves `cell` inside its
                # own safe interval and lands inside the neighbour's.
                arrival = max(t + 1, n_lo)
                if arrival >= n_hi or arrival > hi:  # cannot wait that long here
                    continue
                if arrival > horizon_bound:
                    continue
                departure = arrival - 1
                if departure >= hi:
                    continue
                if constraints.blocks_edge(cell, neighbour, arrival):
                    # An edge constraint only forbids this exact crossing time;
                    # arriving one step later is still legal if the interval
                    # allows it.
                    arrival += 1
                    if arrival >= n_hi or arrival - 1 >= hi or arrival > horizon_bound:
                        continue
                n_state = (neighbour, n_index)
                if n_state in closed:
                    continue
                if arrival < earliest.get(n_state, float("inf")):
                    earliest[n_state] = arrival
                    parent[n_state] = state
                    heapq.heappush(
                        heap,
                        (arrival + h(neighbour, goal), next(tie), arrival, neighbour, n_index),
                    )
    return None


def _default_horizon(grid, constraints: Constraints) -> int:
    last_constraint = max(
        [t for (_, t) in constraints.vertex] + [t for (_, _, t) in constraints.edge] + [0]
    )
    return grid.free_cells + last_constraint + 1


def _unroll(parent, earliest, state) -> List[Cell]:
    """Rebuild the timestep-indexed path, filling waits with repeated vertices."""
    stops: List[Tuple[Cell, int]] = []
    cursor = state
    while cursor is not None:
        stops.append((cursor[0], earliest[cursor]))
        cursor = parent.get(cursor)
    stops.reverse()

    # `stops` holds (vertex, arrival time); expand the gaps into waits.
    path: List[Cell] = []
    for index, (cell, time) in enumerate(stops):
        if index == 0:
            path.append(cell)
            continue
        previous_cell = stops[index - 1][0]
        while len(path) < time:
            path.append(previous_cell)
        path.append(cell)
    return path
