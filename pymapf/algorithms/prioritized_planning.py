"""Prioritized Planning (a.k.a. cooperative A*).

Agents are planned one at a time in priority order; each agent treats the
already-planned agents as moving obstacles (space-time reservations). This is
fast and simple but *incomplete*: a bad priority order can fail on solvable
instances. It is the clean, framework-native replacement for the legacy
``CooperativeAStar``.

References
----------
* Erdmann, M.; and Lozano-Perez, T. 1987. *On multiple moving objects.*
  Algorithmica 2: 477-521.  (prioritized planning)
* Silver, D. 2005. *Cooperative pathfinding.* AIIDE 2005: 117-122.
  (cooperative A* and the space-time reservation table)
* Ma, H.; Harabor, D.; Stuckey, P. J.; Li, J.; and Koenig, S. 2019.
  *Searching with consistent prioritization for multi-agent path finding.* AAAI
  2019: 7643-7650.  (priority orderings and their effect on completeness)
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from ..core.grid import Cell
from ..core.solver import (
    Constraints,
    MAPFProblem,
    MAPFSolver,
    Observer,
    Solution,
    register_solver,
)
from ..core.trace import _Emitter
from .space_time_astar import space_time_astar


@register_solver("prioritized")
class PrioritizedPlanning(MAPFSolver):
    """Plan agents sequentially, reserving space-time cells as we go.

    Args:
        heuristic: name or callable used by the low-level search.
        priority: optional list of agent names giving the planning order.
            Defaults to the order agents appear in the problem.
    """

    def __init__(self, heuristic="manhattan", priority: Optional[List[str]] = None):
        self.heuristic = heuristic
        self.priority = priority

    def solve(
        self, problem: MAPFProblem, observer: Optional[Observer] = None
    ) -> Optional[Solution]:
        emit = _Emitter(observer)
        started = time.perf_counter()
        agents = {a.name: a for a in problem.agents}
        order = self.priority or [a.name for a in problem.agents]
        if set(order) != set(agents):
            raise ValueError("priority must list exactly the problem's agent names")

        emit("root", agents=list(order), cost=0)
        reservations: Dict[str, List[Cell]] = {}
        expansions = 0
        for name in order:
            agent = agents[name]
            # ponytail: horizon grows with the reservations so a low-priority
            # agent always has room to wait for everyone ahead of it to pass.
            # Ceiling: worst case is O(sum of prior path lengths); fine for the
            # grid sizes this library targets.
            horizon = (
                problem.grid.free_cells
                + sum(len(p) for p in reservations.values())
                + len(order)
                + 1
            )
            constraints = _reservations_to_constraints(reservations, horizon)
            path = space_time_astar(
                problem.grid,
                agent.start,
                agent.goal,
                constraints=constraints,
                heuristic=self.heuristic,
                allow_diagonals=problem.allow_diagonals,
                max_timestep=horizon,
            )
            if path is None:
                emit(
                    "failed",
                    reason="agent %r has no path under the reservations of the "
                    "%d higher-priority agents" % (name, len(reservations)),
                )
                return None
            reservations[name] = path
            expansions += 1
            emit("agent_planned", agent=name, path=list(path), cost=len(path) - 1)
            emit(
                "expand",
                node=expansions,
                cost=sum(len(p) - 1 for p in reservations.values()),
                open=len(order) - expansions,
            )

        solution = Solution(
            paths=reservations,
            algorithm=self.name,
            expansions=expansions,
            runtime=time.perf_counter() - started,
        )
        emit(
            "solved",
            cost=solution.sum_of_costs,
            makespan=solution.makespan,
            expansions=expansions,
            paths={n: list(p) for n, p in solution.paths.items()},
        )
        return solution


def _reservations_to_constraints(
    reservations: Dict[str, List[Cell]], horizon: int
) -> Constraints:
    """Turn already-planned paths into constraints for the next agent."""
    c = Constraints()
    for path in reservations.values():
        for t, cell in enumerate(path):
            c.add_vertex(cell, t)
        # A finished agent parks on its goal, so block it for the whole horizon.
        for t in range(len(path), horizon + 1):
            c.add_vertex(path[-1], t)
        # Forbid swapping with this agent's moves.
        for t in range(len(path) - 1):
            c.add_edge(path[t + 1], path[t], t + 1)
    return c
