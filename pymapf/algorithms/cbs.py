"""Conflict-Based Search (CBS).

CBS is the canonical two-level optimal MAPF algorithm (Sharon et al., 2015):

* the **high level** searches a binary constraint tree, best-first on
  sum-of-costs; each node holds a set of constraints per agent;
* the **low level** replans a single agent under its constraints using the
  shared :func:`~pymapf.algorithms.space_time_astar.space_time_astar`.

On each expansion we find the first conflict in the current joint plan and
branch by forbidding one of the two involved agents from that (vertex or edge)
constraint, then replan only that agent. The first conflict-free node is
returned and is optimal in sum-of-costs.
"""

from __future__ import annotations

import heapq
import time
from itertools import count
from typing import Dict, List, Optional

from ..core.solver import (
    Conflict,
    Constraints,
    MAPFProblem,
    MAPFSolver,
    Observer,
    Solution,
    count_conflicts,
    find_first_conflict,
    register_solver,
)
from ..core.trace import _Emitter
from .space_time_astar import space_time_astar


class _Node:
    """A constraint-tree node: constraints, the plan they induce, and its cost.

    ``conflicts`` is the tie-breaker. Among nodes of equal sum-of-costs the one
    closest to being conflict-free is the one worth expanding, and preferring it
    is what keeps CBS from breadth-first-ing through a plateau of equal-cost
    nodes on corridor-heavy maps.
    """

    __slots__ = ("constraints", "paths", "cost", "conflicts", "_expanded")

    def __init__(self, constraints, paths):
        self.constraints: Dict[str, Constraints] = constraints
        self.paths: Dict[str, List] = paths
        self.cost = sum(len(p) - 1 for p in paths.values())
        self.conflicts = count_conflicts(paths)


@register_solver("cbs")
class ConflictBasedSearch(MAPFSolver):
    """Optimal (sum-of-costs) conflict-based search.

    Args:
        heuristic: name or callable for the low-level search.
        max_expansions: safety cap on high-level nodes; returns ``None`` if
            exceeded (guards against pathological/unsolvable instances).
        time_limit: optional wall-clock budget in seconds. CBS is exponential in
            the number of conflicts, so on a hard instance "it will finish
            eventually" is not a useful promise -- a budget turns an unbounded
            wait into a reported failure. ``None`` means no limit.
    """

    def __init__(
        self,
        heuristic="manhattan",
        max_expansions: int = 10000,
        time_limit: Optional[float] = None,
    ):
        self.heuristic = heuristic
        self.max_expansions = max_expansions
        self.time_limit = time_limit

    def solve(
        self, problem: MAPFProblem, observer: Optional[Observer] = None
    ) -> Optional[Solution]:
        emit = _Emitter(observer)
        started = time.perf_counter()
        agents = list(problem.agents)

        # Root: plan every agent independently, with no constraints.
        root_constraints = {a.name: Constraints() for a in agents}
        root_paths = {}
        for a in agents:
            path = self._low_level(problem, a.start, a.goal, root_constraints[a.name])
            if path is None:
                emit("failed", reason="agent %r has no individual path" % a.name)
                return None
            root_paths[a.name] = path
            emit(
                "agent_planned",
                agent=a.name,
                path=list(path),
                cost=len(path) - 1,
            )

        root = _Node(root_constraints, root_paths)
        emit("root", agents=[a.name for a in agents], cost=root.cost)

        counter = count()
        open_heap = [(root.cost, root.conflicts, next(counter), root)]
        expansions = 0

        timed_out = False
        while open_heap and expansions < self.max_expansions:
            if (
                self.time_limit is not None
                and time.perf_counter() - started > self.time_limit
            ):
                timed_out = True
                break
            _, _, _, node = heapq.heappop(open_heap)
            expansions += 1
            # The node's paths ride along so an observer can *show* the plan
            # CBS is currently considering; copying is only paid when observed.
            emit(
                "expand",
                node=expansions,
                cost=node.cost,
                open=len(open_heap),
                paths={n: list(p) for n, p in node.paths.items()} if emit else None,
            )

            conflict = find_first_conflict(node.paths)
            if conflict is None:
                solution = Solution(
                    paths=node.paths,
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

            emit(
                "conflict",
                type=conflict.kind,
                a=conflict.a,
                b=conflict.b,
                t=conflict.t,
                cell=conflict.cell_a,
            )

            for agent_name in self._agents_to_constrain(conflict):
                child = self._branch(problem, node, conflict, agent_name)
                if child is not None:
                    emit(
                        "branch",
                        agent=agent_name,
                        constraint=conflict.kind,
                        cost=child.cost,
                    )
                    heapq.heappush(
                        open_heap,
                        (child.cost, child.conflicts, next(counter), child),
                    )

        if timed_out:
            reason = "time limit (%.3gs) reached after %d nodes" % (
                self.time_limit,
                expansions,
            )
        elif expansions >= self.max_expansions:
            reason = "expansion limit (%d) reached" % self.max_expansions
        else:
            reason = "constraint tree exhausted"
        emit("failed", reason=reason)
        return None

    @staticmethod
    def _agents_to_constrain(conflict: Conflict):
        return (conflict.a, conflict.b)

    def _branch(
        self, problem: MAPFProblem, node: _Node, conflict: Conflict, agent_name: str
    ) -> Optional[_Node]:
        """Create the child node that forbids ``agent_name`` from the conflict."""
        constraints = {n: c.copy() for n, c in node.constraints.items()}
        ac = constraints[agent_name]
        if conflict.kind == "vertex":
            ac.add_vertex(conflict.cell_a, conflict.t)
        else:
            # Edge conflict: agent `a` traversed cell_b -> cell_a and `b`
            # traversed cell_a -> cell_b, both arriving at conflict.t.
            if agent_name == conflict.a:
                ac.add_edge(conflict.cell_b, conflict.cell_a, conflict.t)
            else:
                ac.add_edge(conflict.cell_a, conflict.cell_b, conflict.t)

        agent = next(a for a in problem.agents if a.name == agent_name)
        path = self._low_level(problem, agent.start, agent.goal, ac)
        if path is None:
            return None
        paths = dict(node.paths)
        paths[agent_name] = path
        return _Node(constraints, paths)

    def _low_level(self, problem, start, goal, constraints):
        return space_time_astar(
            problem.grid,
            start,
            goal,
            constraints=constraints,
            heuristic=self.heuristic,
            allow_diagonals=problem.allow_diagonals,
        )
