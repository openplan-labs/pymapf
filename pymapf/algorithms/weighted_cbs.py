"""Bounded-suboptimal CBS (focal search), a.k.a. the ECBS high level.

Vanilla CBS is optimal, and on corridor-heavy maps that optimality is paid for
in an exponential constraint tree: every symmetric way of yielding in a corridor
is its own subtree of equal cost, and CBS must sift through all of them.

Weighted CBS keeps a *focal list* -- every open node whose cost is within a
factor ``w`` of the best cost currently in OPEN -- and expands the node from it
with the fewest conflicts. That is a real guarantee, not a heuristic hope: the
returned solution costs at most ``w`` times the optimum, while conflict-greedy
selection typically drives the search straight at a valid plan.

``w = 1.0`` degenerates to CBS with a conflict tie-break; ``w = 1.5`` solves
instances in milliseconds that plain CBS cannot finish in minutes.
"""

from __future__ import annotations

import heapq
import time
from itertools import count
from typing import Optional

from ..core.solver import (
    Constraints,
    MAPFProblem,
    Observer,
    Solution,
    find_first_conflict,
    register_solver,
)
from ..core.trace import _Emitter
from .cbs import ConflictBasedSearch, _Node


@register_solver("wcbs")
class WeightedCBS(ConflictBasedSearch):
    """Bounded-suboptimal conflict-based search.

    Args:
        weight: suboptimality factor ``w >= 1``. The returned solution's
            sum-of-costs is guaranteed ``<= w * optimal``.
        heuristic: name or callable for the low-level search.
        max_expansions: safety cap on high-level nodes.
        time_limit: optional wall-clock budget in seconds.
    """

    def __init__(
        self,
        weight: float = 1.5,
        heuristic="manhattan",
        max_expansions: int = 10000,
        time_limit: Optional[float] = None,
    ):
        if weight < 1.0:
            raise ValueError("weight must be >= 1.0 (1.0 == optimal CBS)")
        super().__init__(
            heuristic=heuristic, max_expansions=max_expansions, time_limit=time_limit
        )
        self.weight = weight

    def solve(
        self, problem: MAPFProblem, observer: Optional[Observer] = None
    ) -> Optional[Solution]:
        emit = _Emitter(observer)
        started = time.perf_counter()
        agents = list(problem.agents)

        root_constraints = {a.name: Constraints() for a in agents}
        root_paths = {}
        for a in agents:
            path = self._low_level(problem, a.start, a.goal, root_constraints[a.name])
            if path is None:
                emit("failed", reason="agent %r has no individual path" % a.name)
                return None
            root_paths[a.name] = path
            emit("agent_planned", agent=a.name, path=list(path), cost=len(path) - 1)

        root = _Node(root_constraints, root_paths)
        emit("root", agents=[a.name for a in agents], cost=root.cost)

        counter = count()
        # OPEN holds every unexpanded node ordered by cost: its minimum is the
        # search's lower bound. PENDING is the same nodes waiting to become
        # eligible, and FOCAL holds the eligible ones ordered by conflicts.
        # Expanded nodes are left in place and skipped lazily.
        tag = next(counter)
        open_heap = [(root.cost, tag, root)]
        pending = [(root.cost, tag, root)]
        focal: list = []
        expansions = 0
        timed_out = False

        while open_heap and expansions < self.max_expansions:
            if (
                self.time_limit is not None
                and time.perf_counter() - started > self.time_limit
            ):
                timed_out = True
                break

            while open_heap and getattr(open_heap[0][2], "_expanded", False):
                heapq.heappop(open_heap)
            if not open_heap:
                break

            bound = self.weight * open_heap[0][0]
            # The bound never shrinks, so a node admitted to FOCAL stays valid.
            while pending and pending[0][0] <= bound:
                cost, node_tag, candidate = heapq.heappop(pending)
                heapq.heappush(focal, (candidate.conflicts, cost, node_tag, candidate))

            node = None
            while focal:
                _, _, _, candidate = heapq.heappop(focal)
                if getattr(candidate, "_expanded", False):
                    continue
                node = candidate
                break
            if node is None:
                _, _, node = heapq.heappop(open_heap)
            node._expanded = True

            expansions += 1
            emit(
                "expand",
                node=expansions,
                cost=node.cost,
                open=len(open_heap),
                bound=bound,
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
                    bound=bound,
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
                if child is None:
                    continue
                emit(
                    "branch",
                    agent=agent_name,
                    constraint=conflict.kind,
                    cost=child.cost,
                )
                child_tag = next(counter)
                heapq.heappush(open_heap, (child.cost, child_tag, child))
                if child.cost <= bound:
                    heapq.heappush(
                        focal, (child.conflicts, child.cost, child_tag, child)
                    )
                else:
                    heapq.heappush(pending, (child.cost, child_tag, child))

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
