"""LaCAM: Lazy Constraints Addition search for MAPF.

PIBT is fast but incomplete; CBS is complete but its constraint tree explodes.
LaCAM takes a third route: it searches the space of *configurations* (one
vertex per agent, i.e. a joint state) like an ordinary graph search, but
generates successors **lazily**. Instead of enumerating the exponentially many
successors of a configuration up front, each search node carries a queue of
partial assignments -- "agent 3 must take vertex v" -- and pops one at a time,
handing it to PIBT as a constraint. PIBT fills in every other agent in one
sweep, so each expansion produces exactly one successor at the cost of a single
PIBT call.

The consequence is the interesting part: because the constraint queue
eventually enumerates every combination, the search is **complete**, while in
practice PIBT's first suggestion is almost always good enough that the queue is
barely touched. That is how LaCAM solves instances with thousands of agents in
milliseconds where optimal solvers do not finish at all.

A caveat worth knowing before using the first solution as-is: it is the
*depth-first path* through the configuration graph, so its cost can be far from
optimal -- measured here at up to 234x optimal on a hard 8x8 instance where PIBT
flails. LaCAM is a completeness-and-speed algorithm, not a quality one, and its
output is meant to be refined.

``anytime=True`` spends whatever budget remains after the first solution on
**randomised restarts**, keeping the cheapest plan found. That choice is
empirical: continuing the search in place and relaxing g-values over the
explored graph -- our first attempt at LaCAM*'s scheme -- won 0 of 28 paired
instances, while restarts won 26 of 28 (mean cost 0.81x) and handing the first
solution to :class:`~pymapf.algorithms.lns.LargeNeighborhoodSearch` won 25 of 28
(0.79x). See ``docs/survey.md`` section 7.4. This is *not* LaCAM*'s
eventual-optimality guarantee, which requires the full cost-propagation scheme
of Okumura (2023).

References
----------
* Okumura, K. 2023. *LaCAM: Search-based algorithm for quick multi-agent
  pathfinding.* AAAI 2023, 37(10): 11655-11662.
* Okumura, K. 2023. *Improving LaCAM for scalable eventually optimal
  multi-agent pathfinding.* IJCAI 2023: 243-251.  (LaCAM*)
* Okumura, K. 2024. *Engineering LaCAM*: Towards real-time, large-scale, and
  near-optimal multi-agent pathfinding.* AAMAS 2024: 1501-1509.  (LaCAM3:
  swap operations, monte-carlo configuration generation, refinement)
* Okumura, K.; Machida, M.; Defago, X.; and Tamura, Y. 2022. *Priority
  inheritance with backtracking for iterative multi-agent path finding.*
  Artificial Intelligence 310: 103752.  (the configuration generator)
"""

from __future__ import annotations

import random
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

from ..core.grid import Cell
from ..core.solver import (
    MAPFProblem,
    MAPFSolver,
    Observer,
    Solution,
    register_solver,
)
from ..core.trace import _Emitter
from .pibt import _DistanceOracle, pibt_step

__all__ = ["LaCAM"]

Configuration = Tuple[Cell, ...]


class _Constraint:
    """A node of the low-level constraint tree: agents 0..depth-1 are pinned."""

    __slots__ = ("who", "where", "depth")

    def __init__(self, who: Tuple[int, ...], where: Tuple[Cell, ...], depth: int):
        self.who = who
        self.where = where
        self.depth = depth

    def as_forced(self, names: List[str]) -> Dict[str, Cell]:
        return {names[i]: cell for i, cell in zip(self.who, self.where)}


class _Node:
    """A search node: one configuration plus its lazily-expanded constraints."""

    __slots__ = ("config", "parent", "g", "h", "order", "tree", "neighbours", "priorities")

    def __init__(
        self,
        config: Configuration,
        parent,
        g: float,
        h: float,
        order: List[int],
        priorities: Dict[str, float],
    ):
        self.config = config
        self.parent = parent
        self.g = g
        self.h = h
        self.order = order
        # PIBT's priorities are *dynamic*: they accumulate while an agent is
        # away from its goal. Recomputing them from position alone (as a first
        # implementation is tempted to) throws away the deadlock-breaking that
        # makes PIBT work, and the search degenerates into a random walk.
        self.priorities = priorities
        # The constraint tree starts with the empty constraint.
        self.tree = deque([_Constraint((), (), 0)])
        self.neighbours: List["_Node"] = []


@register_solver("lacam")
class LaCAM(MAPFSolver):
    """Complete, very fast configuration-space search (Okumura 2023).

    Args:
        anytime: after the first solution, restart the (randomised) search with
            fresh seeds until the budget is spent and keep the cheapest plan.
            The first solution arrives just as fast either way. For the best
            quality per second, prefer ``pymapf.solve(problem, "lns",
            initial="lacam")``.
        time_limit: wall-clock budget in seconds. LaCAM is meant to be given a
            deadline -- it returns the best solution found within it.
        max_expansions: cap on high-level nodes.
        seed: fixes PIBT's tie-breaking so runs are reproducible.
    """

    def __init__(
        self,
        anytime: bool = False,
        time_limit: Optional[float] = 10.0,
        max_expansions: int = 200000,
        seed: Optional[int] = 0,
    ):
        self.anytime = anytime
        self.time_limit = time_limit
        self.max_expansions = max_expansions
        self.seed = seed

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _transition_cost(a: Configuration, b: Configuration, goals: Configuration) -> int:
        """Sum-of-costs increment: an agent parked on its goal is free."""
        cost = 0
        for u, v, g in zip(a, b, goals):
            if not (u == g and v == g):
                cost += 1
        return cost

    def solve(
        self, problem: MAPFProblem, observer: Optional[Observer] = None
    ) -> Optional[Solution]:
        if not self.anytime:
            return self._search(problem, observer, seed=self.seed, budget=self.time_limit)

        # Anytime: repeat the randomised search until the budget is spent and
        # keep the cheapest plan. Restarts beat continuing in place here; see
        # the module docstring for the measurement.
        started = time.perf_counter()
        best: Optional[Solution] = None
        attempt = 0
        while attempt < 1000:
            elapsed = time.perf_counter() - started
            remaining = (
                None if self.time_limit is None else self.time_limit - elapsed
            )
            if remaining is not None and remaining <= 0:
                break
            per_run = (
                remaining
                if remaining is None
                else min(remaining, max(0.05, (self.time_limit or 1.0) / 8))
            )
            candidate = self._search(
                problem,
                observer if attempt == 0 else None,
                seed=(self.seed or 0) + attempt,
                budget=per_run,
            )
            attempt += 1
            if candidate is not None and (
                best is None or candidate.sum_of_costs < best.sum_of_costs
            ):
                best = candidate
            if self.time_limit is None:
                break
        if best is None:
            return None
        return Solution(
            paths=best.paths,
            algorithm=self.name,
            expansions=attempt,
            runtime=time.perf_counter() - started,
        )

    def _search(
        self,
        problem: MAPFProblem,
        observer: Optional[Observer] = None,
        seed: Optional[int] = 0,
        budget: Optional[float] = None,
    ) -> Optional[Solution]:
        emit = _Emitter(observer)
        started = time.perf_counter()
        rng = random.Random(seed)
        time_limit = budget

        agents = list(problem.agents)
        names = [a.name for a in agents]
        goals_by_name = {a.name: a.goal for a in agents}
        goal_config: Configuration = tuple(a.goal for a in agents)
        start_config: Configuration = tuple(a.start for a in agents)

        distance = _DistanceOracle(problem.grid, goals_by_name, problem.allow_diagonals)
        for index, agent in enumerate(agents):
            if not distance.reachable(agent.name, agent.start):
                emit("failed", reason="agent %r cannot reach its goal" % agent.name)
                return None

        def heuristic(config: Configuration) -> float:
            return sum(distance(names[i], cell) for i, cell in enumerate(config))

        emit("root", agents=list(names), cost=int(heuristic(start_config)))

        base_priority = {
            name: index / (len(names) + 1) for index, name in enumerate(names)
        }
        root = _Node(
            start_config,
            None,
            0.0,
            heuristic(start_config),
            list(range(len(agents))),
            dict(base_priority),
        )
        explored: Dict[Configuration, _Node] = {start_config: root}
        # OPEN is a stack: depth-first is what makes LaCAM reach a goal fast.
        open_stack: List[_Node] = [root]

        best: Optional[List[Configuration]] = None
        best_cost = float("inf")
        expansions = 0

        while open_stack:
            if expansions >= self.max_expansions:
                break
            if time_limit is not None and time.perf_counter() - started > time_limit:
                break

            node = open_stack[-1]

            if node.config == goal_config:
                best = _backtrack(node)
                best_cost = node.g
                emit(
                    "solved",
                    cost=int(best_cost),
                    makespan=len(best) - 1,
                    expansions=expansions,
                    paths=_as_paths(best, names) if emit else None,
                )
                break

            if not node.tree:
                open_stack.pop()
                continue

            constraint = node.tree.popleft()

            # Grow the constraint tree one agent deeper, in the node's order.
            if constraint.depth < len(agents):
                index = node.order[constraint.depth]
                current = node.config[index]
                options = list(problem.grid.neighbors(current, problem.allow_diagonals))
                options.append(current)
                for cell in options:
                    node.tree.append(
                        _Constraint(
                            constraint.who + (index,),
                            constraint.where + (cell,),
                            constraint.depth + 1,
                        )
                    )

            positions = {names[i]: cell for i, cell in enumerate(node.config)}
            nxt = pibt_step(
                problem.grid,
                names,
                positions,
                goals_by_name,
                node.priorities,
                distance,
                allow_diagonals=problem.allow_diagonals,
                forced=constraint.as_forced(names),
                rng=rng,
            )
            if nxt is None:
                continue

            child_config: Configuration = tuple(nxt[name] for name in names)
            step_cost = self._transition_cost(node.config, child_config, goal_config)
            expansions += 1

            child_priorities = {
                name: (
                    base_priority[name]
                    if nxt[name] == goals_by_name[name]
                    else node.priorities[name] + 1
                )
                for name in names
            }

            known = explored.get(child_config)
            if known is None:
                child = _Node(
                    child_config,
                    node,
                    node.g + step_cost,
                    heuristic(child_config),
                    _priority_order(child_config, goal_config, node.order),
                    child_priorities,
                )
                explored[child_config] = child
                node.neighbours.append(child)
                open_stack.append(child)
                emit(
                    "expand",
                    node=expansions,
                    cost=int(child.g + child.h),
                    open=len(open_stack),
                    paths=_as_paths(_backtrack(child), names) if emit else None,
                )
            else:
                node.neighbours.append(known)

        runtime = time.perf_counter() - started
        if best is None:
            emit(
                "failed",
                reason=(
                    "time limit (%.3gs) reached" % time_limit
                    if time_limit is not None and runtime > time_limit
                    else "configuration space exhausted after %d expansions" % expansions
                ),
            )
            return None

        paths = _as_paths(best, names)
        goals_map = {name: goals_by_name[name] for name in names}
        paths = {name: _trim(path, goals_map[name]) for name, path in paths.items()}
        return Solution(
            paths=paths,
            algorithm=self.name,
            expansions=expansions,
            runtime=runtime,
        )


def _priority_order(config: Configuration, goals: Configuration, previous: List[int]) -> List[int]:
    """Agents away from their goal come first, keeping the previous order stable.

    This mirrors LaCAM's insight that the *order* in which the constraint tree
    pins agents matters far more than which vertices it tries: agents that have
    already arrived rarely need to be constrained.
    """
    away = [i for i in previous if config[i] != goals[i]]
    home = [i for i in previous if config[i] == goals[i]]
    return away + home


def _backtrack(node: _Node) -> List[Configuration]:
    configs = []
    cursor: Optional[_Node] = node
    seen = set()
    while cursor is not None and id(cursor) not in seen:
        seen.add(id(cursor))
        configs.append(cursor.config)
        cursor = cursor.parent
    configs.reverse()
    return configs


def _as_paths(configs: List[Configuration], names: List[str]) -> Dict[str, List[Cell]]:
    return {
        name: [config[index] for config in configs] for index, name in enumerate(names)
    }


def _trim(path: List[Cell], goal: Cell) -> List[Cell]:
    end = len(path) - 1
    while end > 0 and path[end] == goal and path[end - 1] == goal:
        end -= 1
    return path[: end + 1]
