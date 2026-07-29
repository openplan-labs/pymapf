"""Anytime MAPF by Large Neighborhood Search (MAPF-LNS).

Every solver here produces *a* solution; LNS makes it better. It starts from an
initial plan -- from PIBT, prioritized planning, LaCAM, anything -- then loops:

1. **destroy**: pick a small neighbourhood of agents (a random set, the agents
   around a congested vertex, or a connected component of interacting agents);
2. **repair**: delete their paths and re-plan them one at a time against the
   *frozen* paths of everyone else, using :mod:`~pymapf.algorithms.sipp`;
3. **accept** the new plan only if the sum of costs did not get worse.

The neighbourhood is small (5-15 agents), so each iteration is milliseconds
even with a thousand agents, and the accepted-only rule makes the cost curve
monotone: stop whenever you like and you keep the best plan so far. Which
destroy operator to use is chosen adaptively -- operators that have been paying
off recently get picked more often (roulette weights, as in the original).

References
----------
* Li, J.; Chen, Z.; Harabor, D.; Stuckey, P. J.; and Koenig, S. 2021. *Anytime
  multi-agent path finding via large neighborhood search.* IJCAI 2021:
  4127-4135.
* Li, J.; Chen, Z.; Harabor, D.; Stuckey, P. J.; and Koenig, S. 2022.
  *MAPF-LNS2: Fast repairing for multi-agent path finding via large
  neighborhood search.* AAAI 2022: 10256-10265.
* Shaw, P. 1998. *Using constraint programming and local search methods to
  solve vehicle routing problems.* CP 1998: 417-431.  (large neighbourhood
  search itself)
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, Sequence

from ..core.grid import Cell
from ..core.solver import (
    Constraints,
    MAPFProblem,
    MAPFSolver,
    Observer,
    Solution,
    find_first_conflict,
    get_solver,
    register_solver,
)
from ..core.trace import _Emitter
from .sipp import sipp

__all__ = ["LargeNeighborhoodSearch"]

DESTROY_OPERATORS = ("random", "congestion", "interaction")


@register_solver("lns")
class LargeNeighborhoodSearch(MAPFSolver):
    """Anytime improvement on top of a fast initial solver.

    Args:
        initial: name of the solver used for the first plan (default PIBT --
            fast, and LNS repairs whatever quality it lacks).
        neighborhood_size: how many agents to re-plan per iteration.
        iterations: hard cap on iterations.
        time_limit: wall-clock budget in seconds; this is the knob that matters,
            since LNS is designed to be stopped rather than to terminate.
        seed: reproducible operator choices and agent sampling.
    """

    def __init__(
        self,
        initial: str = "pibt",
        neighborhood_size: int = 8,
        iterations: int = 100000,
        time_limit: Optional[float] = 5.0,
        seed: Optional[int] = 0,
        initial_kwargs: Optional[Dict] = None,
    ):
        self.initial = initial
        self.neighborhood_size = neighborhood_size
        self.iterations = iterations
        self.time_limit = time_limit
        self.seed = seed
        self.initial_kwargs = initial_kwargs or {}

    # -- destroy operators --------------------------------------------------
    def operators(self):
        """Names of the destroy operators in the roulette.

        Subclasses extend this (and :meth:`_pick_neighborhood`) to add their own
        without reimplementing the search loop.
        """
        return list(DESTROY_OPERATORS)

    def _pick_neighborhood(self, operator, paths, names, rng):
        if operator == "random":
            return self._random_neighborhood(names, rng)
        if operator == "congestion":
            return self._congestion_neighborhood(paths, rng)
        return self._interaction_neighborhood(paths, rng)

    def _random_neighborhood(
        self, names: Sequence[str], rng: random.Random
    ) -> List[str]:
        size = min(self.neighborhood_size, len(names))
        return rng.sample(list(names), size)

    def _congestion_neighborhood(
        self, paths: Dict[str, List[Cell]], rng: random.Random
    ) -> List[str]:
        """Agents that pass through the busiest vertex of the current plan."""
        counts: Dict[Cell, int] = {}
        for path in paths.values():
            for cell in set(path):
                counts[cell] = counts.get(cell, 0) + 1
        if not counts:
            return self._random_neighborhood(list(paths), rng)
        hotspot = max(counts, key=lambda cell: counts[cell])
        users = [name for name, path in paths.items() if hotspot in path]
        rng.shuffle(users)
        chosen = users[: self.neighborhood_size]
        pool = [name for name in paths if name not in chosen]
        rng.shuffle(pool)
        return chosen + pool[: max(0, self.neighborhood_size - len(chosen))]

    def _interaction_neighborhood(
        self, paths: Dict[str, List[Cell]], rng: random.Random
    ) -> List[str]:
        """A random agent plus the agents whose paths brush against its own."""
        names = list(paths)
        seed_agent = rng.choice(names)
        occupied = set(paths[seed_agent])
        neighbours = [
            name
            for name in names
            if name != seed_agent and occupied.intersection(paths[name])
        ]
        rng.shuffle(neighbours)
        chosen = [seed_agent] + neighbours[: self.neighborhood_size - 1]
        pool = [name for name in names if name not in chosen]
        rng.shuffle(pool)
        return chosen + pool[: max(0, self.neighborhood_size - len(chosen))]

    # -- repair -------------------------------------------------------------
    @staticmethod
    def _constraints_from(paths: Dict[str, List[Cell]], horizon: int) -> Constraints:
        constraints = Constraints()
        for path in paths.values():
            for t, cell in enumerate(path):
                constraints.add_vertex(cell, t)
            for t in range(len(path), horizon + 1):
                constraints.add_vertex(path[-1], t)
            for t in range(len(path) - 1):
                constraints.add_edge(path[t + 1], path[t], t + 1)
        return constraints

    def _repair(
        self,
        problem: MAPFProblem,
        kept: Dict[str, List[Cell]],
        replan: Sequence[str],
        agents_by_name: Dict,
        rng: random.Random,
    ) -> Optional[Dict[str, List[Cell]]]:
        order = list(replan)
        rng.shuffle(order)
        result = dict(kept)
        for name in order:
            agent = agents_by_name[name]
            horizon = (
                problem.grid.free_cells
                + sum(len(p) for p in result.values())
                + len(order)
                + 1
            )
            path = sipp(
                problem.grid,
                agent.start,
                agent.goal,
                constraints=self._constraints_from(result, horizon),
                allow_diagonals=problem.allow_diagonals,
                max_timestep=horizon,
            )
            if path is None:
                return None
            result[name] = path
        return result

    # -- main loop ----------------------------------------------------------
    def solve(
        self, problem: MAPFProblem, observer: Optional[Observer] = None
    ) -> Optional[Solution]:
        emit = _Emitter(observer)
        started = time.perf_counter()
        rng = random.Random(self.seed)

        agents_by_name = {a.name: a for a in problem.agents}
        names = list(agents_by_name)
        emit("root", agents=list(names), cost=0)

        initial = get_solver(self.initial, **self.initial_kwargs).solve(problem)
        if initial is None:
            emit("failed", reason="initial solver %r found no solution" % self.initial)
            return None

        best = {name: list(path) for name, path in initial.paths.items()}
        best_cost = sum(len(p) - 1 for p in best.values())
        emit(
            "expand",
            node=0,
            cost=best_cost,
            open=0,
            paths={n: list(p) for n, p in best.items()} if emit else None,
        )

        # Adaptive operator weights: reward whatever is currently working.
        weights = {op: 1.0 for op in self.operators()}
        iterations = 0
        improvements = 0

        while iterations < self.iterations:
            if (
                self.time_limit is not None
                and time.perf_counter() - started > self.time_limit
            ):
                break
            iterations += 1

            operators = self.operators()
            operator = rng.choices(
                operators, weights=[weights[op] for op in operators]
            )[0]
            neighborhood = self._pick_neighborhood(operator, best, names, rng)

            kept = {n: p for n, p in best.items() if n not in neighborhood}
            candidate = self._repair(problem, kept, neighborhood, agents_by_name, rng)
            if candidate is None:
                weights[operator] = max(0.1, weights[operator] * 0.9)
                continue

            cost = sum(len(p) - 1 for p in candidate.values())
            if cost < best_cost and find_first_conflict(candidate) is None:
                best, best_cost = candidate, cost
                improvements += 1
                weights[operator] = min(20.0, weights[operator] + 1.0)
                emit(
                    "expand",
                    node=iterations,
                    cost=best_cost,
                    open=len(neighborhood),
                    paths={n: list(p) for n, p in best.items()} if emit else None,
                )
            else:
                weights[operator] = max(0.1, weights[operator] * 0.97)

        solution = Solution(
            paths=best,
            algorithm=self.name,
            expansions=iterations,
            runtime=time.perf_counter() - started,
        )
        emit(
            "solved",
            cost=solution.sum_of_costs,
            makespan=solution.makespan,
            expansions=iterations,
            improvements=improvements,
            paths={n: list(p) for n, p in solution.paths.items()},
        )
        return solution
