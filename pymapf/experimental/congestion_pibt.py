"""Experiment 1: give PIBT a congestion-aware preference.

**Hypothesis.** PIBT ranks candidate vertices purely by distance-to-goal. On a
map with corridors, that sends every agent down the same shortest route, and
the resulting queue is resolved one collision at a time by priority
inheritance. If the ranking also knew *how contested* a vertex is, agents whose
detour is cheap would spread out on their own, and the flock of inheritance
chains would shrink.

**What changes.** One line of the preference. Instead of

    key(v) = distance(agent, v)

we use

    key(v) = distance(agent, v) + alpha * congestion(v)

where ``congestion(v)`` is a static estimate, computed once per instance, of
how many agents' shortest paths pass through ``v``. Nothing else about PIBT --
priority inheritance, backtracking, dynamic priorities -- is touched.

**Why it might fail.** The estimate is static: it is computed from the initial
shortest paths and never updated, so it is wrong as soon as agents deviate.
And distance-to-goal is what makes PIBT *terminate* in practice; perturbing it
risks agents preferring a scenic route indefinitely. The ``alpha`` sweep in
:mod:`pymapf.experimental.study` is there to find out where that trade turns.

**Relation to published work.** Preference construction in PIBT is an active
topic: Okumura and Nagai (2025) study lightweight preference functions for
large-scale PIBT. This variant is in the same spirit -- a cheap, precomputed
term added to the ranking -- and is offered as a measurable data point, not as
a new algorithm.

References
----------
* Okumura, K.; Machida, M.; Defago, X.; and Tamura, Y. 2022. *Priority
  inheritance with backtracking for iterative multi-agent path finding.*
  Artificial Intelligence 310: 103752.
* Okumura, K.; and Nagai, R. 2025. *Lightweight and effective preference
  construction in PIBT for large-scale multi-agent pathfinding.* SOCS 2025.
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional

from ..algorithms.pibt import PIBT, _DistanceOracle, _trim, pibt_step
from ..algorithms.search import astar
from ..core.grid import Cell
from ..core.solver import MAPFProblem, Observer, Solution, register_solver
from ..core.trace import _Emitter

__all__ = ["CongestionPIBT", "congestion_map"]


def congestion_map(problem: MAPFProblem) -> Dict[Cell, float]:
    """How many agents' individually-shortest paths cross each vertex.

    One A* per agent, ignoring every other agent -- the same lower-bound
    computation CBS already does at its root, so it is essentially free.
    """
    counts: Dict[Cell, float] = {}
    for agent in problem.agents:
        path = astar(
            problem.grid,
            agent.start,
            agent.goal,
            allow_diagonals=problem.allow_diagonals,
        )
        if path is None:
            continue
        for cell in path:
            counts[cell] = counts.get(cell, 0.0) + 1.0
    return counts


@register_solver("x-pibt-congestion")
class CongestionPIBT(PIBT):
    """PIBT whose vertex preference includes a static congestion penalty.

    Args:
        alpha: weight of the congestion term. ``0`` reproduces PIBT exactly.
        normalise: divide the congestion count by the number of agents, so
            ``alpha`` means the same thing regardless of fleet size.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        normalise: bool = True,
        max_timestep: Optional[int] = None,
        seed: Optional[int] = 0,
        time_limit: Optional[float] = None,
    ):
        super().__init__(max_timestep=max_timestep, seed=seed, time_limit=time_limit)
        self.alpha = alpha
        self.normalise = normalise

    def solve(
        self, problem: MAPFProblem, observer: Optional[Observer] = None
    ) -> Optional[Solution]:
        emit = _Emitter(observer)
        started = time.perf_counter()
        rng = random.Random(self.seed)

        agents = list(problem.agents)
        names = [a.name for a in agents]
        goals = {a.name: a.goal for a in agents}
        positions = {a.name: a.start for a in agents}

        exact = _DistanceOracle(problem.grid, goals, problem.allow_diagonals)
        congestion = congestion_map(problem)
        scale = self.alpha / (len(agents) if self.normalise and agents else 1)

        def preference(name: str, cell: Cell) -> float:
            return exact(name, cell) + scale * congestion.get(cell, 0.0)

        # The oracle interface PIBT expects, with the penalty folded in.
        preference.reachable = exact.reachable  # type: ignore[attr-defined]

        for name in names:
            if not exact.reachable(name, positions[name]):
                emit("failed", reason="agent %r cannot reach its goal" % name)
                return None

        base = {name: index / (len(names) + 1) for index, name in enumerate(names)}
        priorities = dict(base)
        horizon = self.max_timestep or (problem.grid.free_cells + 4 * len(names) + 8)
        paths: Dict[str, List[Cell]] = {name: [positions[name]] for name in names}
        emit("root", agents=list(names), cost=0)

        for step in range(horizon):
            if positions == goals:
                break
            if (
                self.time_limit is not None
                and time.perf_counter() - started > self.time_limit
            ):
                emit("failed", reason="time limit (%.3gs) reached" % self.time_limit)
                return None

            nxt = pibt_step(
                problem.grid,
                names,
                positions,
                goals,
                priorities,
                preference,
                allow_diagonals=problem.allow_diagonals,
                rng=rng,
            )
            if nxt is None:
                emit("failed", reason="no valid configuration at timestep %d" % step)
                return None

            positions = nxt
            for name in names:
                paths[name].append(positions[name])
                priorities[name] = (
                    base[name] if positions[name] == goals[name] else priorities[name] + 1
                )

        if positions != goals:
            emit("failed", reason="livelock: %d timesteps without reaching the goals" % horizon)
            return None

        trimmed = {name: _trim(path, goals[name]) for name, path in paths.items()}
        solution = Solution(
            paths=trimmed,
            algorithm=self.name,
            expansions=max((len(p) - 1 for p in trimmed.values()), default=0),
            runtime=time.perf_counter() - started,
        )
        emit(
            "solved",
            cost=solution.sum_of_costs,
            makespan=solution.makespan,
            expansions=solution.expansions,
            paths={n: list(p) for n, p in solution.paths.items()},
        )
        return solution
