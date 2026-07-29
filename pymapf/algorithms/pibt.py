"""Priority Inheritance with Backtracking (PIBT).

PIBT abandons the idea of planning whole paths. It plans *one timestep at a
time*: every agent proposes the neighbouring vertex that takes it closest to
its goal, and conflicts are settled on the spot by priority inheritance -- a
high-priority agent that wants an occupied vertex temporarily lends its
priority to the occupant, which must then move out of the way, recursively. If
the occupant cannot, the caller backtracks and tries its next-best vertex.

The result is startlingly effective: each timestep costs O(agents x degree) and
the whole thing scales to thousands of agents, at the price of completeness --
PIBT can livelock on instances that require an agent to move *away* from its
goal for a long time. :mod:`pymapf.algorithms.lacam` fixes exactly that by
wrapping PIBT in a complete search.

Priorities are dynamic: an agent's priority grows while it is away from its
goal and resets when it arrives, which is what stops a single unlucky agent
from starving forever (Okumura et al. 2022, Section 4).

References
----------
* Okumura, K.; Machida, M.; Defago, X.; and Tamura, Y. 2022. *Priority
  inheritance with backtracking for iterative multi-agent path finding.*
  Artificial Intelligence 310: 103752.  (Earlier version: IJCAI 2019:
  535-542.)
* Okumura, K.; Tamura, Y.; and Defago, X. 2021. *Iterative refinement for
  real-time multi-robot path planning.* IROS 2021: 9690-9697.  (the refinement
  idea this module's ``refine`` hook mirrors)
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.grid import Cell
from ..core.solver import (
    Agent,
    MAPFProblem,
    MAPFSolver,
    Observer,
    Solution,
    register_solver,
)
from ..core.trace import _Emitter
from .search import distance_table

__all__ = ["PIBT", "pibt_step"]

Configuration = Tuple[Cell, ...]


class _DistanceOracle:
    """Exact goal distances per agent, computed once and reused every step.

    PIBT asks "which neighbour is closest to my goal?" millions of times, so a
    Manhattan estimate is not good enough: on a map with walls it points agents
    into dead ends. One backward Dijkstra per goal makes every query exact.
    """

    def __init__(self, grid, goals: Dict[str, Cell], allow_diagonals: bool = False):
        self._tables = {
            name: distance_table(grid, goal, allow_diagonals=allow_diagonals)
            for name, goal in goals.items()
        }

    def __call__(self, name: str, cell: Cell) -> float:
        return self._tables[name].get(cell, float("inf"))

    def reachable(self, name: str, cell: Cell) -> bool:
        return cell in self._tables[name]


def pibt_step(
    grid,
    names: Sequence[str],
    positions: Dict[str, Cell],
    goals: Dict[str, Cell],
    priorities: Dict[str, float],
    distance,
    allow_diagonals: bool = False,
    forced: Optional[Dict[str, Cell]] = None,
    rng: Optional[random.Random] = None,
) -> Optional[Dict[str, Cell]]:
    """Compute one timestep for every agent.

    Args:
        forced: partial assignment ``{agent: vertex}`` that must hold in the
            result. This is the hook :mod:`~pymapf.algorithms.lacam` uses to
            steer PIBT away from a configuration it has already seen.

    Returns the next configuration, or ``None`` if the forced assignment cannot
    be satisfied.
    """
    rng = rng or random
    forced = forced or {}
    occupied_now: Dict[Cell, str] = {cell: name for name, cell in positions.items()}
    occupied_next: Dict[Cell, str] = {}
    decided: Dict[str, Cell] = {}

    # Honour the forced assignment first: those agents are not negotiable.
    for name, cell in forced.items():
        if cell in occupied_next:
            return None
        current = positions[name]
        if cell != current and cell not in grid.neighbors(current, allow_diagonals):
            return None
        occupied_next[cell] = name
        decided[name] = cell

    # Every assignment is journalled so a failed inheritance chain can be undone
    # completely. Undoing only the caller's own choice (the obvious
    # implementation) leaks assignments made deeper in the recursion, and those
    # leaks are exactly where invalid configurations come from.
    journal: List[str] = []

    def assign(name: str, cell: Cell) -> None:
        occupied_next[cell] = name
        decided[name] = cell
        journal.append(name)

    def rollback(mark: int) -> None:
        while len(journal) > mark:
            name = journal.pop()
            cell = decided.pop(name)
            if occupied_next.get(cell) == name:
                del occupied_next[cell]

    def funnel(name: str, higher: Optional[str]) -> bool:
        """Try to give ``name`` a vertex; recursive priority inheritance."""
        current = positions[name]
        candidates = list(grid.neighbors(current, allow_diagonals)) + [current]
        # Closest to the goal first; shuffle beforehand so ties are broken
        # differently on every call (the randomness LaCAM relies on).
        rng.shuffle(candidates)
        candidates.sort(key=lambda cell: distance(name, cell))

        for cell in candidates:
            if cell in occupied_next:
                continue
            other = occupied_now.get(cell)
            # Refuse a head-on swap with the agent that called us.
            if higher is not None and cell == positions[higher]:
                continue
            # ...and with anyone who has already committed to *our* vertex.
            # Without this an earlier-decided agent and a later one can trade
            # places, which is a legal pair of vertex moves but an edge
            # conflict. It is the one collision case priority inheritance does
            # not rule out on its own.
            if other is not None and decided.get(other) == current:
                continue

            mark = len(journal)
            assign(name, cell)
            if other is not None and other != name and other not in decided:
                if not funnel(other, name):
                    rollback(mark)
                    continue
            return True

        # Nowhere to go: stay put if that vertex is still free. If it is not,
        # this agent has no legal action at all and the caller must back out.
        if current not in occupied_next:
            assign(name, current)
            return True
        return False

    order = sorted(names, key=lambda name: -priorities[name])
    for name in order:
        if name not in decided and not funnel(name, None):
            return None  # no valid configuration from here

    if len(decided) != len(names):
        return None

    # Cheap final proof: distinct vertices, and no pair trading places. PIBT is
    # meant to guarantee both; verifying costs O(n) and turns any residual bug
    # into a failed step rather than a silently invalid plan.
    if len(set(decided.values())) != len(decided):
        return None
    for name, cell in decided.items():
        other = occupied_now.get(cell)
        if (
            other is not None
            and other != name
            and decided.get(other) == positions[name]
        ):
            return None

    return dict(decided)


@register_solver("pibt")
class PIBT(MAPFSolver):
    """Rule-based, one-step-at-a-time MAPF (Okumura et al. 2022).

    Fast and highly scalable, but *incomplete*: it may fail to solve instances
    that need agents to back far away from their goals. Use it directly when
    speed dominates, or as the configuration generator inside
    :class:`~pymapf.algorithms.lacam.LaCAM`, which restores completeness.

    Args:
        max_timestep: give up after this many timesteps (default: a bound
            proportional to the map size and the number of agents).
        seed: fixes the tie-breaking, making a run reproducible.
        time_limit: optional wall-clock budget in seconds.
    """

    def __init__(
        self,
        max_timestep: Optional[int] = None,
        seed: Optional[int] = 0,
        time_limit: Optional[float] = None,
    ):
        self.max_timestep = max_timestep
        self.seed = seed
        self.time_limit = time_limit

    def solve(
        self, problem: MAPFProblem, observer: Optional[Observer] = None
    ) -> Optional[Solution]:
        emit = _Emitter(observer)
        started = time.perf_counter()
        rng = random.Random(self.seed)

        agents: List[Agent] = list(problem.agents)
        names = [a.name for a in agents]
        goals = {a.name: a.goal for a in agents}
        positions = {a.name: a.start for a in agents}
        distance = _DistanceOracle(problem.grid, goals, problem.allow_diagonals)

        for name in names:
            if not distance.reachable(name, positions[name]):
                emit("failed", reason="agent %r cannot reach its goal" % name)
                return None

        # Priority = "how long have I been away from my goal", plus a fixed
        # per-agent offset that breaks ties deterministically.
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
                distance,
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
                    base[name]
                    if positions[name] == goals[name]
                    else priorities[name] + 1
                )

            emit(
                "expand",
                node=step + 1,
                cost=sum(len(p) - 1 for p in paths.values()),
                open=sum(1 for n in names if positions[n] != goals[n]),
                paths={n: list(p) for n, p in paths.items()} if emit else None,
            )

        if positions != goals:
            emit(
                "failed",
                reason="livelock: %d timesteps without reaching the goals" % horizon,
            )
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


def _trim(path: List[Cell], goal: Cell) -> List[Cell]:
    """Drop the tail an agent spends parked on its goal (it costs nothing)."""
    end = len(path) - 1
    while end > 0 and path[end] == goal and path[end - 1] == goal:
        end -= 1
    return path[: end + 1]
