"""Experiment 2: choose LNS neighbourhoods by *delay*, not by proximity.

**Hypothesis.** The agents worth re-planning are the ones currently paying the
most for coordination. An agent whose path costs exactly its individual
shortest-path length has nothing to gain from being re-planned; an agent
travelling 14 steps on a 6-step shortest path is carrying the whole cost of
some avoided conflict, and it -- together with whoever pushed it aside -- is
where the slack is.

**What changes.** One destroy operator. ``delay(a) = |path(a)| - lower_bound(a)``
is computed from the individually-shortest paths (already available, and the
same lower bound CBS uses at its root). The operator picks the most-delayed
agent, then fills the neighbourhood with the agents whose paths intersect its
own -- the ones that plausibly caused the delay.

**Why it might fail.** Delay is a symptom, not a cause: the most-delayed agent
may be delayed by a wall, not by a peer, in which case re-planning it changes
nothing and the iteration is wasted. Whether that outweighs the better
targeting is an empirical question, which is the point of the study.

**Relation to published work.** MAPF-LNS already includes an "agent-based"
neighbourhood built from delayed agents; this variant is a re-implementation of
that idea inside this library's operator framework, measured against the random
and congestion operators that ship with :class:`~pymapf.algorithms.lns.LargeNeighborhoodSearch`.

References
----------
* Li, J.; Chen, Z.; Harabor, D.; Stuckey, P. J.; and Koenig, S. 2021. *Anytime
  multi-agent path finding via large neighborhood search.* IJCAI 2021:
  4127-4135.
* Li, J.; Chen, Z.; Harabor, D.; Stuckey, P. J.; and Koenig, S. 2022.
  *MAPF-LNS2: Fast repairing for multi-agent path finding via large
  neighborhood search.* AAAI 2022: 10256-10265.
"""

from __future__ import annotations

import random
from typing import Dict, List

from ..algorithms.lns import LargeNeighborhoodSearch
from ..algorithms.search import astar
from ..core.grid import Cell
from ..core.solver import MAPFProblem, register_solver

__all__ = ["DelayLNS"]


@register_solver("x-lns-delay")
class DelayLNS(LargeNeighborhoodSearch):
    """LNS with a delay-targeted destroy operator added to the roulette."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lower_bounds: Dict[str, int] = {}

    def _prepare(self, problem: MAPFProblem) -> None:
        """Individually-shortest path lengths: the per-agent cost lower bound."""
        self._lower_bounds = {}
        for agent in problem.agents:
            path = astar(
                problem.grid,
                agent.start,
                agent.goal,
                allow_diagonals=problem.allow_diagonals,
            )
            self._lower_bounds[agent.name] = (len(path) - 1) if path else 0

    def _delay_neighborhood(
        self, paths: Dict[str, List[Cell]], rng: random.Random
    ) -> List[str]:
        delays = {
            name: (len(path) - 1) - self._lower_bounds.get(name, 0)
            for name, path in paths.items()
        }
        ranked = sorted(delays, key=lambda name: -delays[name])
        if not ranked or delays[ranked[0]] <= 0:
            return self._random_neighborhood(list(paths), rng)

        # Sample the seed among the most-delayed few, so repeated iterations do
        # not keep attacking the same agent when it cannot be improved.
        seed_agent = rng.choice(ranked[: max(1, len(ranked) // 4)])
        touched = set(paths[seed_agent])
        companions = [
            name
            for name in ranked
            if name != seed_agent and touched.intersection(paths[name])
        ]
        chosen = [seed_agent] + companions[: self.neighborhood_size - 1]
        pool = [name for name in paths if name not in chosen]
        rng.shuffle(pool)
        return chosen + pool[: max(0, self.neighborhood_size - len(chosen))]

    def operators(self):
        return super().operators() + ["delay"]

    def solve(self, problem, observer=None):
        self._prepare(problem)
        return super().solve(problem, observer=observer)

    # The base class picks operators by name; extend the table rather than
    # copying the whole loop.
    def _pick_neighborhood(self, operator, paths, names, rng):
        if operator == "delay":
            return self._delay_neighborhood(paths, rng)
        return super()._pick_neighborhood(operator, paths, names, rng)
