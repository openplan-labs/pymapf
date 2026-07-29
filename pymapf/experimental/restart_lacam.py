"""Experiment 3: what should LaCAM do with a leftover time budget?

**Hypothesis.** LaCAM finds its first solution in milliseconds. Given a budget
of, say, five seconds, the remaining 4.99 s should buy something. LaCAM*
(Okumura 2023) answers this by continuing the same search and relaxing g-values
over the explored configuration graph. This variant asks whether a much simpler
use of the budget competes: **restart** the search with a different random seed
and keep the best solution, exploiting the fact that LaCAM's quality is
dominated by PIBT's randomised tie-breaking.

**What changes.** Nothing inside the search. The solver runs LaCAM repeatedly
with seeds ``seed, seed+1, ...`` until the budget is spent, and returns the
cheapest plan any run produced.

**Why it might fail.** Restarts throw away everything learned; the cost
distribution over seeds may be tight, in which case ten restarts buy nothing
that one run did not already give. It also cannot converge to optimal, which
LaCAM* provably does given enough time.

**Honest positioning.** This is not a better LaCAM. It is a baseline that any
anytime scheme should have to beat, and it exists here because
:mod:`pymapf.experimental.study` measured the in-search continuation and found
it flat on these instances -- a negative result worth recording rather than
hiding.

References
----------
* Okumura, K. 2023. *LaCAM: Search-based algorithm for quick multi-agent
  pathfinding.* AAAI 2023, 37(10): 11655-11662.
* Okumura, K. 2023. *Improving LaCAM for scalable eventually optimal
  multi-agent pathfinding.* IJCAI 2023: 243-251.
* Luby, M.; Sinclair, A.; and Zuckerman, D. 1993. *Optimal speedup of Las Vegas
  algorithms.* Information Processing Letters 47(4): 173-180.  (why restarting
  a randomised search can dominate running it longer)
"""

from __future__ import annotations

import time
from typing import Optional

from ..algorithms.lacam import LaCAM
from ..core.solver import MAPFProblem, MAPFSolver, Observer, Solution, register_solver
from ..core.trace import _Emitter

__all__ = ["RestartLaCAM"]


@register_solver("x-lacam-restart")
class RestartLaCAM(MAPFSolver):
    """Run LaCAM repeatedly with fresh seeds; keep the cheapest plan.

    Args:
        time_limit: total budget across all restarts.
        per_run_limit: budget for a single restart. Small values favour many
            samples; large values give each run room to work through a hard
            instance.
        seed: first seed; subsequent restarts use ``seed + k``.
    """

    def __init__(
        self,
        time_limit: Optional[float] = 5.0,
        per_run_limit: float = 1.0,
        seed: int = 0,
        max_restarts: int = 200,
    ):
        self.time_limit = time_limit
        self.per_run_limit = per_run_limit
        self.seed = seed
        self.max_restarts = max_restarts

    def solve(
        self, problem: MAPFProblem, observer: Optional[Observer] = None
    ) -> Optional[Solution]:
        emit = _Emitter(observer)
        started = time.perf_counter()
        best: Optional[Solution] = None
        restarts = 0

        while restarts < self.max_restarts:
            elapsed = time.perf_counter() - started
            if self.time_limit is not None and elapsed >= self.time_limit:
                break
            remaining = (
                self.per_run_limit
                if self.time_limit is None
                else min(self.per_run_limit, self.time_limit - elapsed)
            )
            if remaining <= 0:
                break

            candidate = LaCAM(
                anytime=False, time_limit=remaining, seed=self.seed + restarts
            ).solve(problem)
            restarts += 1

            if candidate is not None and (
                best is None or candidate.sum_of_costs < best.sum_of_costs
            ):
                best = candidate
                emit(
                    "expand",
                    node=restarts,
                    cost=best.sum_of_costs,
                    open=0,
                    paths={n: list(p) for n, p in best.paths.items()} if emit else None,
                )

        if best is None:
            emit(
                "failed",
                reason="no restart found a solution in %.3gs" % (self.time_limit or 0),
            )
            return None

        solution = Solution(
            paths=best.paths,
            algorithm=self.name,
            expansions=restarts,
            runtime=time.perf_counter() - started,
        )
        emit(
            "solved",
            cost=solution.sum_of_costs,
            makespan=solution.makespan,
            expansions=restarts,
            paths={n: list(p) for n, p in solution.paths.items()},
        )
        return solution
