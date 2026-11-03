"""CBS against brute force on instances small enough to enumerate.

The weighted-CBS bound tests use CBS as their reference optimum, so a CBS bug
would hide in both. This checks CBS against an exhaustive joint-state search
that shares none of its code: Dijkstra over every agent's position at once.
"""

import heapq
import itertools
import random
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pytest

import pymapf
from pymapf.core import Agent, GridMap, MAPFProblem


def brute_force_sum_of_costs(problem):
    """Optimal sum of costs by Dijkstra over joint states, or ``None``.

    A state is every agent's cell plus which agents have *committed* to
    staying on their goal for good. Uncommitted agents pay 1 per timestep and
    committing is free, so the cheapest all-committed state is the optimal sum
    of costs under the same vertex/swap conflict model the solvers use.
    """
    grid, diagonals = problem.grid, problem.allow_diagonals
    goals = tuple(a.goal for a in problem.agents)
    n = len(goals)
    everyone = (1 << n) - 1
    start = (tuple(a.start for a in problem.agents), 0)
    best = {start: 0}
    heap = [(0, 0, start)]
    tie = itertools.count(1)
    while heap:
        cost, _, state = heapq.heappop(heap)
        if cost > best[state]:
            continue
        cells, committed = state
        if committed == everyone:
            return cost
        successors = []
        for i in range(n):  # commit an agent standing on its goal: free
            if not committed >> i & 1 and cells[i] == goals[i]:
                successors.append((cost, (cells, committed | 1 << i)))
        options = [
            (
                [cells[i]]
                if committed >> i & 1
                else grid.neighbors(cells[i], diagonals) + [cells[i]]
            )
            for i in range(n)
        ]
        step = cost + n - bin(committed).count("1")
        for nxt in itertools.product(*options):
            if len(set(nxt)) < n:
                continue  # vertex conflict
            if any(
                nxt[i] == cells[j] and nxt[j] == cells[i]
                for i in range(n)
                for j in range(i + 1, n)
            ):
                continue  # swap conflict
            successors.append((step, (nxt, committed)))
        for new_cost, new_state in successors:
            if new_cost < best.get(new_state, float("inf")):
                best[new_state] = new_cost
                heapq.heappush(heap, (new_cost, next(tie), new_state))
    return None


# Where Manhattan distance overestimates with diagonals (optimum 5 for "a");
# random tiny maps rarely give a path enough diagonals to show it.
_DIAGONAL_TRAP = GridMap(
    [
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ]
)


def _random_instances(seed, count, diagonals):
    if diagonals:
        a = Agent("a", (0, 0), (3, 4))
        b = Agent("b", (3, 3), (0, 1))
        yield MAPFProblem(_DIAGONAL_TRAP, [a], allow_diagonals=True)
        yield MAPFProblem(_DIAGONAL_TRAP, [a, b], allow_diagonals=True)
    rng = random.Random(seed)
    while count:
        # Diagonal instances get room to move: Manhattan distance only
        # overestimates once a path can take several diagonals.
        sizes = [(3, 4), (4, 4), (4, 5)] if diagonals else [(2, 3), (3, 3), (3, 4)]
        h, w = rng.choice(sizes)
        grid = GridMap([[int(rng.random() < 0.25) for _ in range(w)] for _ in range(h)])
        free = [(r, c) for r in range(h) for c in range(w) if grid.is_free((r, c))]
        n = 2 if diagonals else rng.choice([2, 3])
        if len(free) < 2 * n:
            continue
        cells = rng.sample(free, 2 * n)
        agents = [Agent("a%d" % i, cells[i], cells[n + i]) for i in range(n)]
        yield MAPFProblem(grid, agents, allow_diagonals=diagonals)
        count -= 1


@pytest.mark.parametrize("diagonals", [False, True])
def test_cbs_matches_brute_force_on_tiny_instances(diagonals):
    for problem in _random_instances(seed=11, count=40, diagonals=diagonals):
        optimum = brute_force_sum_of_costs(problem)
        solution = pymapf.solve(problem, "cbs", max_expansions=5000)
        if optimum is None:
            assert solution is None
            continue
        assert solution is not None and solution.is_valid(problem)
        assert solution.sum_of_costs == optimum, problem.agents
        bounded = pymapf.solve(problem, "wcbs", weight=1.5, max_expansions=5000)
        assert bounded.is_valid(problem)
        assert bounded.sum_of_costs <= 1.5 * optimum
