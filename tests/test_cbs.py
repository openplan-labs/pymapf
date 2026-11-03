import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pymapf
from pymapf.algorithms.cbs import ConflictBasedSearch
from pymapf.core import Agent, GridMap, MAPFProblem


def _open(n):
    return GridMap([[0] * n for _ in range(n)])


def test_resolves_vertex_conflict_optimally():
    # Both agents' independent shortest paths cross the centre cell (1,1) at t=1.
    prob = MAPFProblem(
        _open(3), [Agent("a", (0, 1), (2, 1)), Agent("b", (1, 0), (1, 2))]
    )
    sol = ConflictBasedSearch().solve(prob)
    assert sol is not None
    assert sol.is_valid()
    # Optimal sum-of-costs: one agent detours/waits one step -> 2 + 3 = 5.
    assert sol.sum_of_costs == 5


def test_resolves_edge_swap_conflict():
    # 2x3 grid so agents can dodge into the other row instead of swapping.
    prob = MAPFProblem(
        GridMap([[0, 0, 0], [0, 0, 0]]),
        [Agent("a", (0, 0), (0, 2)), Agent("b", (0, 2), (0, 0))],
    )
    sol = ConflictBasedSearch().solve(prob)
    assert sol is not None and sol.is_valid()


def test_single_agent_shortest_path():
    prob = MAPFProblem(_open(4), [Agent("a", (0, 0), (3, 3))])
    sol = ConflictBasedSearch().solve(prob)
    assert sol is not None and sol.sum_of_costs == 6  # Manhattan distance


def test_unsolvable_returns_none():
    corridor = GridMap([[0, 0, 0, 0, 0]])
    prob = MAPFProblem(
        corridor, [Agent("a", (0, 0), (0, 4)), Agent("b", (0, 4), (0, 0))]
    )
    assert ConflictBasedSearch().solve(prob) is None


def test_expansion_cap_returns_none():
    prob = MAPFProblem(
        _open(3), [Agent("a", (0, 1), (2, 1)), Agent("b", (1, 0), (1, 2))]
    )
    # A cap of 1 cannot get past the root conflict.
    assert ConflictBasedSearch(max_expansions=1).solve(prob) is None


def test_optimal_with_diagonal_moves():
    # Manhattan distance overestimates with diagonals; the default heuristic
    # must not, or CBS (and everything built on space-time A*) is suboptimal.
    grid = GridMap(
        [
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ]
    )
    problem = MAPFProblem(grid, [Agent("a", (0, 0), (3, 4))], allow_diagonals=True)
    for solver in ("cbs", "wcbs", "prioritized", "lns"):
        kwargs = {"time_limit": 0.2} if solver == "lns" else {}
        if solver == "wcbs":
            kwargs = {"weight": 1.0}
        sol = pymapf.solve(problem, solver, **kwargs)
        assert sol.sum_of_costs == 5, solver


def test_diagonal_crossing_is_not_a_conflict_in_this_model():
    # Documented in find_first_conflict: the standard vertex/edge model does
    # not see two agents crossing diagonally through one 2x2 block.
    from pymapf.core.solver import find_first_conflict

    assert find_first_conflict({"a": [(0, 0), (1, 1)], "b": [(0, 1), (1, 0)]}) is None
