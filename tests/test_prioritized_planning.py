import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from pymapf.algorithms.prioritized_planning import PrioritizedPlanning
from pymapf.core import Agent, GridMap, MAPFProblem


def _open(n):
    return GridMap([[0] * n for _ in range(n)])


def test_solves_crossing_conflict():
    prob = MAPFProblem(
        _open(3), [Agent("a", (0, 1), (2, 1)), Agent("b", (1, 0), (1, 2))]
    )
    sol = PrioritizedPlanning().solve(prob)
    assert sol is not None and sol.is_valid()


def test_paths_start_and_end_correctly():
    prob = MAPFProblem(
        _open(4), [Agent("a", (0, 0), (3, 3)), Agent("b", (3, 0), (0, 3))]
    )
    sol = PrioritizedPlanning().solve(prob)
    assert sol is not None
    for agent in prob.agents:
        p = sol.paths[agent.name]
        assert p[0] == agent.start and p[-1] == agent.goal


def test_priority_order_respected():
    prob = MAPFProblem(
        _open(3), [Agent("a", (0, 0), (0, 2)), Agent("b", (2, 0), (2, 2))]
    )
    sol = PrioritizedPlanning(priority=["b", "a"]).solve(prob)
    assert sol is not None and sol.is_valid()


def test_unsolvable_corridor_swap_returns_none():
    corridor = GridMap([[0, 0, 0, 0, 0]])
    prob = MAPFProblem(
        corridor, [Agent("a", (0, 0), (0, 4)), Agent("b", (0, 4), (0, 0))]
    )
    assert PrioritizedPlanning().solve(prob) is None


def test_priority_with_a_repeated_name_raises():
    import pytest

    from pymapf.algorithms.prioritized_planning import PrioritizedPlanning
    from pymapf.core import Agent, GridMap, MAPFProblem

    problem = MAPFProblem(
        GridMap([[0, 0, 0], [0, 0, 0]]),
        [Agent("a", (0, 0), (0, 2)), Agent("b", (1, 0), (1, 2))],
    )
    with pytest.raises(ValueError, match="exactly once"):
        PrioritizedPlanning(priority=["a", "a", "b"]).solve(problem)
