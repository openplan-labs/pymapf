import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pytest

import pymapf
from pymapf.core import (
    Agent,
    GridMap,
    MAPFProblem,
    available_solvers,
    get_solver,
    register_solver,
)
from pymapf.core.solver import MAPFSolver


def test_builtin_solvers_registered():
    assert "cbs" in available_solvers()
    assert "prioritized" in available_solvers()


def test_get_solver_forwards_kwargs():
    s = get_solver("cbs", heuristic="euclidean")
    assert s.heuristic == "euclidean"


def test_get_unknown_solver_raises():
    with pytest.raises(ValueError):
        get_solver("nope")


def test_top_level_solve_helper():
    grid = GridMap([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    prob = MAPFProblem(grid, [Agent("a", (0, 0), (2, 2)), Agent("b", (2, 0), (0, 2))])
    sol = pymapf.solve(prob, "cbs")
    assert sol is not None and sol.is_valid()
    assert sol.algorithm == "cbs"


def test_duplicate_registration_raises():
    with pytest.raises(ValueError):

        @register_solver("cbs")
        class _Dup(MAPFSolver):
            def solve(self, problem):
                return None


def test_custom_solver_registration_and_use():
    @register_solver("noop-test-solver")
    class _Noop(MAPFSolver):
        def solve(self, problem):
            return None

    assert "noop-test-solver" in available_solvers()
    assert get_solver("noop-test-solver").solve(None) is None


def test_problem_rejects_two_agents_on_one_start():
    # Planners used to return invalid plans (PP, LaCAM) or burn their whole
    # budget (CBS) on an instance that is unsolvable from t = 0.
    grid = GridMap([[0, 0, 0], [0, 0, 0]])
    with pytest.raises(ValueError, match="'a' and 'b' share the start"):
        MAPFProblem(grid, [Agent("a", (0, 0), (1, 2)), Agent("b", (0, 0), (1, 1))])


def test_problem_rejects_two_agents_on_one_goal():
    grid = GridMap([[0, 0, 0], [0, 0, 0]])
    with pytest.raises(ValueError, match="'a' and 'b' share the goal"):
        MAPFProblem(grid, [Agent("a", (0, 0), (1, 2)), Agent("b", (0, 1), (1, 2))])


def test_list_cells_become_tuples():
    grid = GridMap([[0, 0, 0]])
    problem = MAPFProblem(grid, [Agent("a", [0, 0], [0, 2])])
    assert problem.agents[0].start == (0, 0)
    assert pymapf.solve(problem, "cbs").paths["a"] == [(0, 0), (0, 1), (0, 2)]
