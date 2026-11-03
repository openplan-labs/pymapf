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


def test_re_registering_the_same_class_is_allowed():
    # What re-running a notebook cell does: same module, same qualified name.
    from pymapf.core.solver import _REGISTRY

    try:
        for attempt in range(2):

            @register_solver("rerun-test-solver")
            class _Rerun(MAPFSolver):
                version = attempt

                def solve(self, problem, observer=None):
                    return None

        assert get_solver("rerun-test-solver").version == 1
    finally:
        _REGISTRY.pop("rerun-test-solver", None)


def test_replace_overrides_a_different_class():
    from pymapf.core.solver import _REGISTRY

    try:

        @register_solver("replace-test-solver")
        class _First(MAPFSolver):
            def solve(self, problem, observer=None):
                return None

        with pytest.raises(ValueError, match="replace=True"):

            @register_solver("replace-test-solver")
            class _Second(MAPFSolver):
                def solve(self, problem, observer=None):
                    return None

        @register_solver("replace-test-solver", replace=True)
        class _Third(MAPFSolver):
            def solve(self, problem, observer=None):
                return None

        assert type(get_solver("replace-test-solver")) is _Third
    finally:
        _REGISTRY.pop("replace-test-solver", None)


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


def _corridor_problem():
    grid = GridMap([[0, 0, 0], [1, 1, 0]])
    return MAPFProblem(grid, [Agent("a", (0, 0), (0, 2))])


def test_is_valid_without_a_problem_still_only_checks_conflicts():
    from pymapf.core.solver import Solution

    assert Solution({"a": [(0, 0)]}).is_valid()


@pytest.mark.parametrize(
    "path, message",
    [
        ([(0, 0)], "not its goal"),
        ([(0, 1), (0, 2)], "not its start"),
        ([(0, 0), (0, 2)], "jumps"),
        ([(0, 0), (1, 1), (0, 2)], "blocked"),
    ],
)
def test_violations_check_the_whole_definition(path, message):
    from pymapf.core.solver import Solution

    solution = Solution({"a": path})
    assert not solution.is_valid(_corridor_problem())
    assert any(message in v for v in solution.violations(_corridor_problem()))


def test_a_solver_plan_has_no_violations():
    problem = _corridor_problem()
    solution = pymapf.solve(problem, "cbs")
    assert solution.violations(problem) == []
    assert solution.is_valid(problem)


def test_violations_report_conflicts_and_missing_agents():
    from pymapf.core.solver import Solution

    grid = GridMap([[0, 0]])
    problem = MAPFProblem(
        grid, [Agent("a", (0, 0), (0, 1)), Agent("b", (0, 1), (0, 0))]
    )
    swap = Solution({"a": [(0, 0), (0, 1)], "b": [(0, 1), (0, 0)]})
    assert any("edge conflict" in v for v in swap.violations(problem))
    assert any(
        "'b' has no path" in v
        for v in Solution({"a": [(0, 0), (0, 1)]}).violations(problem)
    )
