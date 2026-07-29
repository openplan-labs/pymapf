import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pytest

import pymapf
from pymapf.algorithms import ConflictBasedSearch, WeightedCBS
from pymapf.core import Agent, GridMap, MAPFProblem


def _crossing_problem():
    grid = GridMap([[0] * 3 for _ in range(3)])
    return MAPFProblem(grid, [Agent("a", (0, 1), (2, 1)), Agent("b", (1, 0), (1, 2))])


def test_registered_under_wcbs():
    assert "wcbs" in pymapf.available_solvers()
    assert isinstance(pymapf.get_solver("wcbs"), WeightedCBS)


def test_returns_a_valid_solution():
    solution = WeightedCBS(weight=1.5).solve(_crossing_problem())
    assert solution is not None
    assert solution.is_valid()
    assert solution.algorithm == "wcbs"


def test_weight_one_matches_optimal_cost():
    problem = _crossing_problem()
    optimal = ConflictBasedSearch().solve(problem)
    bounded = WeightedCBS(weight=1.0).solve(problem)
    assert bounded.sum_of_costs == optimal.sum_of_costs


@pytest.mark.parametrize("scenario_name", ["corner_swap", "warehouse", "empty_room"])
def test_stays_within_its_suboptimality_bound(scenario_name):
    scenario = pymapf.build_scenario(scenario_name, seed=2)
    problem = scenario.to_problem()
    optimal = ConflictBasedSearch(time_limit=10).solve(problem)
    if optimal is None:  # instance too hard for the optimal solver: nothing to bound
        pytest.skip("no optimal reference within the time limit")
    weight = 1.5
    bounded = WeightedCBS(weight=weight).solve(problem)
    assert bounded is not None
    assert bounded.is_valid()
    assert bounded.sum_of_costs <= weight * optimal.sum_of_costs


def test_rejects_a_weight_below_one():
    with pytest.raises(ValueError):
        WeightedCBS(weight=0.9)


def test_reports_failure_when_out_of_time():
    scenario = pymapf.build_scenario("maze", seed=0)
    trace = pymapf.SearchTrace()
    solution = WeightedCBS(weight=1.2, time_limit=0.05).solve(
        scenario.to_problem(), observer=trace
    )
    if solution is None:
        assert "time limit" in trace.of_kind("failed")[-1]["reason"]


def test_expansion_cap_is_honoured():
    scenario = pymapf.build_scenario("bottleneck", seed=0)
    trace = pymapf.SearchTrace()
    solution = WeightedCBS(weight=1.1, max_expansions=5).solve(
        scenario.to_problem(), observer=trace
    )
    assert trace.expansions <= 5
    if solution is None:
        assert "expansion limit" in trace.of_kind("failed")[-1]["reason"]


def test_unsolvable_instance_returns_none():
    grid = GridMap([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
    problem = MAPFProblem(
        grid, [Agent("a", (1, 0), (1, 2)), Agent("b", (1, 2), (1, 0))]
    )
    assert WeightedCBS(weight=2.0, max_expansions=200).solve(problem) is None
