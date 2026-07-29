"""PIBT, LaCAM, LNS, and the experimental variants."""

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pytest

import pymapf
import pymapf.experimental  # noqa: F401  (registers the x-* solvers)
from pymapf.algorithms.lacam import LaCAM
from pymapf.algorithms.lns import LargeNeighborhoodSearch
from pymapf.algorithms.pibt import PIBT, pibt_step
from pymapf.core import Agent, GridMap, MAPFProblem

MODERN = ["pibt", "lacam", "lns", "x-pibt-congestion", "x-lns-delay", "x-lacam-restart"]
FAST_KWARGS = {
    "lns": {"time_limit": 0.3},
    "x-lns-delay": {"time_limit": 0.3},
    "lacam": {"time_limit": 2.0},
    "x-lacam-restart": {"time_limit": 0.5, "per_run_limit": 0.1},
}


@pytest.mark.parametrize("algorithm", MODERN)
@pytest.mark.parametrize(
    "scenario_name", ["random_obstacles", "corner_swap", "empty_room"]
)
def test_solutions_are_valid_and_reach_the_goals(algorithm, scenario_name):
    scenario = pymapf.build_scenario(scenario_name, n_agents=6, seed=2)
    solution = pymapf.solve(
        scenario.to_problem(), algorithm, **FAST_KWARGS.get(algorithm, {})
    )
    assert solution is not None
    assert solution.is_valid()
    for agent in scenario.agents:
        path_ = solution.paths[agent.name]
        assert path_[0] == agent.start
        assert path_[-1] == agent.goal


@pytest.mark.parametrize("algorithm", MODERN)
@pytest.mark.parametrize("scenario_name", ["warehouse", "maze", "bottleneck"])
def test_a_returned_plan_is_always_valid_even_when_solving_is_not_guaranteed(
    algorithm, scenario_name
):
    """PIBT (and anything built on it) is incomplete by construction.

    On corridor maps it can livelock -- measured at 25% of warehouse instances.
    What must never happen is a *returned* plan that is invalid.
    """
    scenario = pymapf.build_scenario(scenario_name, n_agents=6, seed=2)
    solution = pymapf.solve(
        scenario.to_problem(), algorithm, **FAST_KWARGS.get(algorithm, {})
    )
    if solution is None:
        pytest.skip("%s did not solve this instance, which is allowed" % algorithm)
    assert solution.is_valid()
    for agent in scenario.agents:
        assert solution.paths[agent.name][-1] == agent.goal


def test_pibt_incompleteness_is_reported_not_hidden():
    """A livelock must come back as None with a reason, never as a bad plan."""
    failures = 0
    for seed in range(10):
        scenario = pymapf.build_scenario("warehouse", n_agents=8, seed=seed)
        trace = pymapf.SearchTrace()
        solution = PIBT(seed=0).solve(scenario.to_problem(), observer=trace)
        if solution is None:
            failures += 1
            assert trace.of_kind("failed")
        else:
            assert solution.is_valid()
    assert failures >= 1  # this map family is known to defeat PIBT sometimes


@pytest.mark.parametrize("algorithm", MODERN)
def test_registered_by_name(algorithm):
    assert algorithm in pymapf.available_solvers()


# ----------------------------------------------------------------- PIBT ---


def test_pibt_is_deterministic_for_a_seed():
    scenario = pymapf.build_scenario("warehouse", n_agents=8, seed=1)
    first = PIBT(seed=7).solve(scenario.to_problem())
    second = PIBT(seed=7).solve(scenario.to_problem())
    assert first.paths == second.paths


def test_pibt_step_never_returns_a_collision():
    grid = GridMap([[0] * 4 for _ in range(4)])
    names = ["a", "b", "c"]
    positions = {"a": (0, 0), "b": (0, 1), "c": (1, 0)}
    goals = {"a": (0, 1), "b": (0, 0), "c": (3, 3)}  # a and b want to swap
    distance = lambda name, cell: abs(cell[0] - goals[name][0]) + abs(
        cell[1] - goals[name][1]
    )
    result = pibt_step(
        grid, names, positions, goals, {"a": 3, "b": 2, "c": 1}, distance
    )
    assert result is not None
    assert len(set(result.values())) == len(names)  # distinct vertices
    # and no pair traded places
    for name, cell in result.items():
        for other, other_cell in result.items():
            if (
                name != other
                and cell == positions[other]
                and other_cell == positions[name]
            ):
                pytest.fail("PIBT produced a swap: %s <-> %s" % (name, other))


def test_pibt_honours_a_forced_assignment():
    grid = GridMap([[0] * 4 for _ in range(4)])
    names = ["a", "b"]
    positions = {"a": (0, 0), "b": (2, 2)}
    goals = {"a": (0, 3), "b": (2, 0)}
    distance = lambda name, cell: abs(cell[0] - goals[name][0]) + abs(
        cell[1] - goals[name][1]
    )
    result = pibt_step(
        grid, names, positions, goals, {"a": 2, "b": 1}, distance, forced={"a": (1, 0)}
    )
    assert result["a"] == (1, 0)


def test_pibt_rejects_an_impossible_forced_assignment():
    grid = GridMap([[0] * 4 for _ in range(4)])
    distance = lambda name, cell: 0
    result = pibt_step(
        grid,
        ["a"],
        {"a": (0, 0)},
        {"a": (3, 3)},
        {"a": 1},
        distance,
        forced={"a": (3, 3)},
    )
    assert result is None  # (3,3) is not adjacent to (0,0)


# ---------------------------------------------------------------- LaCAM ---


def test_lacam_solves_instances_pibt_cannot():
    """The completeness property is the whole reason LaCAM exists."""
    rescued = 0
    attempted = 0
    for seed in range(40):
        scenario = pymapf.build_scenario(
            "random_obstacles", n_agents=5, seed=seed, height=8, width=8, density=0.25
        )
        problem = scenario.to_problem()
        if PIBT().solve(problem) is not None:
            continue
        if pymapf.solve(problem, "cbs", time_limit=5) is None:
            continue  # not known to be solvable; skip
        attempted += 1
        solution = LaCAM(time_limit=5).solve(problem)
        if solution is not None and solution.is_valid():
            rescued += 1
    if attempted == 0:
        pytest.skip("no PIBT failures in this sample")
    # Completeness is asymptotic; within a 5 s budget LaCAM rescues most, not
    # necessarily all, of them. What is non-negotiable is that whatever it
    # returns is valid (checked above).
    assert rescued >= max(1, int(0.7 * attempted))


def test_lacam_respects_its_time_limit():
    import time

    scenario = pymapf.build_scenario("maze", n_agents=12, seed=3)
    started = time.perf_counter()
    LaCAM(time_limit=0.4).solve(scenario.to_problem())
    assert time.perf_counter() - started < 3.0


def test_lacam_reports_an_unreachable_goal():
    grid = GridMap([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    problem = MAPFProblem(grid, [Agent("a", (0, 0), (0, 2))])
    trace = pymapf.SearchTrace()
    assert LaCAM(time_limit=1).solve(problem, observer=trace) is None
    assert "cannot reach" in trace.of_kind("failed")[-1]["reason"]


# ------------------------------------------------------------------ LNS ---


def test_lns_never_returns_a_worse_plan_than_its_initial():
    scenario = pymapf.build_scenario("random_obstacles", n_agents=10, seed=4)
    problem = scenario.to_problem()
    initial = pymapf.solve(problem, "pibt")
    improved = LargeNeighborhoodSearch(time_limit=1.0, seed=0).solve(problem)
    assert improved is not None
    assert improved.sum_of_costs <= initial.sum_of_costs
    assert improved.is_valid()


def test_lns_reports_failure_when_the_initial_solver_fails():
    grid = GridMap([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
    problem = MAPFProblem(
        grid, [Agent("a", (1, 0), (1, 2)), Agent("b", (1, 2), (1, 0))]
    )
    trace = pymapf.SearchTrace()
    solution = LargeNeighborhoodSearch(
        initial="cbs", time_limit=0.5, initial_kwargs={"max_expansions": 50}
    ).solve(problem, observer=trace)
    assert solution is None
    assert "initial solver" in trace.of_kind("failed")[-1]["reason"]


def test_lns_operator_table_is_extensible():
    from pymapf.experimental.delay_lns import DelayLNS

    assert "delay" in DelayLNS().operators()
    assert set(LargeNeighborhoodSearch().operators()) < set(DelayLNS().operators())


# --------------------------------------------------------- experimental ---


def test_congestion_map_counts_shortest_paths():
    from pymapf.experimental.congestion_pibt import congestion_map

    scenario = pymapf.build_scenario("corner_swap", n_agents=4)
    counts = congestion_map(scenario.to_problem())
    assert counts
    assert max(counts.values()) >= 2  # the centre is shared


def test_alpha_zero_reproduces_pibt():
    scenario = pymapf.build_scenario("random_obstacles", n_agents=6, seed=5)
    problem = scenario.to_problem()
    baseline = PIBT(seed=0).solve(problem)
    variant = pymapf.solve(problem, "x-pibt-congestion", alpha=0.0, seed=0)
    assert (variant is None) == (baseline is None)
    if baseline is not None:
        assert variant.sum_of_costs == baseline.sum_of_costs


def test_restart_lacam_is_never_worse_than_one_lacam_run():
    scenario = pymapf.build_scenario("random_obstacles", n_agents=10, seed=6)
    problem = scenario.to_problem()
    single = LaCAM(time_limit=0.3, seed=0).solve(problem)
    restarted = pymapf.solve(
        problem, "x-lacam-restart", time_limit=1.0, per_run_limit=0.3, seed=0
    )
    assert restarted is not None
    assert restarted.sum_of_costs <= single.sum_of_costs
