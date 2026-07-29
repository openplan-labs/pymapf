import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pymapf
from pymapf.core import Agent, GridMap, MAPFProblem, SearchTrace
from pymapf.core.solver import count_conflicts
from pymapf.core.trace import SearchEvent


def _crossing_problem():
    # Both shortest paths run through the centre cell at t=1.
    grid = GridMap([[0] * 3 for _ in range(3)])
    return MAPFProblem(grid, [Agent("a", (0, 1), (2, 1)), Agent("b", (1, 0), (1, 2))])


def test_solving_without_an_observer_still_works():
    assert pymapf.solve(_crossing_problem(), "cbs") is not None


def test_trace_records_the_search_vocabulary():
    trace = SearchTrace()
    solution = pymapf.solve(_crossing_problem(), "cbs", observer=trace)

    kinds = {event.kind for event in trace}
    assert {"agent_planned", "root", "expand", "conflict", "solved"} <= kinds
    assert trace.expansions >= 1
    assert len(trace.conflicts) >= 1
    assert trace.summary()["solved"] is True
    assert trace.summary()["cost"] == solution.sum_of_costs


def test_expand_events_carry_the_plan_under_consideration():
    trace = SearchTrace()
    pymapf.solve(_crossing_problem(), "cbs", observer=trace)
    expand = trace.of_kind("expand")[0]
    assert set(expand["paths"]) == {"a", "b"}
    assert expand["paths"]["a"][0] == (0, 1)


def test_cost_curve_is_non_decreasing_for_cbs():
    scenario = pymapf.build_scenario("corner_swap", n_agents=4)
    trace = SearchTrace()
    pymapf.solve(scenario.to_problem(), "cbs", observer=trace)
    curve = trace.cost_curve()
    assert curve == sorted(curve)


def test_prioritized_planning_emits_one_event_per_agent():
    scenario = pymapf.build_scenario("empty_room", n_agents=3, seed=1)
    trace = SearchTrace()
    pymapf.solve(scenario.to_problem(), "prioritized", observer=trace)
    assert len(trace.of_kind("agent_planned")) == 3
    assert trace.of_kind("solved")


def test_failure_is_reported_with_a_reason():
    # A corridor of length 1 with two agents that must swap has no solution.
    grid = GridMap([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
    problem = MAPFProblem(grid, [Agent("a", (1, 0), (1, 2)), Agent("b", (1, 2), (1, 0))])
    trace = SearchTrace()
    assert pymapf.solve(problem, "cbs", observer=trace, max_expansions=200) is None
    failures = trace.of_kind("failed")
    assert failures and failures[-1]["reason"]


def test_max_events_caps_memory():
    scenario = pymapf.build_scenario("corner_swap", n_agents=4)
    trace = SearchTrace(max_events=5)
    pymapf.solve(scenario.to_problem(), "cbs", observer=trace)
    assert len(trace) == 5


def test_a_plain_callable_is_a_valid_observer():
    seen = []
    pymapf.solve(_crossing_problem(), "cbs", observer=seen.append)
    assert all(isinstance(event, SearchEvent) for event in seen)
    assert seen[0].step == 1
    assert seen[-1].elapsed >= seen[0].elapsed


def test_event_supports_item_and_get_access():
    event = SearchEvent("expand", 1, {"cost": 4})
    assert event["cost"] == 4
    assert event.get("missing", "fallback") == "fallback"


def test_count_conflicts_counts_every_pair():
    # Three agents all sitting on the same cell at t=0: three pairs collide.
    paths = {"a": [(0, 0)], "b": [(0, 0)], "c": [(0, 0)]}
    assert count_conflicts(paths) == 3
    assert count_conflicts({"a": [(0, 0)], "b": [(1, 1)]}) == 0
