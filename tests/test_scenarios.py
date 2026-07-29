import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pytest

from pymapf.scenarios import (
    SCENARIO_BUILDERS,
    available_scenarios,
    build_scenario,
    from_ascii,
    to_ascii,
)


@pytest.mark.parametrize("name", sorted(SCENARIO_BUILDERS))
def test_every_builder_produces_a_valid_problem(name):
    scenario = build_scenario(name)
    problem = scenario.to_problem()  # validates starts/goals are free
    assert scenario.n_agents >= 1
    assert len(problem.agents) == scenario.n_agents
    for agent in scenario.agents:
        assert scenario.grid.is_free(agent.start)
        assert scenario.grid.is_free(agent.goal)


@pytest.mark.parametrize("name", sorted(SCENARIO_BUILDERS))
def test_builders_are_deterministic_for_a_seed(name):
    first = build_scenario(name, seed=7)
    second = build_scenario(name, seed=7)
    assert [(a.name, a.start, a.goal) for a in first.agents] == [
        (a.name, a.start, a.goal) for a in second.agents
    ]
    assert to_ascii(first) == to_ascii(second)


def test_different_seeds_give_different_instances():
    a = build_scenario("random_obstacles", seed=1)
    b = build_scenario("random_obstacles", seed=2)
    assert to_ascii(a) != to_ascii(b)


def test_agents_start_apart_from_their_goals():
    scenario = build_scenario("empty_room", n_agents=4, seed=3)
    for agent in scenario.agents:
        assert agent.start != agent.goal


def test_maze_is_fully_connected_between_agent_endpoints():
    from pymapf.algorithms import space_time_astar

    scenario = build_scenario("maze", seed=4)
    for agent in scenario.agents:
        assert space_time_astar(scenario.grid, agent.start, agent.goal) is not None


def test_unknown_scenario_lists_the_available_ones():
    with pytest.raises(ValueError) as error:
        build_scenario("teleporter")
    assert "empty_room" in str(error.value)


def test_available_scenarios_is_sorted_and_complete():
    assert available_scenarios() == sorted(SCENARIO_BUILDERS)


def test_random_obstacles_rejects_impossible_density():
    with pytest.raises(ValueError):
        build_scenario("random_obstacles", density=0.95)


def test_ascii_round_trip():
    text = "\n".join(
        [
            "######",
            "#a..A#",
            "#.##.#",
            "#B..b#",
            "######",
        ]
    )
    scenario = from_ascii(text)
    assert scenario.n_agents == 2
    names = {agent.name: agent for agent in scenario.agents}
    assert names["A"].start == (1, 1)
    assert names["A"].goal == (1, 4)
    assert names["B"].start == (3, 4)
    assert names["B"].goal == (3, 1)
    assert not scenario.grid.is_free((2, 2))
    assert to_ascii(scenario) == text


def test_ascii_requires_matching_start_and_goal():
    with pytest.raises(ValueError) as error:
        from_ascii("###\n#a#\n###")
    assert "A" in str(error.value)


def test_ascii_rejects_empty_input():
    with pytest.raises(ValueError):
        from_ascii("   \n\n")


def test_scenario_repr_mentions_size_and_agents():
    scenario = build_scenario("corner_swap", size=9, n_agents=4)
    assert "9x9" in repr(scenario)
    assert "agents=4" in repr(scenario)


def test_corner_swap_rejects_too_many_agents():
    with pytest.raises(ValueError):
        build_scenario("corner_swap", n_agents=99)


def test_sampling_fails_loudly_when_the_map_is_too_small():
    with pytest.raises(ValueError) as error:
        build_scenario("empty_room", height=4, width=4, n_agents=8)
    assert "reachable cells" in str(error.value)
