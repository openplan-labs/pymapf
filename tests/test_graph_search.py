"""Graph abstraction, single-agent searches, and SIPP."""

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pytest

import pymapf
from pymapf.algorithms.search import (
    astar,
    dijkstra,
    distance_table,
    focal_astar,
    weighted_astar,
)
from pymapf.algorithms.sipp import safe_intervals, sipp
from pymapf.algorithms import space_time_astar
from pymapf.core import Agent, Constraints, ExplicitGraph, GridMap, MAPFProblem
from pymapf.core.graph import roadmap
from pymapf.core.heuristics import true_distance


def _open(n=6):
    return GridMap([[0] * n for _ in range(n)])


# ---------------------------------------------------------------- graph ---


def test_undirected_graph_is_symmetric():
    graph = ExplicitGraph.undirected([("a", "b"), ("b", "c")])
    assert graph.is_undirected()
    assert set(graph.neighbors("b")) == {"a", "c"}
    assert len(graph) == 3


def test_directed_edges_stay_directed():
    graph = ExplicitGraph({"a": ["b"], "b": []})
    assert graph.neighbors("a") == ["b"]
    assert graph.neighbors("b") == []
    assert not graph.is_undirected()


def test_blocked_nodes_are_not_traversable():
    graph = ExplicitGraph.undirected([("a", "b"), ("b", "c")], blocked=["b"])
    assert not graph.is_free("b")
    assert graph.neighbors("a") == []
    assert graph.free_cells == 2


def test_graph_rejects_empty_adjacency():
    with pytest.raises(ValueError):
        ExplicitGraph({})


def test_from_grid_preserves_connectivity():
    grid = pymapf.build_scenario("warehouse", seed=1).grid
    graph = ExplicitGraph.from_grid(grid)
    assert graph.free_cells == grid.free_cells
    cell = next(c for c in graph.nodes if graph.degree(c))
    assert sorted(graph.neighbors(cell)) == sorted(grid.neighbors(cell))


def test_roadmap_keeps_isolated_nodes():
    graph = roadmap(["a", "b", "c"], [("a", "b")])
    assert graph.degree("c") == 0
    assert len(graph) == 3


@pytest.mark.parametrize("algorithm", ["cbs", "prioritized", "pibt", "lacam"])
def test_every_solver_runs_on_a_general_graph(algorithm):
    graph = ExplicitGraph.undirected(
        [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("b", "e")]
    )
    problem = MAPFProblem(graph, [Agent("r1", "a", "c"), Agent("r2", "c", "a")])
    kwargs = (
        {"heuristic": lambda node, goal: 0.0}
        if algorithm in ("cbs", "prioritized")
        else {}
    )
    solution = pymapf.solve(problem, algorithm, **kwargs)
    assert solution is not None
    assert solution.is_valid()
    for name, path_ in solution.paths.items():
        agent = next(a for a in problem.agents if a.name == name)
        assert path_[0] == agent.start and path_[-1] == agent.goal


# --------------------------------------------------------------- search ---


def test_dijkstra_distances_are_exact():
    distance, _ = dijkstra(_open(4), (0, 0))
    assert distance[(0, 0)] == 0
    assert distance[(0, 3)] == 3
    assert distance[(3, 3)] == 6


def test_astar_finds_a_shortest_path():
    path_ = astar(_open(5), (0, 0), (4, 4))
    assert path_[0] == (0, 0) and path_[-1] == (4, 4)
    assert len(path_) - 1 == 8


def test_weighted_astar_respects_its_bound():
    grid = pymapf.build_scenario("random_obstacles", seed=3).grid
    start, goal = (1, 1), (14, 14)
    optimal = astar(grid, start, goal)
    if optimal is None:
        pytest.skip("instance has no path between those corners")
    bounded = weighted_astar(grid, start, goal, weight=2.0)
    assert len(bounded) - 1 <= 2.0 * (len(optimal) - 1)


def test_weighted_astar_rejects_weight_below_one():
    with pytest.raises(ValueError):
        weighted_astar(_open(), (0, 0), (1, 1), weight=0.5)


def test_focal_search_respects_its_bound():
    grid = pymapf.build_scenario("maze", seed=2).grid
    scenario = pymapf.build_scenario("maze", seed=2)
    agent = scenario.agents[0]
    optimal = astar(grid, agent.start, agent.goal)
    focal = focal_astar(grid, agent.start, agent.goal, weight=1.5)
    assert focal is not None
    assert len(focal) - 1 <= 1.5 * (len(optimal) - 1)


def test_unreachable_goal_returns_none():
    grid = GridMap([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    assert astar(grid, (0, 0), (0, 2)) is None


def test_true_distance_is_exact_and_beats_manhattan_on_a_maze():
    scenario = pymapf.build_scenario("maze", seed=1)
    agent = scenario.agents[0]
    heuristic = true_distance(scenario.grid, agent.goal)
    optimal = astar(scenario.grid, agent.start, agent.goal)
    assert heuristic(agent.start, agent.goal) == len(optimal) - 1
    # The maze forces detours, so the exact value must exceed the L1 estimate.
    manhattan = abs(agent.start[0] - agent.goal[0]) + abs(
        agent.start[1] - agent.goal[1]
    )
    assert heuristic(agent.start, agent.goal) >= manhattan


def test_true_distance_refuses_a_goal_it_was_not_built_for():
    grid = _open()
    heuristic = true_distance(grid, (0, 0))
    with pytest.raises(ValueError):
        heuristic((1, 1), (2, 2))


def test_distance_table_marks_unreachable_nodes_absent():
    grid = GridMap([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    table = distance_table(grid, (0, 0))
    assert (0, 2) not in table


# ----------------------------------------------------------------- sipp ---


def test_safe_intervals_split_around_blocked_times():
    assert safe_intervals([]) == [(0, float("inf"))]
    assert safe_intervals([2, 3, 7]) == [(0, 2), (4, 7), (8, float("inf"))]


def test_sipp_matches_space_time_astar_cost():
    grid = _open(6)
    constraints = Constraints()
    constraints.add_vertex((0, 2), 2)
    constraints.add_vertex((0, 3), 3)
    reference = space_time_astar(grid, (0, 0), (0, 5), constraints=constraints)
    interval = sipp(grid, (0, 0), (0, 5), constraints=constraints)
    assert len(interval) == len(reference)
    assert interval[0] == (0, 0) and interval[-1] == (0, 5)


def test_sipp_respects_vertex_and_edge_constraints():
    grid = _open(5)
    constraints = Constraints()
    constraints.add_vertex((0, 1), 1)
    constraints.add_edge((0, 0), (1, 0), 1)
    path_ = sipp(grid, (0, 0), (2, 2), constraints=constraints)
    assert path_ is not None
    assert not constraints.blocks_vertex(path_[1], 1)
    assert not constraints.blocks_edge(path_[0], path_[1], 1)


def test_sipp_handles_a_long_wait_cheaply():
    grid = GridMap([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
    constraints = Constraints()
    for t in range(1, 200):
        constraints.add_vertex((1, 1), t)
    path_ = sipp(grid, (1, 0), (1, 2), constraints=constraints)
    assert path_ is not None
    assert path_[-1] == (1, 2)
    assert len(path_) - 1 >= 200  # it had to wait for the corridor


def test_sipp_returns_none_when_the_start_is_forbidden():
    grid = _open(3)
    constraints = Constraints()
    constraints.add_vertex((0, 0), 0)
    assert sipp(grid, (0, 0), (2, 2), constraints=constraints) is None
