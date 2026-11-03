from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from pymapf.centralized.cooperative_astar.cooperative_astar import CooperativeAStar
from pymapf.centralized.world import World
from pymapf.decentralized.position import Position
import numpy as np
import pytest
import random
import time


def test_cooperative_astar_sim():
    random.seed(0)
    w = World(12, 12, 0.1)
    campf = CooperativeAStar(w)
    agents_labels = ["A", "B", "C", "D", "E"]
    for label in agents_labels:
        start, goal = w.get_start_goal(random.random() * 0.5)
        campf.register_agent(label, start, goal)

    campf.run_simulation()


def test_cooperative_astar_sim_diagonals(tmp_path):
    random.seed(1)
    w = World(12, 12, 0.1, allow_diagonals=True)
    campf = CooperativeAStar(w)
    agents_labels = ["A", "B", "C", "D", "E"]
    for label in agents_labels:
        start, goal = w.get_start_goal(random.random() * 0.5)
        campf.register_agent(label, start, goal)

    campf.run_simulation()
    campf.visualize(str(tmp_path / "test"))


def test_cooperative_astar_viz(tmp_path):
    random.seed(2)
    w = World(12, 12, 0.1)
    campf = CooperativeAStar(w)
    agents_labels = ["A", "B", "C", "D", "E"]
    for label in agents_labels:
        start, goal = w.get_start_goal(random.random() * 0.5)
        campf.register_agent(label, start, goal)

    campf.run_simulation()
    campf.visualize(str(tmp_path / "test"))


def _cell(path, t):
    # Agents stay where their path ends.
    state = path[min(t, len(path) - 1)]
    return state.x, state.y


@pytest.mark.parametrize("allow_diagonals", [False, True])
def test_cooperative_astar_plans_are_conflict_free(allow_diagonals):
    # Used to return vertex conflicts (shared starts, an earlier agent
    # driving through a later agent's parked goal) and swap conflicts (no
    # edge check at all) on most of these seeded instances.
    for seed in range(20):
        random.seed(seed)
        w = World(10, 10, 0.1, allow_diagonals=allow_diagonals)
        campf = CooperativeAStar(w)
        for label in "ABCDEF":
            start, goal = w.get_start_goal(0.3)
            campf.register_agent(label, start, goal)
        campf.run_simulation()

        paths = campf.paths
        names = list(paths)
        assert len({campf.agents[n].init_pos for n in names}) == len(names)
        assert len({campf.agents[n].goal_pos for n in names}) == len(names)
        horizon = max(len(p) for p in paths.values()) + 1
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                for t in range(horizon):
                    assert _cell(paths[a], t) != _cell(paths[b], t), (seed, a, b, t)
                    if t:
                        swapped = _cell(paths[a], t) == _cell(
                            paths[b], t - 1
                        ) and _cell(paths[b], t) == _cell(paths[a], t - 1)
                        moved = _cell(paths[a], t) != _cell(paths[a], t - 1)
                        assert not (swapped and moved), (seed, a, b, t)
