# -*- coding: utf-8 -*-

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from pymapf.decentralized.velocity_obstacle.velocity_obstacle import (
    MultiAgentVelocityObstacle,
)
from pymapf.decentralized.position import Position
import numpy as np


def test_mavo_agents():
    mavo = MultiAgentVelocityObstacle()
    mavo.register_agent("toto", Position(2, 2), Position(10, 5))
    mavo.register_agent("tata", Position(2, 5), Position(1, 5))
    mavo.register_agent("titi", Position(2, 9), Position(8, 5))
    assert len(mavo.agents) == 3


def test_mavo_obstacles():
    mavo = MultiAgentVelocityObstacle()
    mavo.register_obstacle(2, 3.14, Position(2, 3))
    assert len(mavo.obstacles_objects) == 1


def test_mavo_simulation_no_obs():
    mavo = MultiAgentVelocityObstacle()
    mavo.register_agent("toto", Position(2, 5), Position(4, 10))
    mavo.register_agent("tata", Position(4, 2), Position(4, 1))
    mavo.register_agent("titi", Position(10, 2), Position(0, 5))
    mavo.run_simulation()


def test_mavo_simulation_obstacles():
    mavo = MultiAgentVelocityObstacle()
    mavo.register_agent("tata", Position(2, 2), Position(4, 5))
    mavo.register_obstacle(2, 3.14, Position(2, 3))
    mavo.register_obstacle(2, -3.14, Position(2, 10))
    mavo.run_simulation()


def test_mavo_no_obs_vis(tmp_path):
    mavo = MultiAgentVelocityObstacle()
    mavo.register_agent("toto", Position(2, 5), Position(4, 10))
    mavo.register_agent("tata", Position(4, 2), Position(4, 1))
    mavo.register_agent("titi", Position(10, 2), Position(0, 5))
    mavo.run_simulation()
    mavo.visualize(str(tmp_path / "tata"), 10, 10)


def test_mavo_obs_vis(tmp_path):
    mavo = MultiAgentVelocityObstacle()
    mavo.register_agent("tata", Position(2, 2), Position(4, 5))
    mavo.register_obstacle(2, 3.14, Position(2, 3))
    mavo.register_obstacle(2, -3.14, Position(2, 10))
    mavo.run_simulation()
    mavo.visualize(str(tmp_path / "toto"), 10, 10)


def test_mavo_head_on_swap_keeps_separation():
    # Two agents swap places along the same line.  Each must see the other's
    # real position and velocity (not a velocity in the position slot), and
    # the first registered agent must not be blind to the second, or they
    # drive straight through each other (minimum distance used to be 0.06).
    mavo = MultiAgentVelocityObstacle(simulation_time=8.0, timestep=0.1)
    mavo.register_agent("A", Position(0, 5), Position(10, 5))
    mavo.register_agent("B", Position(10, 5), Position(0, 5))
    mavo.run_simulation()
    a = mavo.global_state_history["A"][:2]
    b = mavo.global_state_history["B"][:2]
    separation = np.linalg.norm(a - b, axis=0)
    radius_sum = mavo.agents["A"].radius + mavo.agents["B"].radius
    assert separation.min() >= radius_sum
    # ...and they still get where they were going.
    assert np.linalg.norm(a[:, -1] - [10, 5]) < 0.5
    assert np.linalg.norm(b[:, -1] - [0, 5]) < 0.5
