# -*- coding: utf-8 -*-

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from pymapf.decentralized.nmpc.nmpc_agent import NMPCAgent
from pymapf.decentralized.position import Position
import numpy as np


def test_nmpc_agent():
    agent = NMPCAgent("toto", Position(2, 2), Position(4, 10), 15, 0.3, 0.1)
    np.testing.assert_array_equal(agent.current_state, [2, 2])
    np.testing.assert_array_equal(agent.goal, [4, 10])
    assert agent.vmax > agent.vmin


def _obstacle_stack(states):
    # The manager's layout: (4, timesteps, n_obstacles) of [x, y, vx, vy].
    return np.stack(states, axis=2)


def test_nmpc_agent_predicts_obstacles_at_the_current_step():
    timesteps = 5
    # Three obstacles (the old code crashed on any count but 1 and 2), each
    # moving at a constant (1, 0) from a different row.
    states = []
    for k in range(3):
        xs = np.arange(timesteps, dtype=float)
        states.append(
            np.stack(
                [
                    xs,
                    np.full(timesteps, 2.0 + k),
                    np.ones(timesteps),
                    np.zeros(timesteps),
                ]
            )
        )
    obstacles = _obstacle_stack(states)
    agent = NMPCAgent(
        "a", Position(0, 0), Position(5, 0), timesteps, 0.3, 0.1, horizon_length=4
    )
    step = 3
    predictions = agent._NMPCAgent__predict_obstacle_positions(obstacles, step, [])
    assert len(predictions) == 3
    for k, prediction in enumerate(predictions):
        # Starts from the obstacle's position *at this step* (x = 3) and moves
        # at 1 m/s for nmpc_timestep = 0.3 s per horizon step.
        expected = np.array([[3.0 + 0.3 * (h + 1), 2.0 + k] for h in range(4)])
        np.testing.assert_allclose(prediction.reshape(4, 2), expected)
    agent.simulate_step(step, obstacles, [])  # used to raise a broadcast error


def test_nmpc_agent_hash_and_reproducibility():
    a = NMPCAgent("a", Position(0, 0), Position(5, 0), 3, 0.3, 0.1)
    b = NMPCAgent("a", Position(0, 0), Position(5, 0), 3, 0.3, 0.1)
    assert isinstance(hash(a), int)
    assert len({a, b}) == 1  # equal agents (same ident) hash equal
    _, vel_a, _ = a.simulate_step(0, [], [np.array([1.0, 0.0, -1.0, 0.0])])
    _, vel_b, _ = b.simulate_step(0, [], [np.array([1.0, 0.0, -1.0, 0.0])])
    np.testing.assert_array_equal(vel_a, vel_b)
