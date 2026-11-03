# -*- coding: utf-8 -*-

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from pymapf.decentralized.nmpc.nmpc import MultiAgentNMPC
from pymapf.decentralized.position import Position


def test_nmpc_agents():
    nmpc = MultiAgentNMPC()
    nmpc.register_agent("toto", Position(1, 2), Position(4, 5))
    nmpc.register_agent("tata", Position(2, 2), Position(7, 5))
    nmpc.register_agent("titi", Position(2, 3), Position(4, 8))
    assert len(nmpc.agents) == 3


def test_nmpc_obstacles():
    nmpc = MultiAgentNMPC()
    nmpc.register_obstacle(2, 3.14, Position(2, 3))
    assert len(nmpc.obstacles_objects) == 1


def test_sim_no_obstacles():
    nmpc = MultiAgentNMPC()
    nmpc.register_agent("toto", Position(1, 2), Position(4, 5))
    nmpc.register_agent("tata", Position(2, 2), Position(7, 5))
    nmpc.register_agent("titi", Position(2, 3), Position(4, 8))
    nmpc.run_simulation()


def test_sim_obstacles():
    nmpc = MultiAgentNMPC()
    nmpc.register_agent("toto", Position(1, 2), Position(4, 5))
    nmpc.register_agent("tata", Position(2, 2), Position(7, 5))
    nmpc.register_agent("titi", Position(2, 3), Position(4, 8))
    nmpc.register_obstacle(2, 3.14, Position(2, 3))
    nmpc.register_obstacle(2, -3.14, Position(10, 7))
    nmpc.run_simulation()


def test_visualize_no_obs(tmp_path):
    nmpc = MultiAgentNMPC()
    nmpc.register_agent("toto", Position(1, 2), Position(4, 5))
    nmpc.register_agent("tata", Position(2, 2), Position(7, 5))
    nmpc.register_agent("titi", Position(2, 3), Position(4, 8))
    nmpc.run_simulation()
    nmpc.visualize(str(tmp_path / "toto"), 10, 10)


def test_visualize_obs(tmp_path):
    nmpc = MultiAgentNMPC()
    nmpc.register_agent("toto", Position(1, 2), Position(4, 5))
    nmpc.register_agent("tata", Position(2, 2), Position(7, 5))
    nmpc.register_agent("titi", Position(2, 3), Position(4, 8))
    nmpc.register_obstacle(2, 3.14, Position(2, 3))
    nmpc.register_obstacle(2, -3.14, Position(10, 7))
    nmpc.run_simulation()
    nmpc.visualize(str(tmp_path / "toto"), 10, 10)


def test_every_agent_sees_the_others(monkeypatch):
    # The first registered agent used to plan against an empty list: the list
    # of "other agents" was rebuilt from scratch inside every step.
    from pymapf.decentralized.nmpc.nmpc_agent import NMPCAgent

    seen = {}
    original = NMPCAgent.simulate_step

    def spy(self, step, obstacles, other_agents):
        seen.setdefault(self.ident, []).append(len(other_agents))
        return original(self, step, obstacles, other_agents)

    monkeypatch.setattr(NMPCAgent, "simulate_step", spy)
    nmpc = MultiAgentNMPC(simulation_time=0.3)
    nmpc.register_agent("first", Position(0, 5), Position(6, 5))
    nmpc.register_agent("second", Position(6, 5), Position(0, 5))
    nmpc.run_simulation()
    steps = nmpc.number_of_timesteps
    assert seen == {"first": [1] * steps, "second": [1] * steps}
