# -*- coding: utf-8 -*-

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from pymapf.decentralized.velocity_obstacle.velocity_agent import VelocityAgent
from pymapf.decentralized.position import Position
import numpy as np


def test_nmpc_agent():
    agent = VelocityAgent("toto", Position(2, 2), Position(4, 6), 15, 0.1)
    np.testing.assert_array_equal(agent.current_state, [2, 2])
    np.testing.assert_array_equal(agent.goal, [4, 6])
    assert agent.vmax > agent.vmin
