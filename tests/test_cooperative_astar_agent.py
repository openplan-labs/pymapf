from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from pymapf.centralized.cooperative_astar.agent import Agent
from pymapf.centralized.world import World
from pymapf.decentralized.position import Position
import numpy as np
import random
import time


def test_agent():
    a = Agent("h", (2, 2), (3, 3), allow_diagonals=True)
    assert a.init_pos == (2, 2) and a.goal_pos == (3, 3)


def test_agent_detects_swap_and_crossing_conflicts():
    from pymapf.centralized.cooperative_astar.state import State

    # The other agent moves (1, 0) -> (0, 0) between t = 0 and t = 1.
    other = {"o": [State(1, 0, 0), State(0, 0, 1)]}
    a = Agent("a", (0, 0), (0, 1))
    # Swapping cells with it is a conflict, even though no cell is shared at
    # the same instant.
    assert a.in_conflict(State(0, 0, 0), State(1, 0, 1), other)
    # Once its path ends it stays at (0, 0) for good.
    assert a.in_conflict(State(0, 1, 5), State(0, 0, 6), other)
    assert not a.in_conflict(State(0, 1, 5), State(1, 1, 6), other)

    # Diagonals: (0, 1) -> (1, 0) crosses the other's (0, 0) -> (1, 1).
    crossing = {"o": [State(0, 0, 0), State(1, 1, 1)]}
    d = Agent("d", (1, 0), (0, 1), allow_diagonals=True)
    assert d.in_conflict(State(0, 1, 0), State(1, 0, 1), crossing)
    # A goal is only usable if nobody planned earlier passes through it later.
    assert not d.can_stay(State(1, 1, 0), crossing)
    assert d.can_stay(State(1, 0, 0), crossing)
