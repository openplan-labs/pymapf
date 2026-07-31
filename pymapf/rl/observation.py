"""What an agent sees, as swappable objects.

The observation is the single biggest design decision in learned MAPF, and it
is the one most worth being able to change without touching the environment.
So encoders are objects behind a registry, exactly like the solvers and the
swarm behaviors::

    from pymapf.rl import MAPFEnv

    MAPFEnv(scenario, observation="local")          # by name
    MAPFEnv(scenario, observation=LocalWindow(radius=6))

Two are provided.

``local``
    The PRIMAL-style egocentric stack (Sartoretti et al. 2019): a small window
    centred on the agent with one channel per thing worth knowing -- obstacles,
    other agents, this agent's goal -- plus a unit vector and normalised
    distance to the goal for when the goal lies outside the window. Partial
    observability is the point: it is what makes a *decentralized* policy, and
    what makes the comparison against a centralized planner interesting rather
    than rigged.

``global``
    The whole grid, every agent, every goal. Fully observable, mostly useful as
    a control: it tells you how much of a policy's failure is the observation
    and how much is the learning.

Both are fixed-size regardless of team size, which is what lets one set of
weights transfer across instances with different numbers of agents.

References
----------
* Sartoretti, G.; Kerr, J.; Shi, Y.; Wagner, G.; Kumar, T. K. S.; Koenig, S.;
  and Choset, H. 2019. *PRIMAL: Pathfinding via Reinforcement and Imitation
  Multi-Agent Learning.* IEEE Robotics and Automation Letters 4(3): 2378-2385.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Type

import numpy as np

from .spaces import Box

__all__ = [
    "ObservationEncoder",
    "LocalWindow",
    "GlobalGrid",
    "register_observation",
    "get_observation",
    "available_observations",
]


class ObservationEncoder(ABC):
    """Turns the environment's state into one array per agent."""

    name = "abstract"

    @abstractmethod
    def space(self, env) -> Box:
        """The observation space, given a bound environment."""

    @abstractmethod
    def encode(self, env, agent: str) -> np.ndarray:
        """This agent's observation of the current state."""

    def encode_all(self, env) -> Dict[str, np.ndarray]:
        return {agent: self.encode(env, agent) for agent in env.agents}

    def __repr__(self) -> str:
        return "%s()" % type(self).__name__


class LocalWindow(ObservationEncoder):
    """An egocentric stack of binary maps plus a goal bearing.

    Channels, in order:

    0. **obstacles** -- walls, and everything outside the grid, which is what
       stops an agent learning that the map edge is passable.
    1. **other agents** -- where the neighbours are now.
    2. **their goals** -- where the neighbours are trying to get to, which is
       what lets a policy yield to someone who needs the cell more.
    3. **own goal** -- set only when the goal falls inside the window.

    The trailing vector carries what the window cannot: a unit vector toward
    the goal and the normalised distance to it, so an agent whose goal is far
    away still knows which way to head. Without it a local policy wanders until
    the goal happens to scroll into view.

    Args:
        radius: window half-width. The window is ``(2 * radius + 1)`` square.
        include_goal_vector: append the bearing/distance block.
    """

    name = "local"
    channels = 4

    def __init__(self, radius: int = 4, include_goal_vector: bool = True):
        if radius < 1:
            raise ValueError("radius must be at least 1, got %d" % radius)
        self.radius = int(radius)
        self.include_goal_vector = include_goal_vector

    @property
    def size(self) -> int:
        return 2 * self.radius + 1

    @property
    def flat_size(self) -> int:
        return self.channels * self.size * self.size + (3 if self.include_goal_vector else 0)

    def space(self, env) -> Box:
        return Box(low=-1.0, high=1.0, shape=(self.flat_size,))

    def encode(self, env, agent: str) -> np.ndarray:
        grid = env.grid
        row, col = env.positions[agent]
        goal = env.goals[agent]
        size = self.size
        window = np.zeros((self.channels, size, size), dtype=np.float32)

        others = {
            position: name
            for name, position in env.positions.items()
            if name != agent
        }
        other_goals = {
            env.goals[name] for name in env.possible_agents if name != agent
        }

        for dr in range(-self.radius, self.radius + 1):
            for dc in range(-self.radius, self.radius + 1):
                cell = (row + dr, col + dc)
                r, c = dr + self.radius, dc + self.radius
                # Off-grid reads as obstacle: an agent must not learn that the
                # boundary is something it can walk through.
                if not grid.in_bounds(cell) or not grid.is_free(cell):
                    window[0, r, c] = 1.0
                    continue
                if cell in others:
                    window[1, r, c] = 1.0
                if cell in other_goals:
                    window[2, r, c] = 1.0
                if cell == goal:
                    window[3, r, c] = 1.0

        flat = window.reshape(-1)
        if not self.include_goal_vector:
            return flat

        delta_row = goal[0] - row
        delta_col = goal[1] - col
        distance = float(np.hypot(delta_row, delta_col))
        if distance > 1e-9:
            bearing = np.array([delta_row / distance, delta_col / distance], dtype=np.float32)
        else:
            bearing = np.zeros(2, dtype=np.float32)
        # Normalised by the grid diagonal so the scale is comparable across maps.
        diagonal = float(np.hypot(grid.height, grid.width)) or 1.0
        extra = np.array(
            [bearing[0], bearing[1], min(1.0, distance / diagonal)], dtype=np.float32
        )
        return np.concatenate([flat, extra])


class GlobalGrid(ObservationEncoder):
    """The whole map, every agent, every goal -- full observability.

    A control rather than a proposal: it does not scale and it is not
    decentralized, but when a policy fails it separates "could not see enough"
    from "could not learn it".
    """

    name = "global"
    channels = 4

    def space(self, env) -> Box:
        size = self.channels * env.grid.height * env.grid.width
        return Box(low=0.0, high=1.0, shape=(size,))

    def encode(self, env, agent: str) -> np.ndarray:
        grid = env.grid
        planes = np.zeros((self.channels, grid.height, grid.width), dtype=np.float32)
        for row in range(grid.height):
            for col in range(grid.width):
                if not grid.is_free((row, col)):
                    planes[0, row, col] = 1.0

        row, col = env.positions[agent]
        planes[1, row, col] = 1.0
        goal_row, goal_col = env.goals[agent]
        planes[2, goal_row, goal_col] = 1.0
        for name, position in env.positions.items():
            if name != agent:
                planes[3, position[0], position[1]] = 1.0
        return planes.reshape(-1)


ENCODERS: Dict[str, Type[ObservationEncoder]] = {
    "local": LocalWindow,
    "global": GlobalGrid,
}


def register_observation(name: str):
    """Class decorator registering an encoder under ``name``."""

    def decorator(cls: Type[ObservationEncoder]) -> Type[ObservationEncoder]:
        ENCODERS[name.lower()] = cls
        return cls

    return decorator


def available_observations() -> List[str]:
    return sorted(ENCODERS)


def get_observation(name, **kwargs) -> ObservationEncoder:
    """Resolve ``name`` (or an instance) to an :class:`ObservationEncoder`."""
    if isinstance(name, ObservationEncoder):
        return name
    try:
        factory = ENCODERS[str(name).lower()]
    except KeyError as error:
        raise ValueError(
            "Unknown observation %r. Available: %s"
            % (name, ", ".join(available_observations()))
        ) from error
    return factory(**kwargs)
