"""The vocabulary every decentralized swarm algorithm in this package shares.

The centralized side of this library is built on a small set of stable types --
``MAPFProblem``, ``Solution``, ``MAPFSolver``, a name registry -- and swapping
one algorithm for another is a string change. This module gives the swarm side
the same shape:

* :class:`SwarmState` is the input (positions and velocities of everyone);
* a :class:`Behavior` maps ``(state, index) -> command`` for one agent;
* :class:`CompositeBehavior` sums weighted behaviors, so new controllers are
  *assembled* rather than rewritten;
* :func:`register_behavior` / :func:`get_behavior` make them addressable by
  name, exactly like the solvers.

The unit of composition is deliberately a single agent's command. That is what
"decentralized" means: an agent computes its own action from what it can see,
and no object in this package is allowed to close over the whole swarm to make
one agent's decision. :class:`Neighborhood` enforces the "what it can see"
half.

    from pymapf.swarm import get_behavior, SwarmState

    behavior = get_behavior("acceleration", cruise_speed=5.0)
    command = behavior.command(state, 0)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

__all__ = [
    "SwarmState",
    "SwarmParams",
    "Behavior",
    "CompositeBehavior",
    "register_behavior",
    "get_behavior",
    "available_behaviors",
    "limit",
]


# --------------------------------------------------------------------------
# state and parameters
# --------------------------------------------------------------------------


@dataclass
class SwarmState:
    """Positions and velocities of the whole swarm at one instant.

    Arrays are ``(n, d)``; ``d`` is 2 or 3 and every algorithm here is written
    dimension-agnostically, so the same controller flies a planar formation and
    a 3D one.
    """

    positions: np.ndarray
    velocities: np.ndarray
    time: float = 0.0

    def __post_init__(self):
        self.positions = np.atleast_2d(np.asarray(self.positions, dtype=float))
        self.velocities = np.atleast_2d(np.asarray(self.velocities, dtype=float))
        if self.positions.shape != self.velocities.shape:
            raise ValueError(
                "positions %s and velocities %s must have the same shape"
                % (self.positions.shape, self.velocities.shape)
            )

    @property
    def n(self) -> int:
        return self.positions.shape[0]

    @property
    def dimension(self) -> int:
        return self.positions.shape[1]

    @property
    def centroid(self) -> np.ndarray:
        return self.positions.mean(axis=0)

    @property
    def speeds(self) -> np.ndarray:
        return np.linalg.norm(self.velocities, axis=1)

    def offsets_from(self, index: int) -> np.ndarray:
        """Vectors from agent ``index`` to everyone else (zero for itself)."""
        return self.positions - self.positions[index]

    def distances_from(self, index: int) -> np.ndarray:
        distances = np.linalg.norm(self.offsets_from(index), axis=1)
        distances[index] = np.inf  # an agent is never its own neighbour
        return distances

    def copy(self) -> "SwarmState":
        return SwarmState(self.positions.copy(), self.velocities.copy(), self.time)

    @classmethod
    def lattice(
        cls,
        n: int,
        dimension: int = 2,
        spacing: float = 3.0,
        jitter: float = 0.15,
        rng: Optional[np.random.Generator] = None,
    ) -> "SwarmState":
        """A legal starting formation: a jittered lattice at ``spacing``.

        A uniform random spawn puts some pairs closer than any sensible
        separation distance before the controller has run a single step, so
        every safety metric then measures the spawn instead of the control law.
        """
        rng = rng or np.random.default_rng(0)
        per_side = int(math.ceil(n ** (1.0 / dimension)))
        coordinates = np.indices((per_side,) * dimension).reshape(dimension, -1).T[:n]
        positions = coordinates.astype(float) * spacing
        positions -= positions.mean(axis=0)
        positions += rng.normal(0, spacing * jitter, size=positions.shape)
        return cls(positions, rng.normal(0, 0.5, size=(n, dimension)))


@dataclass
class SwarmParams:
    """Shared physical limits and gains.

    Behaviors read what they need and ignore the rest; sharing one object is
    what lets two controllers be compared on identical conditions. Behaviors
    that need their own knob declare it as a constructor argument instead of
    growing this class.
    """

    # geometry (metres)
    separation_distance: float = 1.5
    reference_distance: float = 3.0
    sensing_range: float = 6.0

    # limits
    max_speed: float = 8.0
    max_acceleration: float = 6.0
    cruise_speed: float = 4.0

    # environment
    obstacles: Sequence[Tuple[Sequence[float], float]] = ()  # (centre, radius)
    obstacle_gain: float = 12.0
    bounds: Optional[Tuple[float, ...]] = None  # (x0, y0, ..., x1, y1, ...)

    # waypoint
    migration_point: Optional[Sequence[float]] = None
    migration_gain: float = 0.8
    # Share of the acceleration budget the waypoint may claim. Bounded on
    # purpose: an unsaturated `gain * (target - position)` grows with distance
    # and starves collision avoidance once the final clamp kicks in.
    migration_authority: float = 0.35

    noise: float = 0.0
    seed: int = 0

    def replace(self, **changes) -> "SwarmParams":
        return replace(self, **changes)


def limit(vector: np.ndarray, magnitude: float) -> np.ndarray:
    """Clamp a vector's norm, preserving direction."""
    norm = float(np.linalg.norm(vector))
    if norm > magnitude > 0:
        return vector * (magnitude / norm)
    return vector


# --------------------------------------------------------------------------
# behaviors
# --------------------------------------------------------------------------


class Behavior(ABC):
    """One decentralized control law.

    Subclasses implement :meth:`command`, which returns the acceleration for a
    *single* agent given the swarm state. Anything a behavior needs beyond the
    shared :class:`SwarmParams` goes in its constructor, so a behavior is a
    configured object rather than a function with a parameter bag.
    """

    name = "abstract"
    #: What the returned command means: an acceleration, or a velocity the
    #: integrator should track. Mixed swarms are integrated correctly because
    #: the simulator asks.
    output = "acceleration"

    def __init__(self, params: Optional[SwarmParams] = None, neighborhood=None):
        from .neighborhood import MetricNeighborhood  # local: avoids a cycle

        self.params = params or SwarmParams()
        self.neighborhood = neighborhood or MetricNeighborhood()

    # -- lifecycle ----------------------------------------------------------
    def reset(self, state: SwarmState) -> None:
        """Hook for behaviors with memory (integral terms, filters, roles)."""

    def neighbors(self, state: SwarmState, index: int) -> np.ndarray:
        return self.neighborhood.of(state, index, self.params)

    # -- the control law ----------------------------------------------------
    @abstractmethod
    def command(self, state: SwarmState, index: int) -> np.ndarray:
        """Return this agent's command. Must not modify ``state``."""

    def commands(self, state: SwarmState) -> np.ndarray:
        """Every agent's command. Vectorised subclasses override this."""
        return np.array([self.command(state, i) for i in range(state.n)])

    # -- shared terms every controller ends up needing ----------------------
    def migration(self, state: SwarmState, index: int) -> np.ndarray:
        params = self.params
        if params.migration_point is None:
            return np.zeros(state.dimension)
        target = np.asarray(params.migration_point, dtype=float)
        return limit(
            params.migration_gain * (target - state.positions[index]),
            params.migration_authority * params.max_acceleration,
        )

    def obstacle_avoidance(self, state: SwarmState, index: int) -> np.ndarray:
        params = self.params
        push = np.zeros(state.dimension)
        for centre, radius in params.obstacles:
            offset = state.positions[index] - np.asarray(centre, dtype=float)
            distance = float(np.linalg.norm(offset))
            margin = distance - radius
            if margin < params.sensing_range and distance > 1e-9:
                push += params.obstacle_gain * offset / (distance * max(margin, 0.2) ** 2)
        return push

    def finalise(self, command: np.ndarray, state: SwarmState, index: int) -> np.ndarray:
        """Add the shared terms and clamp. Every behavior ends with this."""
        command = command + self.migration(state, index) + self.obstacle_avoidance(state, index)
        return limit(command, self.params.max_acceleration)

    def __repr__(self) -> str:
        return "%s(name=%r, neighborhood=%r)" % (
            type(self).__name__,
            self.name,
            self.neighborhood,
        )


class CompositeBehavior(Behavior):
    """A weighted sum of other behaviors.

    This is the practical payoff of making behaviors objects: "Vicsek alignment
    plus the acceleration model's spacing" is a composition, not a new class, and
    the parts stay independently testable.

        blend = CompositeBehavior([(Vicsek(), 0.5), (AccelerationFlocking(), 1.0)])
    """

    name = "composite"

    def __init__(
        self,
        parts: Sequence[Tuple[Behavior, float]],
        params: Optional[SwarmParams] = None,
        neighborhood=None,
    ):
        super().__init__(params=params, neighborhood=neighborhood)
        if not parts:
            raise ValueError("a composite needs at least one behavior")
        self.parts = list(parts)
        for behavior, _ in self.parts:
            # One shared parameter object, so limits cannot disagree.
            behavior.params = self.params

    def reset(self, state: SwarmState) -> None:
        for behavior, _ in self.parts:
            behavior.reset(state)

    def command(self, state: SwarmState, index: int) -> np.ndarray:
        total = np.zeros(state.dimension)
        for behavior, weight in self.parts:
            total = total + weight * behavior.command(state, index)
        return limit(total, self.params.max_acceleration)

    def __repr__(self) -> str:
        return "CompositeBehavior(%s)" % ", ".join(
            "%s*%.2g" % (b.name, w) for b, w in self.parts
        )


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

_BEHAVIORS: Dict[str, Type[Behavior]] = {}


def register_behavior(name: str):
    """Class decorator registering a behavior under ``name``."""

    def decorator(cls: Type[Behavior]) -> Type[Behavior]:
        key = name.lower()
        if key in _BEHAVIORS:
            raise ValueError("behavior %r already registered" % name)
        cls.name = key
        _BEHAVIORS[key] = cls
        return cls

    return decorator


def available_behaviors() -> List[str]:
    return sorted(_BEHAVIORS)


def get_behavior(name, **kwargs) -> Behavior:
    """Instantiate a registered behavior by name.

    Also accepts a :class:`Behavior` instance or a class, so callers can pass
    "whatever the user configured" without branching.
    """
    if isinstance(name, Behavior):
        return name
    if isinstance(name, type) and issubclass(name, Behavior):
        return name(**kwargs)
    try:
        cls = _BEHAVIORS[str(name).lower()]
    except KeyError as error:
        raise ValueError(
            "Unknown behavior %r. Available: %s"
            % (name, ", ".join(available_behaviors()))
        ) from error
    return cls(**kwargs)
