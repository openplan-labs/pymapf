"""Running a swarm and measuring it.

The simulator is deliberately thin -- integrate, clamp, record -- because the
interesting content belongs in the behaviors. What it does own is the two
things that make comparisons trustworthy: identical initial conditions across
runs, and metrics that separate what the controller achieved from what the
initial condition handed it.

    from pymapf.swarm import SwarmSimulator

    sim = SwarmSimulator("acceleration", n_agents=20)
    result = sim.run(steps=300)
    print(result.metrics.summary())

Observers work the same way as :class:`~pymapf.core.trace.SearchTrace` on the
planning side: pass a callable and it receives every step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .base import Behavior, SwarmParams, SwarmState, get_behavior

__all__ = ["SwarmMetrics", "SwarmResult", "SwarmSimulator", "simulate"]


@dataclass
class SwarmMetrics:
    """What "good flocking" means, measured rather than eyeballed.

    Safety is reported twice on purpose. The steady-state minimum separation is
    what the controller *achieves*; the transient minimum includes start-up,
    where a dense spawn can put agents inside each other's separation distance
    before any control has acted. Reporting only the transient blames the
    controller for the initial condition; reporting only the steady state hides
    real start-up risk.
    """

    order: List[float] = field(default_factory=list)
    cohesion: List[float] = field(default_factory=list)
    min_distance: List[float] = field(default_factory=list)
    speed: List[float] = field(default_factory=list)
    connectivity: List[float] = field(default_factory=list)
    collisions: int = 0
    steady_collisions: int = 0

    def record(
        self,
        state: SwarmState,
        params: SwarmParams,
        steady: bool = False,
        degree: Optional[np.ndarray] = None,
    ) -> None:
        speeds = state.speeds
        total = float(np.linalg.norm(state.velocities.sum(axis=0)))
        denominator = float(speeds.sum())
        self.order.append(total / denominator if denominator > 1e-9 else 0.0)
        self.cohesion.append(
            float(np.mean(np.linalg.norm(state.positions - state.centroid, axis=1)))
        )
        self.speed.append(float(speeds.mean()))
        if degree is not None:
            self.connectivity.append(float(np.mean(degree)))

        if state.n > 1:
            deltas = state.positions[:, None, :] - state.positions[None, :, :]
            distances = np.linalg.norm(deltas, axis=2)
            np.fill_diagonal(distances, np.inf)
            closest = float(distances.min())
            self.min_distance.append(closest)
            if closest < params.separation_distance:
                self.collisions += 1
                if steady:
                    self.steady_collisions += 1

    def summary(self) -> Dict[str, float]:
        half = max(0, len(self.order) // 2)
        tail = slice(half, None)
        return {
            "order": float(np.mean(self.order[tail])) if self.order else 0.0,
            "cohesion": float(np.mean(self.cohesion[tail])) if self.cohesion else 0.0,
            "min_distance": (
                float(np.min(self.min_distance[half:])) if self.min_distance else 0.0
            ),
            "min_distance_transient": (
                float(np.min(self.min_distance)) if self.min_distance else 0.0
            ),
            "mean_speed": float(np.mean(self.speed[tail])) if self.speed else 0.0,
            "connectivity": (
                float(np.mean(self.connectivity[tail])) if self.connectivity else 0.0
            ),
            "collisions": self.collisions,
            "steady_collisions": self.steady_collisions,
        }


@dataclass
class SwarmResult:
    """Trajectory plus metrics -- everything :mod:`pymapf.viz` needs."""

    history: List[SwarmState]
    metrics: SwarmMetrics
    behavior: str
    params: SwarmParams

    @property
    def final(self) -> SwarmState:
        return self.history[-1]

    def positions(self) -> np.ndarray:
        """``(steps, n, d)`` array of the whole run."""
        return np.stack([state.positions for state in self.history])


class SwarmSimulator:
    """Integrates a :class:`~pymapf.swarm.base.Behavior` forward in time.

    Args:
        behavior: name, class or instance. Names come from
            :func:`~pymapf.swarm.base.available_behaviors`.
        params: shared limits/gains; the behavior's own knobs stay in its
            constructor.
        dt: integration step. Semi-implicit Euler (velocity first, then
            position) -- the integrator a flight controller effectively runs,
            and it does not inject the energy explicit Euler does.
    """

    def __init__(
        self,
        behavior="acceleration",
        n_agents: int = 20,
        dimension: int = 2,
        params: Optional[SwarmParams] = None,
        dt: float = 0.1,
        initial: Optional[SwarmState] = None,
        spacing: Optional[float] = None,
        **behavior_kwargs,
    ):
        self.params = params or SwarmParams()
        self.behavior: Behavior = get_behavior(
            behavior, params=self.params, **behavior_kwargs
        )
        self.behavior.params = self.params
        self.dt = dt
        self.n_agents = n_agents
        self.dimension = dimension
        self.rng = np.random.default_rng(self.params.seed)
        self.initial = initial
        self.spacing = spacing or max(
            self.params.reference_distance, self.params.separation_distance * 1.2
        )

    def initial_state(self) -> SwarmState:
        if self.initial is not None:
            return self.initial.copy()
        return SwarmState.lattice(
            self.n_agents,
            self.dimension,
            spacing=self.spacing,
            rng=np.random.default_rng(self.params.seed),
        )

    def step(self, state: SwarmState) -> SwarmState:
        """One integration step. Public so a caller can drive the loop itself."""
        commands = self.behavior.commands(state)
        if self.params.noise:
            commands = commands + self.rng.normal(0, self.params.noise, size=commands.shape)

        nxt = state.copy()
        if getattr(self.behavior, "output", "acceleration") == "velocity":
            nxt.velocities = commands
        else:
            nxt.velocities = state.velocities + commands * self.dt

        speeds = np.linalg.norm(nxt.velocities, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.params.max_speed / np.maximum(speeds, 1e-9))
        nxt.velocities *= scale
        nxt.positions = state.positions + nxt.velocities * self.dt
        nxt.time = state.time + self.dt

        if self.params.bounds is not None:
            half = len(self.params.bounds) // 2
            low = np.asarray(self.params.bounds[:half], dtype=float)
            high = np.asarray(self.params.bounds[half:], dtype=float)
            # Reflect rather than clip: clipping silently zeroes the velocity
            # and makes a wall look like a successful stop.
            below = nxt.positions < low
            above = nxt.positions > high
            nxt.positions = np.where(below, 2 * low - nxt.positions, nxt.positions)
            nxt.positions = np.where(above, 2 * high - nxt.positions, nxt.positions)
            nxt.velocities = np.where(below | above, -nxt.velocities, nxt.velocities)
        return nxt

    def run(
        self,
        steps: int = 300,
        observer: Optional[Callable[[int, SwarmState], None]] = None,
    ) -> SwarmResult:
        state = self.initial_state()
        self.behavior.reset(state)

        history = [state.copy()]
        metrics = SwarmMetrics()
        degree = self.behavior.neighborhood.degree(state, self.params)
        metrics.record(state, self.params, steady=False, degree=degree)
        if observer:
            observer(0, state)

        for step_index in range(steps):
            state = self.step(state)
            history.append(state.copy())
            metrics.record(
                state,
                self.params,
                steady=step_index > steps // 2,
                degree=self.behavior.neighborhood.degree(state, self.params),
            )
            if observer:
                observer(step_index + 1, state)

        return SwarmResult(history, metrics, self.behavior.name, self.params)


def simulate(behavior="acceleration", n_agents: int = 20, steps: int = 300, **kwargs):
    """One-call convenience wrapper around :class:`SwarmSimulator`."""
    dimension = kwargs.pop("dimension", 2)
    params = kwargs.pop("params", None)
    dt = kwargs.pop("dt", 0.1)
    initial = kwargs.pop("initial", None)
    simulator = SwarmSimulator(
        behavior,
        n_agents=n_agents,
        dimension=dimension,
        params=params,
        dt=dt,
        initial=initial,
        **kwargs,
    )
    return simulator.run(steps=steps)
