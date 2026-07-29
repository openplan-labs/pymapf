"""Distribution control: steer the swarm's *shape*, not its individual goals.

Assigning a goal per agent does not scale -- the assignment problem grows, and
the plan has to be recomputed whenever an agent drops out. Distribution control
replaces "agent 47 goes here" with "the swarm should look like *this density*",
and lets each agent work out its own contribution from what it can see. Agents
become interchangeable: lose ten of them and the remaining ones re-spread.

Two controllers, both :class:`~pymapf.swarm.base.Behavior` subclasses -- so they
run in the same simulator, compose with flocking via
:class:`~pymapf.swarm.base.CompositeBehavior`, and obey the same acceleration
limits:

:class:`DensityMatching`
    Each agent estimates the swarm's *current* local density from its
    neighbours (a kernel estimate) and climbs the gradient of
    ``log target - log current``. Where the swarm is over-represented the second
    term dominates and pushes agents out; where it is under-represented the
    first pulls them in. The fixed point is the target density itself.

:class:`MixtureAssignment`
    For a Gaussian-mixture target, each agent picks the component it is most
    responsible for and flies to it, spreading inside it by mutual repulsion.
    Cruder than gradient matching, but it converges from far away and the
    assignment is inspectable, which matters when a human has to sign off on
    what the fleet is doing.

References
----------
* Bandyopadhyay, S.; Chung, S.-J.; and Hadaegh, F. Y. 2017. *Probabilistic and
  distributed control of a large-scale swarm of autonomous agents.* IEEE
  Transactions on Robotics 33(5): 1103-1123.
* Eren, U.; and Acikmese, B. 2017. *Velocity field generation for density
  control of swarms using heat equation and smoothing kernels.* IFAC 50(1).
* Krishnan, V.; and Martinez, S. 2018. *Distributed optimal transport for the
  deployment of swarms.* CDC 2018: 4583-4588.
* Bishop, C. M. 2006. *Pattern Recognition and Machine Learning*, ch. 9.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Behavior, SwarmState, register_behavior
from .density import (
    DensityField,
    GaussianMixtureDensity,
    balanced_assignment,
    get_density,
)

__all__ = ["DensityMatching", "MixtureAssignment"]


@register_behavior("density_matching")
class DensityMatching(Behavior):
    """Climb the gradient of ``log(target) - log(swarm)``.

    The swarm's own density is estimated locally with a Gaussian kernel over the
    agent's neighbours, so nothing global is needed. Both gradients are
    evaluated by central differences, which keeps the controller agnostic to how
    the target density is defined -- analytic mixture, fitted mixture,
    interpolated measurements, all work unchanged.

    Args:
        target: the density to match.
        bandwidth: kernel width for the swarm-density estimate. Too small and
            the estimate is spiky and the agents jitter; too large and distinct
            clusters blur into one.
        repulsion: short-range term that keeps agents apart while they spread.
            Density matching alone has no notion of a minimum separation --
            two agents at the same point are a perfectly good density estimate.
    """

    def __init__(
        self,
        target: DensityField,
        bandwidth: Optional[float] = None,
        gain: float = 6.0,
        repulsion: float = 4.0,
        damping: float = 1.2,
        epsilon: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target = get_density(target)
        self.bandwidth = bandwidth
        self.gain = gain
        self.repulsion = repulsion
        self.damping = damping
        self.epsilon = epsilon

    def _bandwidth(self) -> float:
        return (
            self.bandwidth
            if self.bandwidth is not None
            else self.params.reference_distance
        )

    def swarm_density(
        self, state: SwarmState, points: np.ndarray, index: int
    ) -> np.ndarray:
        """Kernel estimate of the swarm's density at ``points``, from neighbours."""
        neighbours = self.neighbors(state, index)
        sources = state.positions[np.append(neighbours, index).astype(int)]
        h = self._bandwidth()
        offsets = points[:, None, :] - sources[None, :, :]
        squared = np.sum(offsets**2, axis=2)
        kernel = np.exp(-squared / (2 * h**2))
        return kernel.sum(axis=1) / (
            len(sources) * (2 * np.pi * h**2) ** (state.dimension / 2)
        )

    def command(self, state: SwarmState, index: int) -> np.ndarray:
        position = state.positions[index]
        dimension = state.dimension
        delta = 0.25 * self._bandwidth()

        # Central differences of log(target) - log(swarm) around this agent.
        probes = np.repeat(position[None, :], 2 * dimension, axis=0)
        for axis in range(dimension):
            probes[2 * axis, axis] += delta
            probes[2 * axis + 1, axis] -= delta

        target_values = np.maximum(self.target(probes, state.time), self.epsilon)
        swarm_values = np.maximum(
            self.swarm_density(state, probes, index), self.epsilon
        )
        potential = np.log(target_values) - np.log(swarm_values)

        gradient = np.empty(dimension)
        for axis in range(dimension):
            gradient[axis] = (potential[2 * axis] - potential[2 * axis + 1]) / (
                2 * delta
            )

        command = self.gain * gradient
        command -= self.damping * state.velocities[index]

        # Keep agents physically apart; the density objective alone does not.
        for j in self.neighbors(state, index):
            offset = state.positions[j] - position
            distance = float(np.linalg.norm(offset))
            if 1e-9 < distance < self.params.separation_distance:
                command -= (
                    self.repulsion
                    * (self.params.separation_distance - distance)
                    / max(distance, 0.2)
                    * (offset / distance)
                )

        return self.finalise(command, state, index)


@register_behavior("mixture_assignment")
class MixtureAssignment(Behavior):
    """Fly to your component of a Gaussian mixture, then spread inside it.

    Assignment is by responsibility, damped by ``stickiness`` so an agent
    between two components does not oscillate. Within a component, agents
    repel each other and are attracted to the component mean with a strength
    that falls off inside one standard deviation -- so they fill the component
    rather than piling on its centre.

    Because the mixture's weights say how much of the swarm each component
    deserves, this also gives proportional allocation for free: a component with
    half the mass attracts about half the fleet.
    """

    def __init__(
        self,
        mixture: GaussianMixtureDensity,
        stickiness: float = 0.6,
        attraction: float = 2.0,
        spread: float = 3.0,
        damping: float = 1.4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not isinstance(mixture, GaussianMixtureDensity):
            raise TypeError("MixtureAssignment needs a GaussianMixtureDensity")
        self.mixture = mixture
        self.stickiness = stickiness
        self.attraction = attraction
        self.spread = spread
        self.damping = damping
        self._assignment: Optional[np.ndarray] = None

    def reset(self, state: SwarmState) -> None:
        self._assignment = None
        self.assign(state)

    def assign(self, state: SwarmState) -> np.ndarray:
        """Assign agents to components, in proportion to the mixing weights."""
        responsibilities = self.mixture.responsibilities(state.positions)
        quota = self.mixture.weights * state.n
        if self._assignment is None or len(self._assignment) != state.n:
            self._assignment = np.argmax(responsibilities, axis=1)
        self._assignment = balanced_assignment(
            responsibilities, quota, self._assignment, self.stickiness
        )
        return self._assignment

    def command(self, state: SwarmState, index: int) -> np.ndarray:
        if self._assignment is None or len(self._assignment) != state.n:
            self.assign(state)
        if index == 0:  # one agent re-runs the (cheap) assignment each step
            self.assign(state)

        component = int(self._assignment[index])
        mean = self.mixture.means[component]
        covariance = self.mixture.covariances[component]
        sigma = float(np.sqrt(np.mean(np.diag(covariance))))

        offset = mean - state.positions[index]
        distance = float(np.linalg.norm(offset))
        # Attraction saturates inside one sigma: fill the component, do not
        # collapse onto its mean.
        strength = self.attraction * np.tanh(
            max(0.0, distance - sigma) / max(sigma, 1e-6)
        )
        command = strength * offset / max(distance, 1e-9)
        command -= self.damping * state.velocities[index]

        for j in self.neighbors(state, index):
            difference = state.positions[j] - state.positions[index]
            gap = float(np.linalg.norm(difference))
            if 1e-9 < gap < self.params.reference_distance:
                command -= (
                    self.spread
                    * (self.params.reference_distance - gap)
                    / max(gap, 0.2)
                    * (difference / gap)
                )

        return self.finalise(command, state, index)
