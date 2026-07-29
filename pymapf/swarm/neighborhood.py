"""Who can each agent see? -- as a strategy object, not a hard-coded radius.

Every flocking law sums something over "neighbours", and which neighbours those
are changes the collective behaviour more than most of the gains do. Making the
choice a pluggable object means the same control law can be run with a metric
radius, a topological k-nearest rule, or a smooth kernel, and the difference
measured rather than argued about.

Four strategies, each with a reason to exist:

:class:`MetricNeighborhood`
    Everyone within ``sensing_range``. What a radio or a range sensor gives you.

:class:`TopologicalNeighborhood`
    The k nearest, whatever the distance. Starlings do this, and it is what
    keeps a real flock cohesive under compression: density changes do not change
    how many neighbours each bird tracks (Ballerini et al. 2008).

:class:`ConeNeighborhood`
    Metric, minus whatever falls outside a forward field of view or behind
    another agent. The honest model of a camera-based swarm.

:class:`GaussianKernelNeighborhood`
    Not a set but a *weighting*: influence decays smoothly with distance, so
    neighbours enter and leave without a step change in anyone's command. This
    is the arbitration mechanism of Manoni et al. (2022), and it is the natural
    partner for the Gaussian-mixture density control in
    :mod:`pymapf.swarm.distribution`.

References
----------
* Ballerini, M.; et al. 2008. *Interaction ruling animal collective behavior
  depends on topological rather than metric distance.* PNAS 105(4): 1232-1237.
* Manoni, T.; Albani, D.; et al. 2022. *Adaptive arbitration of aerial swarm
  interactions through a Gaussian kernel for coherent group motion.* Frontiers
  in Robotics and AI 9: 1006786.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .base import SwarmParams, SwarmState

__all__ = [
    "Neighborhood",
    "MetricNeighborhood",
    "TopologicalNeighborhood",
    "ConeNeighborhood",
    "GaussianKernelNeighborhood",
    "get_neighborhood",
    "NEIGHBORHOODS",
]


class Neighborhood(ABC):
    """Selects (and optionally weights) the agents one agent interacts with."""

    name = "abstract"

    @abstractmethod
    def of(self, state: SwarmState, index: int, params: SwarmParams) -> np.ndarray:
        """Indices of ``index``'s neighbours."""

    def weights(self, state: SwarmState, index: int, params: SwarmParams) -> np.ndarray:
        """Influence of each neighbour, aligned with :meth:`of`.

        Uniform by default; kernel neighbourhoods override it. Behaviors that
        multiply by these weights work unchanged under either.
        """
        return np.ones(len(self.of(state, index, params)))

    def degree(self, state: SwarmState, params: SwarmParams) -> np.ndarray:
        """Neighbour count per agent -- the connectivity of the swarm graph."""
        return np.array([len(self.of(state, i, params)) for i in range(state.n)])

    def __repr__(self) -> str:
        return "%s()" % type(self).__name__


class MetricNeighborhood(Neighborhood):
    """Everyone within ``params.sensing_range``."""

    name = "metric"

    def __init__(self, radius: Optional[float] = None):
        self.radius = radius

    def of(self, state: SwarmState, index: int, params: SwarmParams) -> np.ndarray:
        radius = self.radius if self.radius is not None else params.sensing_range
        return np.flatnonzero(state.distances_from(index) <= radius)

    def __repr__(self) -> str:
        return "MetricNeighborhood(radius=%s)" % self.radius


class TopologicalNeighborhood(Neighborhood):
    """The ``k`` nearest agents, regardless of how far away they are."""

    name = "topological"

    def __init__(self, k: int = 7):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k

    def of(self, state: SwarmState, index: int, params: SwarmParams) -> np.ndarray:
        distances = state.distances_from(index)
        k = min(self.k, state.n - 1)
        if k <= 0:
            return np.empty(0, dtype=int)
        return np.argsort(distances)[:k]

    def __repr__(self) -> str:
        return "TopologicalNeighborhood(k=%d)" % self.k


class ConeNeighborhood(Neighborhood):
    """Metric range restricted to a forward field of view.

    ``half_angle`` is measured from the agent's heading; an agent with no speed
    sees everything (it has no heading to be in front of).
    """

    name = "cone"

    def __init__(
        self, half_angle: float = np.pi * 0.75, radius: Optional[float] = None
    ):
        self.half_angle = float(half_angle)
        self.radius = radius

    def of(self, state: SwarmState, index: int, params: SwarmParams) -> np.ndarray:
        radius = self.radius if self.radius is not None else params.sensing_range
        candidates = np.flatnonzero(state.distances_from(index) <= radius)
        velocity = state.velocities[index]
        speed = float(np.linalg.norm(velocity))
        if speed < 1e-6 or not len(candidates):
            return candidates
        heading = velocity / speed
        offsets = state.positions[candidates] - state.positions[index]
        norms = np.linalg.norm(offsets, axis=1)
        norms[norms < 1e-9] = 1e-9
        cosines = (offsets @ heading) / norms
        return candidates[cosines >= np.cos(self.half_angle)]

    def __repr__(self) -> str:
        return "ConeNeighborhood(half_angle=%.2f)" % self.half_angle


class GaussianKernelNeighborhood(Neighborhood):
    """A smooth, distance-weighted neighbourhood (Manoni et al. 2022).

    Every agent inside ``cutoff`` is a neighbour, but its influence is
    ``exp(-d^2 / 2 sigma^2)``. Two consequences matter:

    * the command is continuous in the positions -- an agent drifting across
      the sensing boundary does not cause a step change in anyone's control;
    * ``sigma`` becomes a single knob arbitrating between local and global
      cohesion, which is what makes it adaptive: raise it and the swarm behaves
      as one group, lower it and it splits into locally coherent clusters.
    """

    name = "gaussian"

    def __init__(
        self,
        sigma: Optional[float] = None,
        cutoff_sigmas: float = 3.0,
        radius: Optional[float] = None,
    ):
        self.sigma = sigma
        self.cutoff_sigmas = cutoff_sigmas
        self.radius = radius

    def _sigma(self, params: SwarmParams) -> float:
        return self.sigma if self.sigma is not None else params.reference_distance

    def of(self, state: SwarmState, index: int, params: SwarmParams) -> np.ndarray:
        sigma = self._sigma(params)
        radius = self.radius
        if radius is None:
            radius = min(params.sensing_range, self.cutoff_sigmas * sigma)
        return np.flatnonzero(state.distances_from(index) <= radius)

    def weights(self, state: SwarmState, index: int, params: SwarmParams) -> np.ndarray:
        neighbours = self.of(state, index, params)
        if not len(neighbours):
            return np.empty(0)
        sigma = self._sigma(params)
        distances = state.distances_from(index)[neighbours]
        weights = np.exp(-(distances**2) / (2 * sigma**2))
        # Normalised to mean 1: the kernel decides *who matters more*, it does
        # not quietly scale the whole interaction down (which would let
        # self-propulsion outrun cohesion and disperse the flock).
        mean = float(weights.mean())
        return weights / mean if mean > 1e-12 else weights

    def __repr__(self) -> str:
        return "GaussianKernelNeighborhood(sigma=%s)" % self.sigma


NEIGHBORHOODS = {
    "metric": MetricNeighborhood,
    "topological": TopologicalNeighborhood,
    "cone": ConeNeighborhood,
    "gaussian": GaussianKernelNeighborhood,
}


def get_neighborhood(name, **kwargs) -> Neighborhood:
    if isinstance(name, Neighborhood):
        return name
    try:
        return NEIGHBORHOODS[str(name).lower()](**kwargs)
    except KeyError as error:
        raise ValueError(
            "Unknown neighborhood %r. Available: %s"
            % (name, ", ".join(sorted(NEIGHBORHOODS)))
        ) from error
