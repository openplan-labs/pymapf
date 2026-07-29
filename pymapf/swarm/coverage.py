"""Coverage controllers: deploy a team to watch a region.

Each controller is an object with one required method -- ``step(positions)`` --
and it is written against a :class:`~pymapf.swarm.domain.Domain` and a
:class:`~pymapf.swarm.density.DensityField` rather than against the plane. The
consequence is that the same five controllers deploy over a rectangle, a disk,
a sphere, a dome or an arbitrary mesh, and adding a sixth region costs a Domain
subclass rather than a fork of every algorithm.

``lloyd``
    The classical centroidal-Voronoi descent (Lloyd 1982; Cortes et al. 2004).

``limited_range``
    The same, but an agent owns only what its sensor reaches. Agents that start
    clustered then have *no gradient* to spread along, which is the defining
    difficulty of large-team deployment rather than a bug.

``adaptive``
    The density is not given but *estimated online* from what agents measure as
    they move (Schwager et al. 2009). Coverage and learning happen together.

``gmm``
    Agents are assigned to the components of a Gaussian mixture by
    responsibility, then run Lloyd inside their component. This turns a
    multi-modal region of interest into an explicit team split -- three
    buildings, three sub-teams -- without a central allocator.

``time_varying``
    Targets move, so the controller pursues rather than converges. It adds a
    feed-forward term along the density's own motion, which is the difference
    between trailing a moving target and tracking it.

References
----------
* Lloyd, S. P. 1982. *Least squares quantization in PCM.* IEEE T-IT 28(2): 129-137.
* Cortes, J.; Martinez, S.; Karatas, T.; and Bullo, F. 2004. *Coverage control
  for mobile sensing networks.* IEEE T-RA 20(2): 243-255.
* Schwager, M.; Rus, D.; and Slotine, J.-J. 2009. *Decentralized, adaptive
  coverage control for networked robots.* IJRR 28(3): 357-375.
* Bertoncelli, F.; Belal, M.; Albani, D.; Pratissoli, F.; and Sabattini, L.
  2024. *On limited-range coverage control for large-scale teams of aerial
  drones: Deployment and study.* DARS 2022, Springer.
* Belal, M.; Manoni, T.; Albani, D.; and Sabattini, L. 2026. *Decentralized
  multi-robot coverage of hemispherical surfaces via fortune-based
  partitioning.* ANTS 2026.
* Manoni, T.; et al. 2024. *Understanding the role of time-varying targets in
  adaptive distributed area coverage control.* DARS 2024, Springer.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import numpy as np

from .density import (
    DensityField,
    GaussianMixtureDensity,
    UniformDensity,
    balanced_assignment,
    get_density,
)
from .domain import Domain, PlanarDomain, get_domain

__all__ = [
    "CoverageController",
    "LloydCoverage",
    "LimitedRangeCoverage",
    "AdaptiveCoverage",
    "MixtureCoverage",
    "TimeVaryingCoverage",
    "CoverageResult",
    "CoverageSimulator",
    "register_coverage",
    "get_coverage",
    "available_coverage",
]


class CoverageController(ABC):
    """A decentralized deployment law over a domain.

    Args:
        domain: where coverage happens; a name or a :class:`Domain`.
        density: importance field; a name, a :class:`DensityField`, or a
            callable.
        samples: quadrature resolution. The cells, centroids and cost are all
            computed on this fixed sample set.
        gain: fraction of the way to the centroid to move per step.
        max_step: motion limit per iteration, so a controller cannot teleport.
    """

    name = "abstract"

    def __init__(
        self,
        domain=None,
        density=None,
        samples: int = 2000,
        gain: float = 1.0,
        max_step: float = 1.5,
    ):
        self.domain: Domain = (
            get_domain(domain) if domain is not None else PlanarDomain()
        )
        self.density: DensityField = (
            get_density(density) if density is not None else UniformDensity()
        )
        self.gain = gain
        self.max_step = max_step
        self._points = self.domain.sample(samples)

    # -- geometry -----------------------------------------------------------
    @property
    def points(self) -> np.ndarray:
        return self._points

    def weights(self, time: float = 0.0) -> np.ndarray:
        return self.density(self._points, time)

    def ownership(self, positions: np.ndarray, time: float = 0.0):
        """``(owner, distances)``; ``owner == -1`` means nobody covers it."""
        distances = self.domain.distance(self._points, positions)
        owner = np.argmin(distances, axis=1)
        return owner, distances

    def cost(self, positions: np.ndarray, time: float = 0.0) -> float:
        """Locational cost ``sum_i integral_{V_i} d(q, p_i)^2 phi(q) dq``.

        Lower is better; Lloyd is gradient descent on exactly this. Normalised
        by the sample count and scaled by the domain measure, so costs are
        comparable across domains and resolutions.
        """
        _, distances = self.ownership(positions, time)
        nearest = distances[np.arange(len(self._points)), np.argmin(distances, axis=1)]
        nearest = self._clip_range(nearest)
        weights = self.weights(time)
        return float(np.mean(weights * nearest**2) * self.domain.measure)

    def _clip_range(self, nearest: np.ndarray) -> np.ndarray:
        """Hook for limited-range variants; unlimited by default."""
        return nearest

    # -- the control law ----------------------------------------------------
    def targets(self, positions: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Where each agent wants to be. Subclasses usually override this."""
        owner, _ = self.ownership(positions, time)
        weights = self.weights(time)
        targets = positions.copy()
        for index in range(len(positions)):
            mask = owner == index
            if not np.any(mask):
                continue  # nothing to serve: hold position
            targets[index] = self.domain.centroid(self._points[mask], weights[mask])
        return targets

    def step(self, positions: np.ndarray, time: float = 0.0) -> np.ndarray:
        """One iteration: move toward the targets, limited and projected."""
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        targets = self.targets(positions, time)
        steps = self.gain * (targets - positions)
        norms = np.linalg.norm(steps, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.max_step / np.maximum(norms, 1e-9))
        return self.domain.project(positions + steps * scale)

    def observe(self, positions: np.ndarray, time: float = 0.0) -> None:
        """Hook for controllers that learn from what agents measure."""

    def __repr__(self) -> str:
        return "%s(domain=%r, density=%r)" % (
            type(self).__name__,
            self.domain,
            self.density,
        )


_CONTROLLERS: Dict[str, Type[CoverageController]] = {}


def register_coverage(name: str):
    def decorator(cls: Type[CoverageController]) -> Type[CoverageController]:
        key = name.lower()
        if key in _CONTROLLERS:
            raise ValueError("coverage controller %r already registered" % name)
        cls.name = key
        _CONTROLLERS[key] = cls
        return cls

    return decorator


def available_coverage() -> List[str]:
    return sorted(_CONTROLLERS)


def get_coverage(name, **kwargs) -> CoverageController:
    if isinstance(name, CoverageController):
        return name
    try:
        cls = _CONTROLLERS[str(name).lower()]
    except KeyError as error:
        raise ValueError(
            "Unknown coverage controller %r. Available: %s"
            % (name, ", ".join(available_coverage()))
        ) from error
    return cls(**kwargs)


@register_coverage("lloyd")
class LloydCoverage(CoverageController):
    """Classical centroidal-Voronoi descent."""


@register_coverage("limited_range")
class LimitedRangeCoverage(CoverageController):
    """Lloyd where an agent owns only what its sensor can reach.

    An agent whose footprint is empty has no local gradient at all -- the
    honest behaviour is to hold position, and the honest conclusion is that
    limited-range deployment needs a separate exploration mechanism.
    ``exploration`` supplies the cheapest such mechanism: drift away from the
    team's own centroid until something comes into view.
    """

    def __init__(self, sensing_range: float = 5.0, exploration: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.sensing_range = sensing_range
        self.exploration = exploration

    def _clip_range(self, nearest: np.ndarray) -> np.ndarray:
        return np.minimum(nearest, self.sensing_range)

    def ownership(self, positions: np.ndarray, time: float = 0.0):
        distances = self.domain.distance(self._points, positions)
        owner = np.argmin(distances, axis=1)
        nearest = distances[np.arange(len(self._points)), owner]
        return np.where(nearest <= self.sensing_range, owner, -1), distances

    def targets(self, positions: np.ndarray, time: float = 0.0) -> np.ndarray:
        owner, _ = self.ownership(positions, time)
        weights = self.weights(time)
        targets = positions.copy()
        centre = positions.mean(axis=0)
        for index in range(len(positions)):
            mask = owner == index
            if np.any(mask):
                targets[index] = self.domain.centroid(self._points[mask], weights[mask])
            elif self.exploration:
                # Nothing in range: push outward, away from the team.
                away = positions[index] - centre
                norm = float(np.linalg.norm(away))
                if norm > 1e-9:
                    targets[index] = positions[index] + self.exploration * away / norm
        return targets


@register_coverage("adaptive")
class AdaptiveCoverage(CoverageController):
    """Coverage while *learning* the density (Schwager et al. 2009).

    The controller is given a set of basis functions and starts with flat
    weights; every step, agents "measure" the true density at their own
    positions and the weights are nudged toward explaining those measurements.
    Coverage and estimation run together, which is the realistic setting: nobody
    hands a survey team the map it was sent to make.

    The estimate is deliberately conservative -- weights stay non-negative and
    are updated by a small gradient step -- because an over-eager estimator
    sends the whole team to the first hot reading.
    """

    def __init__(
        self,
        truth: Optional[DensityField] = None,
        basis=None,
        rate: float = 0.25,
        memory: int = 400,
        steps_per_update: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.truth = truth
        self.rate = rate
        if basis is None:
            # A coarse grid of bumps over the domain's own samples.
            step = max(1, len(self._points) // 16)
            centres = self._points[::step][:16]
            basis = GaussianMixtureDensity(
                means=centres,
                covariances=[self._basis_width()] * len(centres),
                floor=0.0,
            )
        self.basis = basis
        self.estimate = np.ones(self.basis.k) / self.basis.k
        self.history: List[float] = []
        # Measurements are *accumulated*. Fitting only the current positions is
        # under-determined (a handful of agents, many basis weights) and
        # self-confirming: agents move to where the estimate says to look, then
        # confirm the estimate there. A running memory of past readings is the
        # cheapest form of the persistence of excitation this needs.
        self.memory = memory
        self.steps_per_update = steps_per_update
        self._samples: List[np.ndarray] = []
        self._values: List[float] = []

    def _basis_width(self) -> float:
        span = float(np.ptp(self._points, axis=0).max())
        return (span / 4.0) ** 2

    def weights(self, time: float = 0.0) -> np.ndarray:
        components = self.basis.component_densities(self._points)
        return 1e-3 + components @ self.estimate

    def observe(self, positions: np.ndarray, time: float = 0.0) -> None:
        if self.truth is None:
            return
        measured = np.asarray(self.truth(positions, time), dtype=float)
        for position, value in zip(np.atleast_2d(positions), measured):
            self._samples.append(np.asarray(position, dtype=float))
            self._values.append(float(value))
        if len(self._samples) > self.memory:
            self._samples = self._samples[-self.memory :]
            self._values = self._values[-self.memory :]

        sample_points = np.asarray(self._samples)
        targets = np.asarray(self._values)
        components = self.basis.component_densities(sample_points)
        # Step size scaled by the basis magnitude: the components are normalised
        # Gaussians whose peak height depends on the domain size, so a fixed
        # rate is either inert or divergent depending on the map.
        scale = float(np.mean(components**2)) + 1e-12

        for _ in range(self.steps_per_update):
            predicted = components @ self.estimate
            gradient = components.T @ (predicted - targets) / max(1, len(targets))
            # Projected gradient: a negative importance is not a thing. No
            # renormalisation -- this is a least-squares fit of an unnormalised
            # field, and forcing the weights to sum to one fixes the wrong scale.
            self.estimate = np.maximum(
                0.0, self.estimate - (self.rate / scale) * gradient
            )

        residual = components @ self.estimate - targets
        self.history.append(float(np.mean(residual**2)))


@register_coverage("gmm")
class MixtureCoverage(CoverageController):
    """Split the team across the components of a Gaussian mixture.

    Each agent is assigned to the component it is most responsible for, then
    covers only that component's share of the domain. Two things fall out:
    the split is *explicit* (you can see which sub-team serves which region),
    and the number of agents per component follows the mixing weights, so a
    component carrying 60% of the importance gets roughly 60% of the fleet.

    Assignment is recomputed each step but damped by ``stickiness``, because an
    agent that re-assigns every iteration oscillates between two regions and
    covers neither.
    """

    def __init__(
        self,
        mixture: Optional[GaussianMixtureDensity] = None,
        stickiness: float = 0.7,
        **kwargs,
    ):
        if mixture is not None:
            kwargs.setdefault("density", mixture)
        super().__init__(**kwargs)
        if mixture is None:
            if not isinstance(self.density, GaussianMixtureDensity):
                raise ValueError("MixtureCoverage needs a GaussianMixtureDensity")
            mixture = self.density
        self.mixture = mixture
        self.stickiness = stickiness
        self._assignment: Optional[np.ndarray] = None

    def assign(self, positions: np.ndarray) -> np.ndarray:
        """Which component each agent serves, respecting the mixing weights."""
        responsibilities = self.mixture.responsibilities(positions)
        quota = self.mixture.weights * len(positions)
        if self._assignment is None or len(self._assignment) != len(positions):
            # First call: seed by nearest component so the quota logic has
            # counts to work with, then immediately rebalance.
            self._assignment = np.argmax(responsibilities, axis=1)
        self._assignment = balanced_assignment(
            responsibilities, quota, self._assignment, self.stickiness
        )
        return self._assignment

    def targets(self, positions: np.ndarray, time: float = 0.0) -> np.ndarray:
        assignment = self.assign(positions)
        weights = self.weights(time)
        component_weights = self.mixture.component_densities(self._points)
        targets = positions.copy()

        for component in range(self.mixture.k):
            members = np.flatnonzero(assignment == component)
            if not len(members):
                continue
            # Restrict the quadrature to this component's share of importance,
            # then run an ordinary Lloyd step among its members only.
            share = component_weights[:, component] * self.mixture.weights[component]
            relevant = (
                share > share.max() * 0.05
                if share.max() > 0
                else np.ones(len(share), bool)
            )
            points = self._points[relevant]
            local_weights = weights[relevant] * share[relevant]
            if not len(points):
                continue
            distances = self.domain.distance(points, positions[members])
            owner = np.argmin(distances, axis=1)
            for local_index, agent in enumerate(members):
                mask = owner == local_index
                if np.any(mask):
                    targets[agent] = self.domain.centroid(
                        points[mask], local_weights[mask]
                    )
        return targets


@register_coverage("time_varying")
class TimeVaryingCoverage(CoverageController):
    """Pursue a moving density instead of converging on a static one.

    The centroid an agent chases has already moved by the time it arrives, so
    the controller adds a feed-forward term: the observed velocity of its own
    target, estimated by finite differences. Without it the team trails the
    target permanently; with it the lag is bounded by the estimator, not by the
    controller gain.
    """

    def __init__(self, lookahead: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.lookahead = lookahead
        self._previous: Optional[np.ndarray] = None
        self._previous_time: float = 0.0

    def targets(self, positions: np.ndarray, time: float = 0.0) -> np.ndarray:
        targets = super().targets(positions, time)
        if self._previous is not None and len(self._previous) == len(targets):
            dt = max(1e-6, time - self._previous_time)
            velocity = (targets - self._previous) / dt
            targets = targets + self.lookahead * velocity * dt
        self._previous = targets.copy()
        self._previous_time = time
        return targets


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------


@dataclass
class CoverageResult:
    history: List[np.ndarray]
    costs: List[float]
    controller: str
    extras: Dict[str, List[float]] = field(default_factory=dict)

    @property
    def final(self) -> np.ndarray:
        return self.history[-1]

    @property
    def improvement(self) -> float:
        """Fraction of the initial cost removed; 1.0 would be perfect."""
        return 1.0 - self.costs[-1] / self.costs[0] if self.costs[0] else 0.0


class CoverageSimulator:
    """Runs a coverage controller and records cost per iteration."""

    def __init__(
        self,
        controller="lloyd",
        n_agents: int = 8,
        dt: float = 1.0,
        seed: int = 0,
        clustered_start: bool = True,
        **controller_kwargs,
    ):
        self.controller = get_coverage(controller, **controller_kwargs)
        self.n_agents = n_agents
        self.dt = dt
        self.seed = seed
        self.clustered_start = clustered_start

    def initial_positions(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return self.controller.domain.random_positions(
            self.n_agents, rng, clustered=self.clustered_start
        )

    def run(
        self, steps: int = 50, initial: Optional[np.ndarray] = None
    ) -> CoverageResult:
        positions = (
            np.array(initial, dtype=float)
            if initial is not None
            else self.initial_positions()
        )
        history = [positions.copy()]
        costs = [self.controller.cost(positions, 0.0)]

        for index in range(steps):
            time = (index + 1) * self.dt
            self.controller.observe(positions, time)
            positions = self.controller.step(positions, time)
            history.append(positions.copy())
            costs.append(self.controller.cost(positions, time))

        extras = {}
        if isinstance(self.controller, AdaptiveCoverage) and self.controller.history:
            extras["estimation_error"] = list(self.controller.history)
        return CoverageResult(history, costs, self.controller.name, extras)
