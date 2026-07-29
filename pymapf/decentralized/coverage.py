"""Functional façade over :mod:`pymapf.swarm.coverage` (backwards compatibility).

Coverage moved to :mod:`pymapf.swarm.coverage`, where controllers are objects
written against a :class:`~pymapf.swarm.domain.Domain` and a
:class:`~pymapf.swarm.density.DensityField` -- so the same algorithm deploys a
team on a rectangle, a disk, a sphere, a dome or a mesh. This module keeps the
older functions working by delegating to that layer.

New code should prefer::

    from pymapf.swarm import CoverageSimulator
    result = CoverageSimulator("lloyd", domain="hemisphere", n_agents=10).run(steps=40)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..swarm.coverage import CoverageSimulator, LimitedRangeCoverage, LloydCoverage
from ..swarm.density import GaussianMixtureDensity, UniformDensity
from ..swarm.domain import HemisphereDomain, PlanarDomain

__all__ = [
    "CoverageParams",
    "uniform_density",
    "gaussian_density",
    "lloyd_step",
    "coverage_cost",
    "simulate_coverage",
    "hemisphere_samples",
    "spherical_lloyd_step",
    "simulate_spherical_coverage",
]


@dataclass
class CoverageParams:
    """Environment and gains for a planar coverage run."""

    bounds: Tuple[float, float, float, float] = (0.0, 0.0, 20.0, 20.0)
    resolution: float = 0.5
    sensing_range: Optional[float] = None
    gain: float = 1.0
    max_speed: float = 1.5
    seed: int = 0
    density: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def controller(self):
        """The :mod:`pymapf.swarm.coverage` controller these params describe."""
        x0, y0, x1, y1 = self.bounds
        samples = max(
            64,
            int(((x1 - x0) / self.resolution + 1) * ((y1 - y0) / self.resolution + 1)),
        )
        common = dict(
            domain=PlanarDomain(self.bounds),
            density=self.density if self.density is not None else UniformDensity(),
            samples=samples,
            gain=self.gain,
            max_step=self.max_speed,
        )
        if self.sensing_range is None:
            return LloydCoverage(**common)
        return LimitedRangeCoverage(sensing_range=self.sensing_range, **common)


def uniform_density(points: np.ndarray) -> np.ndarray:
    """Every point matters equally."""
    return np.ones(len(points))


def gaussian_density(
    centres: Sequence[Tuple[float, float]], sigma: float = 3.0, floor: float = 0.02
) -> Callable[[np.ndarray], np.ndarray]:
    """Importance concentrated around points of interest.

    Now a thin wrapper over :class:`~pymapf.swarm.density.GaussianMixtureDensity`,
    which additionally supports per-component covariances, weights, sampling and
    EM fitting.
    """
    mixture = GaussianMixtureDensity(
        means=centres, covariances=[sigma**2] * len(centres), floor=0.0
    )

    def density(points: np.ndarray) -> np.ndarray:
        # Preserve the original scaling: peak 1 per component, plus a floor.
        components = mixture.component_densities(np.atleast_2d(points))
        peak = float(mixture._norms[0]) if mixture._norms else 1.0
        return floor + components.sum(axis=1) / max(peak, 1e-300)

    return density


def coverage_cost(
    positions: np.ndarray, params: Optional[CoverageParams] = None
) -> float:
    """Locational cost of a configuration."""
    params = params or CoverageParams()
    return params.controller().cost(np.atleast_2d(np.asarray(positions, dtype=float)))


def lloyd_step(
    positions: np.ndarray, params: Optional[CoverageParams] = None
) -> np.ndarray:
    """One decentralized Lloyd iteration."""
    params = params or CoverageParams()
    return params.controller().step(np.atleast_2d(np.asarray(positions, dtype=float)))


def simulate_coverage(
    n_agents: int = 8,
    steps: int = 60,
    params: Optional[CoverageParams] = None,
    initial: Optional[np.ndarray] = None,
    spawn: str = "corner",
) -> Tuple[List[np.ndarray], List[float]]:
    """Run Lloyd descent; returns ``(history, costs)`` as it always did."""
    params = params or CoverageParams()
    controller = params.controller()

    if initial is None:
        rng = np.random.default_rng(params.seed)
        x0, y0, x1, y1 = params.bounds
        if spawn == "corner":
            initial = rng.uniform(0, 0.15, size=(n_agents, 2)) * np.array(
                [x1 - x0, y1 - y0]
            )
            initial += np.array([x0, y0]) + 0.5
        else:
            initial = rng.uniform([x0, y0], [x1, y1], size=(n_agents, 2))

    simulator = CoverageSimulator(controller, n_agents=n_agents, seed=params.seed)
    result = simulator.run(steps=steps, initial=np.asarray(initial, dtype=float))
    return result.history, result.costs


def hemisphere_samples(radius: float = 10.0, count: int = 2000) -> np.ndarray:
    """Near-uniform samples on the upper hemisphere (Fibonacci lattice)."""
    return HemisphereDomain(radius).sample(count)


def spherical_lloyd_step(
    positions: np.ndarray,
    samples: np.ndarray,
    radius: float = 10.0,
    gain: float = 1.0,
    sensing_angle: Optional[float] = None,
) -> np.ndarray:
    """One Lloyd iteration on a sphere, with geodesic cells and centroids."""
    domain = HemisphereDomain(radius)
    controller = LloydCoverage(
        domain=domain, samples=len(samples), gain=gain, max_step=radius
    )
    controller._points = np.asarray(samples, dtype=float)
    if sensing_angle is not None:
        controller = LimitedRangeCoverage(
            sensing_range=sensing_angle * radius,
            domain=domain,
            samples=len(samples),
            gain=gain,
            max_step=radius,
        )
        controller._points = np.asarray(samples, dtype=float)
    return controller.step(np.atleast_2d(np.asarray(positions, dtype=float)))


def simulate_spherical_coverage(
    n_agents: int = 10,
    steps: int = 40,
    radius: float = 10.0,
    sample_count: int = 2000,
    sensing_angle: Optional[float] = None,
    seed: int = 0,
) -> Tuple[List[np.ndarray], List[float]]:
    """Deploy a team over a hemispherical surface; returns ``(history, costs)``."""
    domain = HemisphereDomain(radius)
    kwargs = dict(domain=domain, samples=sample_count, max_step=radius)
    controller = (
        LloydCoverage(**kwargs)
        if sensing_angle is None
        else LimitedRangeCoverage(sensing_range=sensing_angle * radius, **kwargs)
    )

    rng = np.random.default_rng(seed)
    start = rng.normal(0, 0.12, size=(n_agents, 3)) + np.array([0.0, 0.0, 1.0])
    initial = start / np.linalg.norm(start, axis=1, keepdims=True) * radius

    simulator = CoverageSimulator(controller, n_agents=n_agents, seed=seed)
    result = simulator.run(steps=steps, initial=initial)
    # The historical cost scale was the mean squared arc length times r^2.
    costs = [
        float(
            np.mean(
                np.minimum(
                    domain.distance(controller.points, positions).min(axis=1),
                    sensing_angle * radius if sensing_angle else np.inf,
                )
                ** 2
            )
        )
        for positions in result.history
    ]
    return result.history, costs
