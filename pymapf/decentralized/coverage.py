"""Decentralized coverage control: spreading a team to watch an area.

Coverage is the other half of the decentralized story. Where flocking asks
"how do we move together?", coverage asks "where should each of us sit so that
the team as a whole sees the most?" -- the deployment problem behind
surveillance, environmental monitoring and inspection.

The classical answer is Lloyd's algorithm on a Voronoi partition: each agent
owns the region of the environment it is closest to, and repeatedly moves to
that region's centroid, weighted by how much you care about each point. It is
fully decentralized (an agent only needs its Voronoi neighbours), it provably
descends the coverage cost, and it converges to a centroidal Voronoi
configuration.

Two variants matter in practice and are implemented here:

* **limited range** -- a real sensor sees R metres, not the whole plane, so the
  region an agent owns is its Voronoi cell *intersected with a disc*. Agents
  that start far apart then have disjoint regions and no gradient between them,
  which is exactly the deployment regime studied for large aerial teams.
* **curved domains** -- inspecting a dome, a tank or a hemisphere is coverage
  on a surface, where "centroid" means the spherical centroid of a geodesic
  cell rather than the planar one.

The implementation is deliberately grid-quadrature based: the environment is
sampled, each sample is assigned to its owner, and centroids are computed by
weighted averaging. That is O(samples x agents) per step and needs no
computational-geometry dependency, while giving the same fixed points as an
exact Voronoi construction as the sampling gets finer.

References
----------
* Lloyd, S. P. 1982. *Least squares quantization in PCM.* IEEE Transactions on
  Information Theory 28(2): 129-137.
* Cortes, J.; Martinez, S.; Karatas, T.; and Bullo, F. 2004. *Coverage control
  for mobile sensing networks.* IEEE Transactions on Robotics and Automation
  20(2): 243-255.
* Schwager, M.; Rus, D.; and Slotine, J.-J. 2009. *Decentralized, adaptive
  coverage control for networked robots.* The International Journal of Robotics
  Research 28(3): 357-375.
* Bertoncelli, F.; Belal, M.; Albani, D.; Pratissoli, F.; and Sabattini, L.
  2024. *On limited-range coverage control for large-scale teams of aerial
  drones: Deployment and study.* In Distributed Autonomous Robotic Systems
  (DARS 2022), Springer Proceedings in Advanced Robotics.
* Belal, M.; Manoni, T.; Albani, D.; and Sabattini, L. 2026. *Decentralized
  multi-robot coverage of hemispherical surfaces via fortune-based
  partitioning.* ANTS 2026.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

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

    bounds: Tuple[float, float, float, float] = (0.0, 0.0, 20.0, 20.0)  # x0, y0, x1, y1
    resolution: float = 0.5  # quadrature spacing
    sensing_range: Optional[float] = None  # None = unlimited (classic Lloyd)
    gain: float = 1.0  # step toward the centroid, in [0, 1]
    max_speed: float = 1.5
    seed: int = 0
    density: Optional[Callable[[np.ndarray], np.ndarray]] = None


def uniform_density(points: np.ndarray) -> np.ndarray:
    """Every point matters equally."""
    return np.ones(len(points))


def gaussian_density(
    centres: Sequence[Tuple[float, float]], sigma: float = 3.0, floor: float = 0.02
) -> Callable[[np.ndarray], np.ndarray]:
    """Importance concentrated around points of interest.

    This is what turns coverage from "spread out evenly" into "watch these
    places, and use whatever is left over to cover the rest".
    """

    centre_array = np.asarray(centres, dtype=float)

    def density(points: np.ndarray) -> np.ndarray:
        weights = np.full(len(points), floor)
        for centre in centre_array:
            offsets = points - centre
            weights += np.exp(-np.sum(offsets ** 2, axis=1) / (2 * sigma ** 2))
        return weights

    return density


def _quadrature(params: CoverageParams) -> Tuple[np.ndarray, np.ndarray]:
    """Sample points and their importance weights over the environment."""
    x0, y0, x1, y1 = params.bounds
    xs = np.arange(x0, x1 + 1e-9, params.resolution)
    ys = np.arange(y0, y1 + 1e-9, params.resolution)
    grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    density = params.density or uniform_density
    return grid, density(grid)


def _ownership(points: np.ndarray, positions: np.ndarray, sensing_range: Optional[float]):
    """Which agent owns each sample point (its Voronoi cell, optionally clipped)."""
    distances = np.linalg.norm(points[:, None, :] - positions[None, :, :], axis=2)
    owner = np.argmin(distances, axis=1)
    if sensing_range is not None:
        # Outside every sensor footprint means nobody owns it -- the defining
        # feature of limited-range coverage.
        nearest = distances[np.arange(len(points)), owner]
        owner = np.where(nearest <= sensing_range, owner, -1)
    return owner, distances


def coverage_cost(
    positions: np.ndarray, params: Optional[CoverageParams] = None
) -> float:
    """The locational cost sum_i integral_{V_i} |q - p_i|^2 phi(q) dq.

    Lower is better; Lloyd's algorithm is gradient descent on exactly this.
    Points outside every sensor range are charged at the range limit, so the
    cost stays comparable between the limited and unlimited variants.
    """
    params = params or CoverageParams()
    points, weights = _quadrature(params)
    owner, distances = _ownership(points, positions, params.sensing_range)
    nearest = distances[np.arange(len(points)), np.argmin(distances, axis=1)]
    if params.sensing_range is not None:
        nearest = np.minimum(nearest, params.sensing_range)
    return float(np.sum(weights * nearest ** 2) * params.resolution ** 2)


def lloyd_step(
    positions: np.ndarray, params: Optional[CoverageParams] = None
) -> np.ndarray:
    """One decentralized Lloyd iteration: move each agent toward its centroid.

    An agent whose region is empty (it sees nothing within range) holds
    position: with a limited sensor there is genuinely no local gradient to
    follow, which is the honest behaviour and the reason limited-range
    deployment needs a separate exploration mechanism.
    """
    params = params or CoverageParams()
    points, weights = _quadrature(params)
    owner, _ = _ownership(points, positions, params.sensing_range)

    updated = positions.copy()
    for index in range(len(positions)):
        mask = owner == index
        if not np.any(mask):
            continue
        cell_weights = weights[mask]
        total = float(cell_weights.sum())
        if total <= 1e-12:
            continue
        centroid = (points[mask] * cell_weights[:, None]).sum(axis=0) / total
        step = params.gain * (centroid - positions[index])
        norm = float(np.linalg.norm(step))
        if norm > params.max_speed:
            step *= params.max_speed / norm
        updated[index] = positions[index] + step

    x0, y0, x1, y1 = params.bounds
    updated[:, 0] = np.clip(updated[:, 0], x0, x1)
    updated[:, 1] = np.clip(updated[:, 1], y0, y1)
    return updated


def simulate_coverage(
    n_agents: int = 8,
    steps: int = 60,
    params: Optional[CoverageParams] = None,
    initial: Optional[np.ndarray] = None,
    spawn: str = "corner",
) -> Tuple[List[np.ndarray], List[float]]:
    """Run Lloyd descent and record positions and cost per iteration.

    ``spawn="corner"`` starts the team clustered in one corner, which is the
    interesting case: it is how a real deployment begins (everyone launches from
    the same place) and it is where limited-range coverage differs most from the
    classical version.
    """
    params = params or CoverageParams()
    rng = np.random.default_rng(params.seed)
    x0, y0, x1, y1 = params.bounds

    if initial is not None:
        positions = np.asarray(initial, dtype=float).copy()
    elif spawn == "corner":
        positions = rng.uniform(0, 0.15, size=(n_agents, 2)) * np.array([x1 - x0, y1 - y0])
        positions += np.array([x0, y0]) + 0.5
    else:
        positions = rng.uniform([x0, y0], [x1, y1], size=(n_agents, 2))

    history = [positions.copy()]
    costs = [coverage_cost(positions, params)]
    for _ in range(steps):
        positions = lloyd_step(positions, params)
        history.append(positions.copy())
        costs.append(coverage_cost(positions, params))
    return history, costs


# --------------------------------------------------------------------------
# coverage on a curved surface
# --------------------------------------------------------------------------


def hemisphere_samples(radius: float = 10.0, count: int = 2000) -> np.ndarray:
    """Near-uniform samples on the upper hemisphere (Fibonacci lattice).

    Sampling a sphere by latitude/longitude clusters points at the pole and
    biases every centroid computed from them; the Fibonacci lattice does not.
    """
    indices = np.arange(count, dtype=float) + 0.5
    # z in (0, 1] for the upper half only.
    z = 1.0 - indices / count
    r = np.sqrt(np.maximum(0.0, 1.0 - z ** 2))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    theta = golden * indices
    points = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)
    return points * radius


def spherical_lloyd_step(
    positions: np.ndarray,
    samples: np.ndarray,
    radius: float = 10.0,
    gain: float = 1.0,
    sensing_angle: Optional[float] = None,
) -> np.ndarray:
    """Lloyd on a sphere: geodesic ownership, spherical centroids, re-projected.

    Distances are great-circle, so cells are geodesic Voronoi regions; the
    centroid of a cell is the normalised mean of its points pushed back onto the
    surface. ``sensing_angle`` (radians of arc) gives the limited-range variant.

    This is the discrete-quadrature counterpart of the exact fortune-style
    partitioning of Belal et al. (2026); it converges to the same centroidal
    configurations while staying dependency-free.
    """
    unit_positions = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    unit_samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)

    # Great-circle distance via the dot product, clipped for numerical safety.
    cosines = np.clip(unit_samples @ unit_positions.T, -1.0, 1.0)
    arcs = np.arccos(cosines)
    owner = np.argmin(arcs, axis=1)
    if sensing_angle is not None:
        nearest = arcs[np.arange(len(samples)), owner]
        owner = np.where(nearest <= sensing_angle, owner, -1)

    updated = unit_positions.copy()
    for index in range(len(positions)):
        mask = owner == index
        if not np.any(mask):
            continue
        mean = unit_samples[mask].mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm < 1e-9:
            continue
        target = mean / norm
        moved = unit_positions[index] + gain * (target - unit_positions[index])
        moved_norm = float(np.linalg.norm(moved))
        if moved_norm > 1e-9:
            updated[index] = moved / moved_norm
    return updated * radius


def simulate_spherical_coverage(
    n_agents: int = 10,
    steps: int = 40,
    radius: float = 10.0,
    sample_count: int = 2000,
    sensing_angle: Optional[float] = None,
    seed: int = 0,
) -> Tuple[List[np.ndarray], List[float]]:
    """Deploy a team over a hemispherical surface and record the coverage cost."""
    rng = np.random.default_rng(seed)
    samples = hemisphere_samples(radius, sample_count)

    # Start clustered near the pole: the realistic "everyone launches together"
    # initial condition.
    start = rng.normal(0, 0.12, size=(n_agents, 3)) + np.array([0.0, 0.0, 1.0])
    positions = start / np.linalg.norm(start, axis=1, keepdims=True) * radius

    def cost(current: np.ndarray) -> float:
        unit_p = current / np.linalg.norm(current, axis=1, keepdims=True)
        unit_s = samples / np.linalg.norm(samples, axis=1, keepdims=True)
        arcs = np.arccos(np.clip(unit_s @ unit_p.T, -1.0, 1.0))
        nearest = arcs.min(axis=1)
        if sensing_angle is not None:
            nearest = np.minimum(nearest, sensing_angle)
        return float(np.mean(nearest ** 2) * radius ** 2)

    history = [positions.copy()]
    costs = [cost(positions)]
    for _ in range(steps):
        positions = spherical_lloyd_step(
            positions, samples, radius=radius, sensing_angle=sensing_angle
        )
        history.append(positions.copy())
        costs.append(cost(positions))
    return history, costs
