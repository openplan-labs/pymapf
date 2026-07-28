"""Where coverage happens -- as an object, so one algorithm serves every shape.

Lloyd's algorithm is three operations: assign each sample point to its nearest
agent, compute the weighted centroid of each cell, move toward it. None of the
three mention geometry, and yet a planar implementation cannot deploy a team
over a dome, because "nearest" is great-circle distance there and "centroid" is
a spherical mean.

A :class:`Domain` supplies exactly those three primitives -- sampling,
distance, centroid -- plus projection back onto the surface. Every coverage
controller in :mod:`pymapf.swarm.coverage` is then written once and runs on all
of them.

    PlanarDomain      a rectangle -- the classical setting
    DiskDomain        a circular arena
    SphereDomain      a full sphere: satellite constellations, tank inspection
    HemisphereDomain  a dome: the fortune-based partitioning setting
    AnnulusDomain     a ring -- a perimeter to patrol rather than an area
    MeshDomain        an arbitrary point cloud, for anything the above misses

Sampling is quadrature-based throughout (a fixed sample set, reused every
iteration), which keeps the whole package dependency-free and gives the same
fixed points as an exact Voronoi construction as the sampling gets finer.

References
----------
* Lloyd, S. P. 1982. *Least squares quantization in PCM.* IEEE Transactions on
  Information Theory 28(2): 129-137.
* Cortes, J.; Martinez, S.; Karatas, T.; and Bullo, F. 2004. *Coverage control
  for mobile sensing networks.* IEEE T-RA 20(2): 243-255.
* Belal, M.; Manoni, T.; Albani, D.; and Sabattini, L. 2026. *Decentralized
  multi-robot coverage of hemispherical surfaces via fortune-based
  partitioning.* ANTS 2026.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Sequence, Tuple

import numpy as np

__all__ = [
    "Domain",
    "PlanarDomain",
    "DiskDomain",
    "SphereDomain",
    "HemisphereDomain",
    "AnnulusDomain",
    "MeshDomain",
    "get_domain",
    "DOMAINS",
]


class Domain(ABC):
    """A region to be covered, with the geometry the algorithms need."""

    name = "abstract"
    dimension = 2

    @abstractmethod
    def sample(self, count: int = 2000) -> np.ndarray:
        """Quadrature points, as uniformly spread over the region as possible."""

    def distance(self, points: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """``(len(points), len(positions))`` distances. Euclidean by default."""
        return np.linalg.norm(points[:, None, :] - positions[None, :, :], axis=2)

    def centroid(self, points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Weighted centroid of a cell, on the domain."""
        total = float(weights.sum())
        if total <= 1e-12:
            return points.mean(axis=0)
        return (points * weights[:, None]).sum(axis=0) / total

    def project(self, positions: np.ndarray) -> np.ndarray:
        """Push positions back onto the domain (a no-op for full-dimensional ones)."""
        return positions

    def random_positions(self, n: int, rng, clustered: bool = True) -> np.ndarray:
        """Starting positions. Clustered by default -- teams launch together."""
        points = self.sample(max(200, 20 * n))
        if clustered:
            anchor = points[rng.integers(len(points))]
            order = np.argsort(np.linalg.norm(points - anchor, axis=1))
            return self.project(points[order[:n]].copy())
        return self.project(points[rng.choice(len(points), size=n, replace=False)].copy())

    @property
    def measure(self) -> float:
        """Area/volume, used to normalise costs across domains."""
        return 1.0

    def __repr__(self) -> str:
        return "%s()" % type(self).__name__


class PlanarDomain(Domain):
    """An axis-aligned rectangle."""

    name = "planar"

    def __init__(self, bounds: Tuple[float, float, float, float] = (0.0, 0.0, 20.0, 20.0)):
        self.bounds = tuple(float(b) for b in bounds)

    def sample(self, count: int = 2000) -> np.ndarray:
        x0, y0, x1, y1 = self.bounds
        aspect = max(1e-9, (x1 - x0) / max(1e-9, (y1 - y0)))
        nx = max(2, int(round(math.sqrt(count * aspect))))
        ny = max(2, int(round(count / nx)))
        xs = np.linspace(x0, x1, nx)
        ys = np.linspace(y0, y1, ny)
        return np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)

    def project(self, positions: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.bounds
        clipped = positions.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], x0, x1)
        clipped[:, 1] = np.clip(clipped[:, 1], y0, y1)
        return clipped

    @property
    def measure(self) -> float:
        x0, y0, x1, y1 = self.bounds
        return (x1 - x0) * (y1 - y0)

    def __repr__(self) -> str:
        return "PlanarDomain(bounds=%s)" % (self.bounds,)


class DiskDomain(Domain):
    """A circular arena -- the shape of most flight cages and test arenas."""

    name = "disk"

    def __init__(self, radius: float = 10.0, centre: Sequence[float] = (0.0, 0.0)):
        self.radius = float(radius)
        self.centre = np.asarray(centre, dtype=float)

    def sample(self, count: int = 2000) -> np.ndarray:
        # Sunflower (Vogel) spiral: near-uniform density on a disk, unlike a
        # polar grid, which piles points at the centre.
        indices = np.arange(count, dtype=float) + 0.5
        radii = self.radius * np.sqrt(indices / count)
        golden = math.pi * (3.0 - math.sqrt(5.0))
        angles = golden * indices
        return self.centre + np.stack(
            [radii * np.cos(angles), radii * np.sin(angles)], axis=1
        )

    def project(self, positions: np.ndarray) -> np.ndarray:
        offsets = positions - self.centre
        norms = np.linalg.norm(offsets, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.radius / np.maximum(norms, 1e-9))
        return self.centre + offsets * scale

    @property
    def measure(self) -> float:
        return math.pi * self.radius ** 2

    def __repr__(self) -> str:
        return "DiskDomain(radius=%.3g)" % self.radius


class SphereDomain(Domain):
    """A spherical surface: distances are great-circle, centroids spherical."""

    name = "sphere"
    dimension = 3

    def __init__(self, radius: float = 10.0):
        self.radius = float(radius)

    def sample(self, count: int = 2000) -> np.ndarray:
        # Fibonacci lattice. Sampling by latitude/longitude clusters points at
        # the poles and biases every centroid computed from them.
        indices = np.arange(count, dtype=float) + 0.5
        z = 1.0 - 2.0 * indices / count
        r = np.sqrt(np.maximum(0.0, 1.0 - z ** 2))
        golden = math.pi * (3.0 - math.sqrt(5.0))
        theta = golden * indices
        return self.radius * np.stack(
            [r * np.cos(theta), r * np.sin(theta), z], axis=1
        )

    def distance(self, points: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Great-circle distance -- the only meaningful one on a surface."""
        unit_points = points / np.maximum(
            np.linalg.norm(points, axis=1, keepdims=True), 1e-12
        )
        unit_positions = positions / np.maximum(
            np.linalg.norm(positions, axis=1, keepdims=True), 1e-12
        )
        cosines = np.clip(unit_points @ unit_positions.T, -1.0, 1.0)
        return self.radius * np.arccos(cosines)

    def centroid(self, points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        total = float(weights.sum())
        mean = (
            points.mean(axis=0)
            if total <= 1e-12
            else (points * weights[:, None]).sum(axis=0) / total
        )
        norm = float(np.linalg.norm(mean))
        if norm < 1e-9:  # antipodal cell: the mean is at the centre, no direction
            return points[0].copy()
        return mean / norm * self.radius

    def project(self, positions: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(positions, axis=1, keepdims=True)
        return positions / np.maximum(norms, 1e-9) * self.radius

    @property
    def measure(self) -> float:
        return 4 * math.pi * self.radius ** 2

    def __repr__(self) -> str:
        return "SphereDomain(radius=%.3g)" % self.radius


class HemisphereDomain(SphereDomain):
    """The upper half of a sphere: a dome, a tank top, a radome.

    Inheriting from :class:`SphereDomain` is the point -- the geodesic distance
    and spherical centroid are unchanged, only the sampled region differs -- but
    the boundary is not: a cell touching the equator has its centroid pulled
    inward, which is exactly the effect that makes bounded-surface coverage
    harder than the closed-surface case.
    """

    name = "hemisphere"

    def sample(self, count: int = 2000) -> np.ndarray:
        indices = np.arange(count, dtype=float) + 0.5
        z = 1.0 - indices / count  # (0, 1] -- upper half only
        r = np.sqrt(np.maximum(0.0, 1.0 - z ** 2))
        golden = math.pi * (3.0 - math.sqrt(5.0))
        theta = golden * indices
        return self.radius * np.stack(
            [r * np.cos(theta), r * np.sin(theta), z], axis=1
        )

    def project(self, positions: np.ndarray) -> np.ndarray:
        projected = super().project(positions)
        # Reflect anything that slipped below the equator back onto the dome.
        below = projected[:, 2] < 0
        projected[below, 2] = np.abs(projected[below, 2])
        return super().project(projected)

    @property
    def measure(self) -> float:
        return 2 * math.pi * self.radius ** 2


class AnnulusDomain(Domain):
    """A ring: a perimeter to watch rather than an area to fill."""

    name = "annulus"

    def __init__(self, inner: float = 6.0, outer: float = 10.0, centre=(0.0, 0.0)):
        if inner >= outer:
            raise ValueError("inner radius must be smaller than outer")
        self.inner = float(inner)
        self.outer = float(outer)
        self.centre = np.asarray(centre, dtype=float)

    def sample(self, count: int = 2000) -> np.ndarray:
        indices = np.arange(count, dtype=float) + 0.5
        fraction = indices / count
        radii = np.sqrt(self.inner ** 2 + fraction * (self.outer ** 2 - self.inner ** 2))
        golden = math.pi * (3.0 - math.sqrt(5.0))
        angles = golden * indices
        return self.centre + np.stack(
            [radii * np.cos(angles), radii * np.sin(angles)], axis=1
        )

    def project(self, positions: np.ndarray) -> np.ndarray:
        offsets = positions - self.centre
        norms = np.linalg.norm(offsets, axis=1, keepdims=True)
        clamped = np.clip(norms, self.inner, self.outer)
        return self.centre + offsets / np.maximum(norms, 1e-9) * clamped

    @property
    def measure(self) -> float:
        return math.pi * (self.outer ** 2 - self.inner ** 2)


class MeshDomain(Domain):
    """An arbitrary point cloud: whatever shape the others do not cover.

    Distances are Euclidean in the ambient space and centroids are projected
    back to the nearest sample, which makes it correct on any surface as the
    cloud gets denser.
    """

    name = "mesh"

    def __init__(self, points: np.ndarray):
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2:
            raise ValueError("mesh points must be a (m, d) array")
        self.dimension = self.points.shape[1]

    def sample(self, count: int = 2000) -> np.ndarray:
        if count >= len(self.points):
            return self.points.copy()
        step = max(1, len(self.points) // count)
        return self.points[::step][:count].copy()

    def centroid(self, points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        mean = super().centroid(points, weights)
        return self.project(mean[None, :])[0]

    def project(self, positions: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(
            positions[:, None, :] - self.points[None, :, :], axis=2
        )
        return self.points[np.argmin(distances, axis=1)].copy()


DOMAINS = {
    "planar": PlanarDomain,
    "disk": DiskDomain,
    "sphere": SphereDomain,
    "hemisphere": HemisphereDomain,
    "annulus": AnnulusDomain,
}


def get_domain(name, **kwargs) -> Domain:
    if isinstance(name, Domain):
        return name
    try:
        return DOMAINS[str(name).lower()](**kwargs)
    except KeyError:
        raise ValueError(
            "Unknown domain %r. Available: %s" % (name, ", ".join(sorted(DOMAINS)))
        )
