"""Importance over the domain: what the swarm should care about, and where.

Coverage without a density function spreads a team evenly, which is rarely what
anyone wants. A density field says "this corner matters ten times more than
that one", and everything downstream -- Voronoi cells, centroids, the coverage
cost -- is weighted by it.

The same object doubles as the *target distribution* for
:mod:`pymapf.swarm.distribution`, where the swarm is steered to match a density
rather than to watch one. That is why this is its own module: a Gaussian
mixture is both "where to look" and "where to be".

    from pymapf.swarm.density import GaussianMixtureDensity

    density = GaussianMixtureDensity(
        means=[(4, 4), (16, 15)], covariances=[2.0, 6.0], weights=[0.7, 0.3]
    )

References
----------
* Cortes, J.; Martinez, S.; Karatas, T.; and Bullo, F. 2004. *Coverage control
  for mobile sensing networks.* IEEE T-RA 20(2): 243-255.  (weighted coverage)
* Schwager, M.; Rus, D.; and Slotine, J.-J. 2009. *Decentralized, adaptive
  coverage control for networked robots.* IJRR 28(3): 357-375.  (estimating the
  density online instead of being given it)
* Bishop, C. M. 2006. *Pattern Recognition and Machine Learning*, ch. 9.
  (mixture models and EM, the fitting used by :meth:`GaussianMixtureDensity.fit`)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union

import numpy as np

__all__ = [
    "balanced_assignment",
    "DensityField",
    "UniformDensity",
    "GaussianDensity",
    "GaussianMixtureDensity",
    "TimeVaryingDensity",
    "SampledDensity",
    "get_density",
]


class DensityField(ABC):
    """A non-negative importance function over the domain."""

    name = "abstract"

    @abstractmethod
    def __call__(self, points: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Importance at each point; shape ``(len(points),)``."""

    def normalised(self, points: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Importance summing to 1 over the sample -- a discrete distribution."""
        values = np.asarray(self(points, time), dtype=float)
        total = float(values.sum())
        return values / total if total > 1e-12 else np.full(len(points), 1.0 / max(1, len(points)))

    def __repr__(self) -> str:
        return "%s()" % type(self).__name__


class UniformDensity(DensityField):
    """Every point matters equally."""

    name = "uniform"

    def __call__(self, points: np.ndarray, time: float = 0.0) -> np.ndarray:
        return np.ones(len(points))


class GaussianDensity(DensityField):
    """A single isotropic hot spot, plus a floor so the rest is not ignored."""

    name = "gaussian"

    def __init__(self, mean: Sequence[float], sigma: float = 3.0, floor: float = 0.02):
        self.mean = np.asarray(mean, dtype=float)
        self.sigma = float(sigma)
        self.floor = float(floor)

    def __call__(self, points: np.ndarray, time: float = 0.0) -> np.ndarray:
        offsets = points - self.mean
        return self.floor + np.exp(-np.sum(offsets ** 2, axis=1) / (2 * self.sigma ** 2))


class GaussianMixtureDensity(DensityField):
    """A weighted sum of Gaussians -- the workhorse importance model.

    A mixture is the right default for multi-agent work because it is closed
    under the operations you actually need: it can be *fitted* to observations
    (:meth:`fit`), *sampled* from (:meth:`sample`), evaluated cheaply anywhere,
    and it describes a multi-modal region of interest -- several buildings, a
    few thermals, two crowds -- which a single Gaussian cannot.

    Args:
        means: ``(k, d)`` component centres.
        covariances: per component, either a scalar variance, a ``(d,)``
            diagonal, or a full ``(d, d)`` matrix. Mixed forms are fine.
        weights: mixing coefficients; normalised internally.
        floor: added everywhere, so a region far from every component is still
            worth a little -- without it, agents outside the support get no
            gradient at all.
    """

    name = "gmm"

    def __init__(
        self,
        means: Sequence[Sequence[float]],
        covariances: Optional[Sequence[Union[float, Sequence]]] = None,
        weights: Optional[Sequence[float]] = None,
        floor: float = 0.02,
    ):
        self.means = np.atleast_2d(np.asarray(means, dtype=float))
        self.k, self.dimension = self.means.shape
        self.floor = float(floor)

        if covariances is None:
            covariances = [1.0] * self.k
        self.covariances = [self._as_matrix(c) for c in covariances]
        if len(self.covariances) != self.k:
            raise ValueError("need one covariance per component")

        weights = np.ones(self.k) if weights is None else np.asarray(weights, dtype=float)
        if len(weights) != self.k:
            raise ValueError("need one weight per component")
        if np.any(weights < 0):
            raise ValueError("mixing weights must be non-negative")
        self.weights = weights / weights.sum()

        self._inverses = [np.linalg.inv(c) for c in self.covariances]
        self._norms = [
            1.0 / math.sqrt(((2 * math.pi) ** self.dimension) * max(np.linalg.det(c), 1e-300))
            for c in self.covariances
        ]

    def _as_matrix(self, covariance) -> np.ndarray:
        array = np.asarray(covariance, dtype=float)
        if array.ndim == 0:
            return np.eye(self.dimension) * float(array)
        if array.ndim == 1:
            return np.diag(array)
        return array

    def component_densities(self, points: np.ndarray) -> np.ndarray:
        """``(len(points), k)`` un-mixed component densities."""
        out = np.empty((len(points), self.k))
        for index in range(self.k):
            offsets = points - self.means[index]
            exponent = -0.5 * np.einsum(
                "ij,jk,ik->i", offsets, self._inverses[index], offsets
            )
            out[:, index] = self._norms[index] * np.exp(exponent)
        return out

    def __call__(self, points: np.ndarray, time: float = 0.0) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        return self.floor + self.component_densities(points) @ self.weights

    def responsibilities(self, points: np.ndarray) -> np.ndarray:
        """``(len(points), k)`` posterior component membership.

        This is what turns a mixture into a *task allocation*: the component an
        agent is most responsible for is the sub-region it should serve.
        """
        weighted = self.component_densities(points) * self.weights
        totals = weighted.sum(axis=1, keepdims=True)
        return np.divide(weighted, np.maximum(totals, 1e-300))

    def sample(self, count: int, rng=None) -> np.ndarray:
        """Draw points from the mixture (targets, waypoints, or a spawn)."""
        rng = rng or np.random.default_rng(0)
        choices = rng.choice(self.k, size=count, p=self.weights)
        out = np.empty((count, self.dimension))
        for index in range(self.k):
            mask = choices == index
            if not np.any(mask):
                continue
            out[mask] = rng.multivariate_normal(
                self.means[index], self.covariances[index], size=int(mask.sum())
            )
        return out

    @classmethod
    def fit(
        cls,
        points: np.ndarray,
        k: int = 3,
        iterations: int = 60,
        seed: int = 0,
        floor: float = 0.02,
        regularisation: float = 1e-6,
    ) -> "GaussianMixtureDensity":
        """Fit a mixture to observations by expectation-maximisation.

        This is the bridge from data to control: fly a survey, collect where the
        interesting readings were, fit a mixture, hand it to a coverage
        controller as the importance field. Plain EM with a small ridge on the
        covariances -- enough for the sample sizes swarm work produces, and it
        keeps the package dependency-free.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        n, dimension = points.shape
        if n < k:
            raise ValueError("need at least as many points as components")
        rng = np.random.default_rng(seed)

        means = points[rng.choice(n, size=k, replace=False)].copy()
        covariances = [np.cov(points.T) + regularisation * np.eye(dimension) for _ in range(k)]
        weights = np.full(k, 1.0 / k)

        for _ in range(iterations):
            model = cls(means, covariances, weights, floor=0.0)
            responsibilities = model.responsibilities(points)
            counts = responsibilities.sum(axis=0) + 1e-12

            weights = counts / n
            means = (responsibilities.T @ points) / counts[:, None]
            covariances = []
            for index in range(k):
                offsets = points - means[index]
                covariance = (
                    (responsibilities[:, index][:, None] * offsets).T @ offsets
                ) / counts[index]
                covariances.append(covariance + regularisation * np.eye(dimension))

        return cls(means, covariances, weights, floor=floor)

    def __repr__(self) -> str:
        return "GaussianMixtureDensity(k=%d, dim=%d)" % (self.k, self.dimension)


class TimeVaryingDensity(DensityField):
    """A density whose components move -- the realistic case.

    Targets do not hold still, and a coverage team tracking them is solving a
    different problem from one deploying over a static field: the controller
    never converges, it *pursues*. ``motion`` maps time to an offset applied to
    every component.
    """

    name = "time_varying"

    def __init__(self, base: DensityField, motion, period: Optional[float] = None):
        self.base = base
        self.motion = motion
        self.period = period

    def __call__(self, points: np.ndarray, time: float = 0.0) -> np.ndarray:
        if self.period:
            time = time % self.period
        offset = np.asarray(self.motion(time), dtype=float)
        return self.base(points - offset)

    def __repr__(self) -> str:
        return "TimeVaryingDensity(%r)" % self.base


class SampledDensity(DensityField):
    """Importance given as values at known points, interpolated by inverse distance.

    For a density that came from measurements or a map rather than a formula.
    """

    name = "sampled"

    def __init__(self, points: np.ndarray, values: np.ndarray, power: float = 2.0, floor: float = 0.02):
        self.points = np.atleast_2d(np.asarray(points, dtype=float))
        self.values = np.asarray(values, dtype=float)
        if len(self.points) != len(self.values):
            raise ValueError("need one value per point")
        self.power = power
        self.floor = floor

    def __call__(self, points: np.ndarray, time: float = 0.0) -> np.ndarray:
        distances = np.linalg.norm(
            np.atleast_2d(points)[:, None, :] - self.points[None, :, :], axis=2
        )
        weights = 1.0 / np.maximum(distances, 1e-6) ** self.power
        return self.floor + (weights @ self.values) / weights.sum(axis=1)


def balanced_assignment(
    responsibilities: np.ndarray,
    quota: np.ndarray,
    previous: Optional[np.ndarray] = None,
    stickiness: float = 0.6,
) -> np.ndarray:
    """Assign agents to mixture components, respecting each component's share.

    Plain ``argmax`` of the responsibilities does not split a team: agents that
    launch together are all nearest the same component, all pick it, and nothing
    ever moves them. Scaling the scores by a quota pressure does not fix it
    either -- responsibilities are near-degenerate (``1e-30`` versus ``1``), and
    scaling zero by anything is still zero.

    So this is a capacity-constrained allocation instead. Components get integer
    capacities from ``quota`` (largest-remainder rounding, summing to the fleet
    size); every (agent, component) pair is scored by log-responsibility, with a
    bonus for the agent's previous choice; pairs are then taken in descending
    score order, skipping agents already placed and components already full.
    Greedy rather than optimal, but deterministic, ``O(nk log nk)``, and it
    produces exactly the requested split.

    Note this rule needs the *current counts* -- one integer per component, which
    a swarm can broadcast cheaply. It is the one piece of shared state in this
    module, and it is what buys proportional allocation without an allocator.

    Args:
        responsibilities: ``(n, k)`` posterior membership.
        quota: ``(k,)`` desired agent count per component; scaled to the fleet.
        previous: last iteration's assignment, for hysteresis.
        stickiness: score bonus for staying put, damping oscillation.
    """
    scores = np.log(np.maximum(np.asarray(responsibilities, dtype=float), 1e-300))
    n, k = scores.shape
    if n == 0:
        return np.zeros(0, dtype=int)

    if previous is not None and len(previous) == n:
        scores = scores.copy()
        scores[np.arange(n), previous] += stickiness * np.abs(scores).max()

    # Integer capacities by largest remainder, so they sum to exactly n.
    quota = np.asarray(quota, dtype=float)
    share = quota / max(quota.sum(), 1e-12) * n
    capacity = np.floor(share).astype(int)
    for index in np.argsort(-(share - capacity))[: n - int(capacity.sum())]:
        capacity[index] += 1
    capacity = np.maximum(capacity, 0)

    order = np.argsort(-scores, axis=None)
    assignment = np.full(n, -1, dtype=int)
    remaining = capacity.copy()
    for flat in order:
        agent, component = divmod(int(flat), k)
        if assignment[agent] >= 0 or remaining[component] <= 0:
            continue
        assignment[agent] = component
        remaining[component] -= 1

    # Anyone left over (capacities can exhaust on ties) goes to their best fit.
    leftover = assignment < 0
    if np.any(leftover):
        assignment[leftover] = np.argmax(scores[leftover], axis=1)
    return assignment


DENSITIES = {
    "uniform": UniformDensity,
    "gaussian": GaussianDensity,
    "gmm": GaussianMixtureDensity,
}


def get_density(name, **kwargs) -> DensityField:
    if isinstance(name, DensityField):
        return name
    if callable(name) and not isinstance(name, type):
        # Plain callables stay supported: wrap them so everything downstream
        # can rely on the DensityField interface.
        class _Wrapped(DensityField):
            name = "callable"

            def __call__(self, points, time: float = 0.0):
                return np.asarray(name(points), dtype=float)

        return _Wrapped()
    try:
        return DENSITIES[str(name).lower()](**kwargs)
    except KeyError:
        raise ValueError(
            "Unknown density %r. Available: %s" % (name, ", ".join(sorted(DENSITIES)))
        )
