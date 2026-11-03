"""Joint trajectory optimisation for a fleet, in continuous space.

Every other planner in this library reasons about a *path*: a sequence of
vertices on a graph, timed afterwards. This one optimises the whole fleet's
**trajectories** directly -- vector-valued polynomials of time in R^n, with
no grid anywhere -- and it optimises them *jointly*: one problem whose
decision variables are every vehicle's polynomial coefficients at once, whose
objective is the fleet's total effort, and whose constraints couple the
vehicles to each other.

    minimise    Σ_i ∫ ‖p_i⁽ᵈ⁾(t)‖² dt                    (snap, by default)
    subject to  p_i(0) = start_i,  p_i(T) = goal_i,      and derivatives at rest
                p_i is C^k across its knots
                ‖p_i(t) - p_j(t)‖ >= r_i + r_j           for all pairs, all t
                ‖p_i(t) - o‖     >= r_i + r_o            for every obstacle
                ‖ṗ_i(t)‖ <= v_max,  ‖p̈_i(t)‖ <= a_max

The cost and the equalities are convex; the separation constraints are not
-- "stay at least this far apart" is the complement of a ball. The standard
remedy, and the one used here, is **sequential convex programming**: at each
iteration, replace each separation constraint by the half-space that
supports it at the current iterate,

    n̂ᵀ(p_i(t) - p_j(t)) >= r_i + r_j,     n̂ = (p_i(t) - p_j(t)) / ‖·‖

which is linear, and *conservative* -- a point satisfying it satisfies the
true constraint, since the half-space is contained in the feasible set. Solve
the resulting convex program, re-linearise, repeat, with a trust region so
the linearisation stays valid. That is Augugliaro, Schoellig and D'Andrea's
formulation for quadrocopter fleets (IROS 2012), and Chen, Cutler and How's
(ICRA 2015) for the same idea with a discrete plan as the seed.

Three things are worth knowing about this implementation.

**The equalities are eliminated once, not solved every iteration.** Boundary
conditions and continuity do not change between iterations, so the affine
set they define is factored once into a particular solution and an
orthonormal null-space basis; every subproblem is then solved in the reduced
coordinates. On a six-vehicle instance that turns a 816-dimensional
saddle-point system into a 144-dimensional one, which is the difference
between seconds and milliseconds per iteration.

**Speed and acceleration are kept inside the limits, then met exactly by
dilating time.** The natural linearisation of ``‖ṗ‖ <= v_max`` -- the tangent
plane at the current heading -- is an *outer* approximation (the bound is
convex, so its tangent plane lies outside it), and the optimiser exploits it:
the plane caps the speed along the old heading and nothing to the side. So the
limits are written as a polytope *inscribed* in the ball instead, with a
vertex on the current heading -- conservative in the same way the separation
half-spaces are, and still exact along the direction of flight. That holds at
the sample times; between them the speed can peak a hair over, so the result
is time-dilated by the smallest factor that brings both within bounds. Dilation
is safe in a way no other repair is: it scales every vehicle's clock by the
same factor, so the fleet's geometry at corresponding times -- and therefore
every pairwise separation -- is exactly preserved.

**A discrete plan makes a good seed -- for the route.**
:func:`waypoints_from_solution` turns any :class:`~pymapf.Solution` into the
waypoint lists this optimiser starts from, which is how a fleet gets a
*globally* sensible route (which way round each other and round the walls)
from a complete MAPF solver before the continuous optimiser makes it smooth
and fast. Without a seed the initial guess is a straight line, and the
optimiser will find a way around convex obstacles but has no way to reason
about which way *around* is better. Two things do not carry over. The discrete
plan's waits are dropped and knot times come from leg lengths, so who goes
first is the optimiser's decision. And the grid's blocked cells are not
constraints here: the waypoints are not pinned, so a smoothed corner can cut
across a blocked cell. Only the spheres passed as ``obstacles`` are avoided.
"""

from __future__ import annotations

import functools
import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .polynomial import (
    PiecewisePolynomial,
    _time_allocation,
    cost_matrix,
    derivative_row,
)
from .qp import solve_qp

__all__ = [
    "JointPlan",
    "Vehicle",
    "plan_joint_trajectories",
    "waypoints_from_solution",
]

Point = Sequence[float]


@dataclass(frozen=True)
class Vehicle:
    """One vehicle: where it starts, where it must end, and how big it is.

    Args:
        name: unique.
        start / goal: points in R^n, in metres. Every vehicle in a fleet must
            have the same dimension; 2 and 3 are the useful ones.
        radius: the vehicle's own radius. Two vehicles must stay
            ``radius_i + radius_j`` apart, so a fleet of unequal vehicles
            keeps unequal distances, which is right.
        v_max / a_max: per-vehicle limits; the fleet's are used when these
            are ``None``.
    """

    name: str
    start: Point
    goal: Point
    radius: float = 0.25
    v_max: Optional[float] = None
    a_max: Optional[float] = None

    def __post_init__(self):
        if self.radius < 0:
            raise ValueError("radius cannot be negative")
        object.__setattr__(self, "start", tuple(float(x) for x in self.start))
        object.__setattr__(self, "goal", tuple(float(x) for x in self.goal))
        if len(self.start) != len(self.goal):
            raise ValueError("%r: start and goal have different dimensions" % self.name)


# --------------------------------------------------------------------------
# the result
# --------------------------------------------------------------------------


@dataclass
class JointPlan:
    """The optimised trajectories and what they actually achieve.

    ``trajectories[name]`` is a :class:`~pymapf.trajectory.PiecewisePolynomial`
    over ``[0, duration]``; call it with a time in seconds. ``cost`` is the
    objective of those trajectories as returned -- ``½ Σ_i ∫ ‖p_i⁽ᵈ⁾‖² dt``
    for the ``minimise``-th derivative, after any time dilation -- while
    ``history`` records the optimiser's own iterates, before it.
    """

    trajectories: Dict[str, PiecewisePolynomial]
    vehicles: List[Vehicle]
    duration: float
    cost: float
    iterations: int
    converged: bool
    obstacles: Tuple[Tuple[Tuple[float, ...], float], ...] = ()
    clearance: float = 0.0
    v_max: float = math.inf
    a_max: float = math.inf
    time_scale: float = 1.0
    history: List[Dict[str, float]] = field(default_factory=list)

    # -- what it achieved ------------------------------------------------
    def times(self, dt: float = 0.02) -> np.ndarray:
        return np.arange(0.0, self.duration + 0.5 * dt, dt)

    def positions(self, dt: float = 0.02) -> Dict[str, np.ndarray]:
        times = self.times(dt)
        return {name: p.sample(times) for name, p in self.trajectories.items()}

    def required_separation(self, a: str, b: str) -> float:
        radii = {v.name: v.radius for v in self.vehicles}
        return radii[a] + radii[b] + self.clearance

    def min_separation(self, dt: float = 0.02) -> Tuple[float, float, Tuple[str, str]]:
        """Closest approach over the whole flight: ``(distance, time, pair)``."""
        names = list(self.trajectories)
        if len(names) < 2:
            return math.inf, 0.0, ("", "")
        times = self.times(dt)
        points = np.stack([self.trajectories[n].sample(times) for n in names])
        best = (math.inf, 0.0, (names[0], names[1]))
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                distances = np.linalg.norm(points[i] - points[j], axis=1)
                k = int(np.argmin(distances))
                if distances[k] < best[0]:
                    best = (float(distances[k]), float(times[k]), (names[i], names[j]))
        return best

    def separation_margin(self, dt: float = 0.02) -> float:
        """How much closer than allowed the closest pair came; ``<= 0`` is safe."""
        names = list(self.trajectories)
        if len(names) < 2:
            return -math.inf
        times = self.times(dt)
        points = {n: self.trajectories[n].sample(times) for n in names}
        worst = -math.inf
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                distances = np.linalg.norm(points[a] - points[b], axis=1)
                worst = max(worst, self.required_separation(a, b) - distances.min())
        return float(worst)

    def obstacle_margin(self, dt: float = 0.02) -> float:
        """How far any vehicle penetrated an obstacle; ``<= 0`` is safe."""
        if not self.obstacles:
            return -math.inf
        times = self.times(dt)
        worst = -math.inf
        radii = {v.name: v.radius for v in self.vehicles}
        for name, trajectory in self.trajectories.items():
            points = trajectory.sample(times)
            for centre, radius in self.obstacles:
                distances = np.linalg.norm(points - np.asarray(centre), axis=1)
                worst = max(
                    worst, radius + radii[name] + self.clearance - distances.min()
                )
        return float(worst)

    def max_speed(self, dt: float = 0.02) -> float:
        times = self.times(dt)
        return max(
            float(np.linalg.norm(t.sample(times, 1), axis=1).max())
            for t in self.trajectories.values()
        )

    def max_acceleration(self, dt: float = 0.02) -> float:
        times = self.times(dt)
        return max(
            float(np.linalg.norm(t.sample(times, 2), axis=1).max())
            for t in self.trajectories.values()
        )

    def is_safe(self, dt: float = 0.02) -> bool:
        """Separation and obstacle clearance both hold over the sampled flight."""
        return self.separation_margin(dt) <= 1e-9 and self.obstacle_margin(dt) <= 1e-9

    def path_length(self, name: str) -> float:
        return self.trajectories[name].path_length()

    def summary(self) -> Dict[str, float]:
        separation, when, _pair = self.min_separation()
        return {
            "vehicles": len(self.vehicles),
            "duration": self.duration,
            "cost": self.cost,
            "iterations": self.iterations,
            "converged": self.converged,
            "min_separation": separation,
            "min_separation_at": when,
            "separation_margin": self.separation_margin(),
            "max_speed": self.max_speed(),
            "max_acceleration": self.max_acceleration(),
            "time_scale": self.time_scale,
            "path_length": sum(t.path_length() for t in self.trajectories.values()),
        }

    def __repr__(self) -> str:
        return "JointPlan(vehicles=%d, duration=%.2fs, min_separation=%.3f)" % (
            len(self.vehicles),
            self.duration,
            self.min_separation()[0],
        )


# --------------------------------------------------------------------------
# the variable layout
# --------------------------------------------------------------------------


class _Layout:
    """Where each vehicle's coefficients live in the flat variable vector.

    One block per vehicle, inside it ``[segment][dimension][coefficient]``.
    Everything downstream -- the cost, the equalities, the sampled
    inequalities -- writes into slices this class hands out, so the ordering
    is stated in exactly one place.
    """

    def __init__(
        self,
        names: Sequence[str],
        segments: Mapping[str, int],
        dimension: int,
        order: int,
    ):
        self.names = list(names)
        self.segments = dict(segments)
        self.dimension = dimension
        self.order = order
        self.width = order + 1
        self.base: Dict[str, int] = {}
        offset = 0
        for name in self.names:
            self.base[name] = offset
            offset += self.segments[name] * dimension * self.width
        self.size = offset

    def slice(self, name: str, segment: int, dim: int) -> slice:
        start = self.base[name] + (segment * self.dimension + dim) * self.width
        return slice(start, start + self.width)

    def unpack(self, x: np.ndarray, name: str) -> np.ndarray:
        block = x[
            self.base[name] : self.base[name]
            + self.segments[name] * self.dimension * self.width
        ]
        return block.reshape(self.segments[name], self.dimension, self.width)


def _locate(
    knots: np.ndarray, durations: Sequence[float], t: float
) -> Tuple[int, float]:
    """``(segment, normalised time)`` -- the evaluation rule, shared."""
    total = knots[-1]
    if t <= 0.0:
        return 0, 0.0
    if t >= total:
        return len(durations) - 1, 1.0
    k = int(np.searchsorted(knots, t, side="right") - 1)
    k = min(max(k, 0), len(durations) - 1)
    return k, (t - knots[k]) / durations[k]


# --------------------------------------------------------------------------
# the equality set, eliminated once
# --------------------------------------------------------------------------


class _Affine:
    """The affine set ``Ax = b`` as ``x = particular + basis @ z``.

    The boundary conditions and the continuity of every derivative are the
    same at every iteration of the outer loop, so they are factored once:
    ``basis`` is an orthonormal basis of the null space of ``A`` (from the
    SVD) and ``particular`` is the minimum-norm solution. Optimising over
    ``z`` then satisfies them identically, and the subproblems shrink by the
    number of equalities -- which is most of the problem.
    """

    def __init__(self, a: np.ndarray, b: np.ndarray, size: int):
        if a.size == 0:
            self.particular = np.zeros(size)
            self.basis = np.eye(size)
            self.rank = 0
            return
        # Equilibrate the rows first. A derivative row carries 1/T^d, so on
        # a short segment the jerk rows outweigh the position rows by many
        # orders of magnitude, the rank test below mistakes the position
        # rows for noise, and a perfectly good 1 cm hop was reported as
        # "inconsistent". Scaling rows changes neither the solution set nor
        # its minimum-norm point.
        norms = np.linalg.norm(a, axis=1)
        norms[norms == 0.0] = 1.0
        a = a / norms[:, None]
        b = b / norms
        u, singular, vt = np.linalg.svd(a, full_matrices=True)
        tol = max(a.shape) * (singular[0] if singular.size else 0.0) * 1e-12
        rank = int((singular > tol).sum())
        self.rank = rank
        self.basis = vt[rank:].T  # columns span the null space, orthonormal
        inverse = np.zeros_like(singular)
        inverse[:rank] = 1.0 / singular[:rank]
        self.particular = vt[: len(singular)].T @ (inverse * (u.T @ b)[: len(singular)])
        residual = a @ self.particular - b
        if np.linalg.norm(residual) > 1e-6 * max(1.0, np.linalg.norm(b)):
            raise ValueError(
                "the boundary and continuity conditions are inconsistent; "
                "check that every vehicle has as many segments as it needs "
                "for the derivatives asked of it"
            )

    @property
    def dimension(self) -> int:
        return self.basis.shape[1]

    def to_full(self, z: np.ndarray) -> np.ndarray:
        return self.particular + self.basis @ z

    def to_reduced(self, x: np.ndarray) -> np.ndarray:
        return self.basis.T @ (x - self.particular)


# --------------------------------------------------------------------------
# the planner
# --------------------------------------------------------------------------


def waypoints_from_solution(
    solution,
    positions: Optional[Mapping[object, Sequence[float]]] = None,
    cell_size: float = 1.0,
    simplify: bool = True,
) -> Dict[str, np.ndarray]:
    """Waypoints in metres from a discrete :class:`~pymapf.Solution`.

    Args:
        solution: any valid plan, on a grid, a volume or an explicit graph.
        positions: ``{vertex: point}`` when the vertices are not themselves
            coordinates; grid cells are, scaled by ``cell_size``.
        simplify: drop waits, and waypoints that lie on the straight line
            between their neighbours. A grid path has a waypoint per cell,
            and pinning the optimiser's knots to every one of them wastes
            segments on a corridor where one would do; the corners are what
            carry the routing decision.

    Use the result as ``waypoints=`` to :func:`plan_joint_trajectories`.
    The result is geometry only: waits are not represented, and nothing
    here tells the optimiser which cells were blocked -- see the module
    docstring.
    """
    out: Dict[str, np.ndarray] = {}
    for name, path in solution.paths.items():
        if positions is not None:
            points = [np.asarray(positions[v], dtype=float) for v in path]
        else:
            points = [np.asarray(v, dtype=float) * cell_size for v in path]
        if simplify:
            # A wait repeats a vertex; drop the repeats before looking for
            # corners. The continuous plan decides its own timing, and a
            # collinearity test straddling a wait would skip the corner the
            # agent happened to wait on.
            distinct = [points[0]]
            for point in points[1:]:
                if np.linalg.norm(point - distinct[-1]) > 1e-12:
                    distinct.append(point)
            points = distinct
        if len(points) == 1:
            # Already at its goal: a leg of length zero, so that it still
            # has a trajectory -- parked -- for the others to avoid.
            out[name] = np.array([points[0], points[0]])
            continue
        kept = [points[0]]
        for previous, current, following in zip(points, points[1:], points[2:]):
            if not simplify:
                kept.append(current)
                continue
            before = current - previous
            after = following - current
            before /= np.linalg.norm(before)
            after /= np.linalg.norm(after)
            if np.linalg.norm(before - after) > 1e-9:
                kept.append(current)
        kept.append(points[-1])
        out[name] = np.array(kept)
    return out


def _default_waypoints(vehicle: Vehicle, segments: int) -> np.ndarray:
    """The straight line from start to goal, split into ``segments`` legs."""
    start = np.asarray(vehicle.start, dtype=float)
    goal = np.asarray(vehicle.goal, dtype=float)
    fractions = np.linspace(0.0, 1.0, segments + 1)[:, None]
    return start + fractions * (goal - start)


def _equalities(
    layout: _Layout,
    vehicles: Sequence[Vehicle],
    durations: Mapping[str, List[float]],
    boundary: int,
    continuity: int,
    pinned: Optional[Mapping[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Boundary conditions, continuity across knots, and optional pinning.

    ``boundary`` is how many derivatives are held at zero at each end (1 =
    at rest, 2 = also zero acceleration, ...); ``continuity`` is the highest
    derivative that must match across an internal knot.
    """
    rows: List[np.ndarray] = []
    rhs: List[float] = []
    order = layout.order
    for vehicle in vehicles:
        name = vehicle.name
        times = durations[name]
        last = layout.segments[name] - 1
        for dim in range(layout.dimension):
            # -- start: position, then derivatives at rest
            for d in range(boundary + 1):
                row = np.zeros(layout.size)
                row[layout.slice(name, 0, dim)] = derivative_row(
                    order, times[0], 0.0, d
                )
                rows.append(row)
                rhs.append(vehicle.start[dim] if d == 0 else 0.0)
            # -- goal
            for d in range(boundary + 1):
                row = np.zeros(layout.size)
                row[layout.slice(name, last, dim)] = derivative_row(
                    order, times[last], 1.0, d
                )
                rows.append(row)
                rhs.append(vehicle.goal[dim] if d == 0 else 0.0)
            # -- continuity at every internal knot
            for k in range(last):
                for d in range(continuity + 1):
                    row = np.zeros(layout.size)
                    row[layout.slice(name, k, dim)] = derivative_row(
                        order, times[k], 1.0, d
                    )
                    row[layout.slice(name, k + 1, dim)] = -derivative_row(
                        order, times[k + 1], 0.0, d
                    )
                    rows.append(row)
                    rhs.append(0.0)
            # -- optional: pin the interior knots to the seed's waypoints
            if pinned is not None:
                points = pinned[name]
                for k in range(1, last + 1):
                    row = np.zeros(layout.size)
                    row[layout.slice(name, k, dim)] = derivative_row(
                        order, times[k], 0.0, 0
                    )
                    rows.append(row)
                    rhs.append(float(points[k][dim]))
    return np.array(rows), np.array(rhs)


def _hessian(
    layout: _Layout, durations: Mapping[str, List[float]], minimise: int
) -> np.ndarray:
    """Block-diagonal cost: the integral of the squared ``minimise``-th
    derivative, summed over vehicles, dimensions and segments."""
    h = np.zeros((layout.size, layout.size))
    for name in layout.names:
        for k, duration in enumerate(durations[name]):
            block = cost_matrix(layout.order, duration, minimise)
            for dim in range(layout.dimension):
                s = layout.slice(name, k, dim)
                h[s, s] += block
    return h


class _Sampler:
    """Rows that evaluate a vehicle's position (or a derivative) at a time.

    Building a constraint row means writing one basis row into one slice per
    dimension. The basis row depends only on the vehicle, the time and the
    derivative -- not on the direction the constraint is projected onto --
    so they are computed once per iteration and reused across every pair,
    obstacle and limit that shares them.
    """

    def __init__(
        self,
        layout: _Layout,
        durations: Mapping[str, List[float]],
        times: np.ndarray,
        derivatives: Sequence[int],
    ):
        self.layout = layout
        self.times = times
        self.rows: Dict[Tuple[str, int], List[Tuple[int, np.ndarray]]] = {}
        for name in layout.names:
            segment_times = durations[name]
            knots = np.cumsum([0.0] + list(segment_times))
            for d in derivatives:
                entries = []
                for t in times:
                    k, s = _locate(knots, segment_times, float(t))
                    entries.append(
                        (k, derivative_row(layout.order, segment_times[k], s, d))
                    )
                self.rows[(name, d)] = entries

    def write(
        self,
        row: np.ndarray,
        name: str,
        index: int,
        direction: np.ndarray,
        derivative: int,
        scale: float = 1.0,
    ) -> None:
        """Add ``scale * directionᵀ (derivative of p_name at times[index])``."""
        k, basis = self.rows[(name, derivative)][index]
        for dim in range(self.layout.dimension):
            if direction[dim]:
                row[self.layout.slice(name, k, dim)] += scale * direction[dim] * basis

    def values(self, x: np.ndarray, name: str, derivative: int) -> np.ndarray:
        """``(samples, dimension)`` values of a derivative, from a full vector."""
        block = self.layout.unpack(x, name)
        entries = self.rows[(name, derivative)]
        return np.array([block[k] @ basis for k, basis in entries])


def plan_joint_trajectories(
    vehicles: Sequence[Vehicle],
    duration: Optional[float] = None,
    segments: int = 4,
    order: int = 7,
    minimise: int = 4,
    v_max: float = 2.0,
    a_max: float = 4.0,
    obstacles: Sequence[Tuple[Point, float]] = (),
    clearance: float = 0.0,
    waypoints: Optional[Mapping[str, np.ndarray]] = None,
    samples: int = 60,
    iterations: int = 20,
    trust_region: float = 2.0,
    boundary: int = 3,
    continuity: int = 4,
    tolerance: float = 1e-4,
    cost_tolerance: float = 1e-3,
    inter_sample: bool = True,
    enforce_limits: bool = True,
) -> JointPlan:
    """Optimise every vehicle's trajectory at once, in continuous space.

    Args:
        vehicles: the fleet. All must share a dimension.
        duration: the fleet's makespan in seconds. ``None`` sizes it from the
            longest seed path at ``0.7 * v_max``, leaving room to accelerate
            and to detour.
        segments: polynomial pieces per vehicle. With waypoints each leg is
            a segment, and the longest legs are halved until there are at
            least this many: a single straight leg leaves the optimiser
            nothing to work with.
        order: polynomial degree per piece. 7 is the classical choice for a
            snap objective -- eight coefficients, enough for position,
            velocity, acceleration and jerk at both ends of a segment.
        minimise: the derivative whose square is integrated. 4 is snap
            (Mellinger and Kumar's objective, the one a quadrotor's inputs
            are flat in), 3 is jerk, 2 acceleration.
        v_max / a_max: fleet limits; a vehicle may override either.
        obstacles: ``(centre, radius)`` spheres to stay clear of.
        clearance: extra metres required beyond the radii, for both
            vehicle-vehicle and vehicle-obstacle separation.
        waypoints: ``{name: points}`` to seed from -- see
            :func:`waypoints_from_solution`. Interior waypoints are *not*
            pinned in the optimisation; they set the time allocation and the
            initial guess, and the optimiser is free to leave them. Only the
            geometry is used: knot times come from leg lengths, so a wait in
            the discrete plan does not survive into the seed.
        samples: times at which the constraints are enforced, over the whole
            flight. The guarantee is at these times; the result's
            :meth:`JointPlan.min_separation` samples much more finely, and
            the tests check that the two agree.
        iterations: outer convex iterations.
        trust_region: how far the coefficients may move per iteration,
            shrunk when a step makes the true constraints worse.
        boundary: derivatives held at zero at the start and the goal.
        continuity: highest derivative matched across an internal knot.
        tolerance: convergence, on the worst constraint violation in metres.
        cost_tolerance: convergence, on the relative change in the objective
            between iterations; two settled iterations in a row stop the loop.
        inter_sample: require the separation constraints to hold *between*
            the sample times too, by the bound in :func:`_linearise`. Turning
            it off enforces them only at the samples, which is cheaper and
            what the literature usually reports -- and lets a pair dip below
            the requirement between two samples, which the tests demonstrate.
        enforce_limits: dilate time at the end so that the speed and
            acceleration limits hold exactly. See the module docstring for
            why this is safe.

    Returns:
        A :class:`JointPlan`.

    Raises:
        ValueError: on an inconsistent fleet (mixed dimensions, duplicate
            names, a vehicle starting inside another) or an impossible
            polynomial (fewer coefficients than boundary conditions).
    """
    fleet = list(vehicles)
    obstacles = tuple((tuple(float(x) for x in c), float(r)) for c, r in obstacles)
    dimension = _validate(fleet, obstacles, order, boundary, segments, clearance)
    names = [v.name for v in fleet]

    seeds, durations, duration = _seed(
        fleet, waypoints, segments, dimension, duration, v_max
    )
    layout = _Layout(names, {n: len(durations[n]) for n in names}, dimension, order)
    _check_freedom(layout, order, boundary, continuity)

    hessian = _hessian(layout, durations, minimise)
    gradient = np.zeros(layout.size)

    times = np.linspace(0.0, duration, samples)
    sampler = _Sampler(layout, durations, times, derivatives=(0, 1, 2))
    limits = {
        v.name: (
            v.v_max if v.v_max is not None else v_max,
            v.a_max if v.a_max is not None else a_max,
        )
        for v in fleet
    }

    # -- the initial guess: minimum-snap through the seed's waypoints --------
    pinned_a, pinned_b = _equalities(
        layout, fleet, durations, boundary, continuity, pinned=seeds
    )
    guess = solve_qp(hessian, gradient, pinned_a, pinned_b).x

    # -- the affine set the optimiser really lives on ------------------------
    a, b = _equalities(layout, fleet, durations, boundary, continuity)
    affine = _Affine(a, b, layout.size)
    guess = _within_limits(guess, a, b, sampler, layout, names, limits)
    z = affine.to_reduced(guess)

    # The true violation is measured on a finer grid than the one the
    # constraints are written on, so that "converged" means what
    # JointPlan.is_safe() means and not merely "feasible where we looked".
    # Without the inter-sample bound there is no claim about the times in
    # between, so the check is made where the constraints actually are --
    # anything else would report a failure the formulation never promised
    # to avoid, and would make the optimiser chase it.
    checker = (
        _Sampler(
            layout,
            durations,
            np.linspace(0.0, duration, max(4 * samples, 200)),
            derivatives=(0,),
        )
        if inter_sample
        else sampler
    )
    radii = {v.name: v.radius for v in fleet}

    reduced_h = affine.basis.T @ hessian @ affine.basis
    reduced_g = affine.basis.T @ (hessian @ affine.particular + gradient)

    def violation(x: np.ndarray) -> float:
        """The worst true (not linearised) constraint violation, in metres."""
        worst = 0.0
        points = {name: checker.values(x, name, 0) for name in names}
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                need = radii[first] + radii[second] + clearance
                distance = np.linalg.norm(points[first] - points[second], axis=1)
                worst = max(worst, float(need - distance.min()))
            for centre, radius in obstacles:
                distance = np.linalg.norm(points[first] - np.asarray(centre), axis=1)
                worst = max(
                    worst, float(radius + radii[first] + clearance - distance.min())
                )
        return worst

    x, history, used, converged, current = _iterate(
        affine,
        z,
        violation,
        reduced_h,
        reduced_g,
        hessian,
        _linearise,
        dict(
            sampler=sampler,
            layout=layout,
            names=names,
            radii=radii,
            limits=limits,
            obstacles=obstacles,
            clearance=clearance,
            affine=affine,
            inter_sample=inter_sample,
        ),
        iterations,
        trust_region,
        tolerance,
        cost_tolerance,
    )
    trajectories = {
        name: PiecewisePolynomial(tuple(durations[name]), layout.unpack(x, name))
        for name in names
    }
    plan = JointPlan(
        trajectories=trajectories,
        vehicles=fleet,
        duration=duration,
        cost=float(0.5 * x @ hessian @ x),
        iterations=used,
        converged=converged and current <= tolerance,
        obstacles=obstacles,
        clearance=clearance,
        v_max=v_max,
        a_max=a_max,
        history=history,
    )
    if enforce_limits:
        _dilate(plan, limits)
        # Stretching time by f scales the d-th derivative by f^-d and the
        # interval by f, so the objective of what is returned is the
        # optimiser's times f^(1 - 2d).
        plan.cost *= plan.time_scale ** (1 - 2 * minimise)
    return plan


def _within_limits(guess, a, b, sampler, layout, names, limits) -> np.ndarray:
    """The nearest trajectory to ``guess`` that keeps the speed and
    acceleration limits at the samples, on the affine set ``Ax = b``.

    The minimum-snap curve through a seed's waypoints is usually faster than
    the limits allow -- 2.9 m/s on a straight 10 m flight limited to 2 -- and
    the first trust region is too small to slow it down that much. That
    subproblem is then infeasible, and the compromise a solver returns for an
    infeasible problem is not even symmetric: the lone straight-line vehicle
    came out weaving 0.6 m from side to side, and no later iteration undid
    it. So the guess is first projected onto the limits (a convex problem:
    the rows of :func:`_limit_rows` at the guess's own headings). From a
    point inside them every later iterate satisfies its own limit rows, and
    the trust region never has to choose between the two. The pinned
    waypoints are released here -- they are what makes a fast seed fast --
    but staying as close as possible to the guess keeps its shape. If the
    limits cannot be met at all in the time given, the guess is returned
    unchanged and the dilation at the end does the work.
    """
    velocities = {name: sampler.values(guess, name, 1) for name in names}
    accelerations = {name: sampler.values(guess, name, 2) for name in names}
    default = np.zeros(layout.dimension)
    default[0] = 1.0
    rows: List[np.ndarray] = []
    upper: List[float] = []
    for name in names:
        _limit_rows(
            rows,
            upper,
            sampler,
            layout,
            name,
            velocities,
            accelerations,
            limits,
            _unit,
            default,
        )
    if not rows:
        return guess
    nearest = solve_qp(
        np.eye(layout.size),
        -guess,
        a,
        b,
        np.array(rows),
        np.array(upper),
        x0=guess,
    )
    return nearest.x if nearest.converged else guess


def _unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return fallback
    return vector / norm


def _iterate(
    affine,
    z,
    violation,
    reduced_h,
    reduced_g,
    hessian,
    linearise,
    arguments,
    iterations,
    trust_region,
    tolerance,
    cost_tolerance,
):
    """The sequential convex loop: linearise, solve, accept or shrink.

    Returns the best iterate found rather than the last. Re-linearising turns
    the normals slightly every iteration, so the tail of a run oscillates
    between feasible and just-barely-infeasible points of nearly equal cost;
    returning the last one would make the result depend on where the
    iteration budget happened to stop.
    """
    history: List[Dict[str, float]] = []
    x = affine.to_full(z)
    current = violation(x)
    radius = trust_region
    previous_cost = math.inf
    stable = 0
    converged = False
    used = 0
    best_key = (math.inf, math.inf)
    best_x = x.copy()
    identity = np.eye(affine.dimension)
    for iteration in range(iterations):
        used = iteration + 1
        rows, upper = linearise(x, **arguments)
        rows = (
            np.vstack([rows, identity, -identity])
            if len(rows)
            else np.vstack([identity, -identity])
        )
        upper = np.concatenate([upper, z + radius, -z + radius])
        candidate_z = solve_qp(reduced_h, reduced_g, None, None, rows, upper, x0=z).x
        candidate_x = affine.to_full(candidate_z)
        candidate = violation(candidate_x)
        cost = float(0.5 * candidate_x @ hessian @ candidate_x)
        history.append(
            {
                "iteration": iteration,
                "violation": candidate,
                "step": float(np.linalg.norm(candidate_z - z)),
                "trust_region": radius,
                "cost": cost,
            }
        )
        if candidate > current + 1e-9 and radius > 1e-3:
            radius *= 0.5  # the linearisation was trusted too far
            continue
        settled = abs(cost - previous_cost) <= cost_tolerance * max(1.0, abs(cost))
        z, x, current = candidate_z, candidate_x, candidate
        previous_cost = cost
        key = (max(current - tolerance, 0.0), cost)
        if key < best_key:
            best_key, best_x = key, x.copy()
        radius = min(trust_region, radius * 1.4)
        # Converged when the trajectories are feasible and the objective has
        # stopped moving. The step size is not the test: re-linearising turns
        # the normals a little every iteration, so the coefficients keep
        # drifting long after the cost, and the shape, have settled.
        if current <= tolerance and settled:
            stable += 1
            if stable >= 2:
                converged = True
                break
        else:
            stable = 0
    else:
        converged = current <= tolerance
    return best_x, history, used, converged, violation(best_x)


def _check_endpoints(fleet, clearance) -> None:
    """Two vehicles that already overlap where they start or finish.

    No trajectory can fix that, and reporting it here beats letting the
    optimiser fail to converge on a problem that has no solution.
    """
    for i, a in enumerate(fleet):
        for b in fleet[i + 1 :]:
            gap = a.radius + b.radius + clearance
            for attribute in ("start", "goal"):
                distance = math.dist(getattr(a, attribute), getattr(b, attribute))
                if distance < gap - 1e-9:
                    raise ValueError(
                        "%s and %s are %.3f apart at their %s but need %.3f; no "
                        "trajectory can fix an infeasible endpoint"
                        % (a.name, b.name, distance, attribute, gap)
                    )


def _seed(fleet, waypoints, segments, dimension, duration, v_max):
    """Waypoints, the fleet's makespan and each vehicle's segment times."""
    seeds = {
        v.name: (
            np.asarray(waypoints[v.name], dtype=float)
            if waypoints is not None and v.name in waypoints
            else _default_waypoints(v, segments)
        )
        for v in fleet
    }
    for name, points in seeds.items():
        if points.ndim != 2 or points.shape[1] != dimension:
            raise ValueError("waypoints for %r have the wrong shape" % name)
        if len(points) < 2:
            raise ValueError("waypoints for %r need at least two points" % name)
        seeds[name] = _split_legs(points, segments)
    if duration is None:
        longest = max(
            float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            for points in seeds.values()
        )
        if longest <= 0.0:
            # Nobody moves, so any makespan is as good as another. (This
            # used to floor at a millisecond, and the derivative rows of a
            # 1 ms segment are too ill-conditioned for the equalities to be
            # solved: a parked fleet was reported as "inconsistent".)
            duration = 1.0
        elif not 0.0 < v_max < math.inf:
            raise ValueError(
                "duration=None sizes the flight from v_max, which must be "
                "finite and positive; pass a duration"
            )
        else:
            # Cruising at 70% of the limit leaves room to accelerate into
            # the corners and to detour around whoever is in the way.
            duration = longest / (0.7 * v_max)
    elif not 0.0 < duration < math.inf:
        raise ValueError("duration must be positive and finite")
    durations = {
        name: _time_allocation(points, duration, v_max)
        for name, points in seeds.items()
    }
    return seeds, durations, duration


def _split_legs(points: np.ndarray, pieces: int) -> np.ndarray:
    """Halve the longest leg until there are at least ``pieces`` of them.

    A seed from a grid plan is simplified to its corners, so a vehicle that
    flies straight gets a single leg -- and one segment of order 7 has eight
    coefficients per axis, every one of them fixed by the eight boundary
    conditions. Such a vehicle cannot swerve, slow down or wait: two of them
    crossing a 5x5 grid finished 0.024 m apart for a 0.6 m requirement, and
    nothing said why. The extra knots only shape the initial guess and the
    time allocation; they are not pinned.
    """
    points = [np.asarray(point, dtype=float) for point in points]
    while len(points) - 1 < pieces:
        legs = [np.linalg.norm(b - a) for a, b in zip(points, points[1:])]
        k = int(np.argmax(legs))
        points.insert(k + 1, 0.5 * (points[k] + points[k + 1]))
    return np.array(points)


def _check_freedom(layout, order, boundary, continuity) -> None:
    """Refuse a vehicle whose trajectory the equalities fix completely.

    Each segment has ``order + 1`` coefficients per axis; the two ends take
    ``boundary + 1`` conditions each and every internal knot ``continuity +
    1``. If that leaves nothing, the optimiser can move the vehicle nowhere,
    and would report a collision it was never able to avoid.
    """
    for name, pieces in layout.segments.items():
        coefficients = pieces * (order + 1)
        conditions = 2 * (boundary + 1) + (pieces - 1) * (continuity + 1)
        if coefficients <= conditions:
            raise ValueError(
                "%r has no freedom left: %d segment(s) of order %d have %d "
                "coefficients per axis and its boundary and continuity "
                "conditions fix all of them; use more segments or a higher "
                "order" % (name, pieces, order, coefficients)
            )


def _validate(fleet, obstacles, order, boundary, segments, clearance) -> int:
    """Everything that makes an instance impossible before any solving."""
    if not fleet:
        raise ValueError("no vehicles")
    names = [v.name for v in fleet]
    if len(set(names)) != len(names):
        raise ValueError("vehicle names must be unique")
    dimension = len(fleet[0].start)
    if any(len(v.start) != dimension for v in fleet):
        raise ValueError("every vehicle must have the same dimension")
    if order + 1 < 2 * (boundary + 1):
        raise ValueError(
            "order %d gives %d coefficients per segment, too few for %d boundary "
            "conditions at each end" % (order, order + 1, boundary + 1)
        )
    if segments < 1:
        raise ValueError("segments must be at least 1")
    for centre, _radius in obstacles:
        if len(centre) != dimension:
            raise ValueError("an obstacle centre has the wrong dimension")
    _check_endpoints(fleet, clearance)
    return dimension


def _dilate(plan: JointPlan, limits: Mapping[str, Tuple[float, float]]) -> None:
    """Slow the whole fleet down until every vehicle's limits hold, in place.

    One factor for the fleet, not one per vehicle: dilating vehicles by
    different factors would change who is where when, and the separation the
    optimiser established would no longer mean anything. So the factor is
    the worst vehicle's, and the rest fly further inside their envelopes.
    """
    # At least a thousand samples: a fixed 10 ms step saw a sub-centimetre
    # hop at t = 0 only, and stretched nothing.
    times = plan.times(min(0.01, plan.duration / 1000.0))
    factor = 1.0
    for name, trajectory in plan.trajectories.items():
        speed_limit, acceleration_limit = limits[name]
        speed = float(np.linalg.norm(trajectory.sample(times, 1), axis=1).max())
        acceleration = float(np.linalg.norm(trajectory.sample(times, 2), axis=1).max())
        if speed_limit > 0:
            factor = max(factor, speed / speed_limit)
        if acceleration_limit > 0:
            factor = max(factor, math.sqrt(acceleration / acceleration_limit))
    if factor <= 1.0 + 1e-12:
        return
    # A hair of headroom: the factor is derived from a sampled maximum, and
    # the true maximum between two samples is a little higher.
    factor *= 1.0005
    plan.trajectories = {
        name: trajectory.dilated(factor)
        for name, trajectory in plan.trajectories.items()
    }
    plan.duration *= factor
    plan.time_scale = factor


def _linearise(
    x: np.ndarray,
    sampler: _Sampler,
    layout: _Layout,
    names: Sequence[str],
    radii: Mapping[str, float],
    limits: Mapping[str, Tuple[float, float]],
    obstacles: Sequence[Tuple[Tuple[float, ...], float]],
    clearance: float,
    affine: _Affine,
    inter_sample: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """The convex subproblem's inequalities, in reduced coordinates.

    Separation and obstacle rows are *supporting half-spaces*: satisfying
    them implies the true constraint, so a feasible subproblem solution is
    genuinely collision-free at the sampled times. Speed and acceleration
    rows are the faces of a polytope inscribed in the limit's ball, so they
    are conservative in the same way; see :func:`_limit_rows`.
    """
    positions = {name: sampler.values(x, name, 0) for name in names}
    velocities = {name: sampler.values(x, name, 1) for name in names}
    accelerations = {name: sampler.values(x, name, 2) for name in names}
    rows: List[np.ndarray] = []
    upper: List[float] = []
    times = sampler.times
    n_samples = len(times)
    step = float(times[1] - times[0]) if n_samples > 1 else 0.0

    unit = _unit

    def dip(relative_speed: np.ndarray, relative_acceleration: float) -> np.ndarray:
        """How far a distance can dip *between* two samples.

        The constraints are enforced at finitely many times, and a pair that
        clears the bar at both ends of an interval can still dip inside it.
        The dip is bounded: the distance between two points changes no faster
        than their relative speed, so over an interval of length ``h`` with
        both ends feasible the interior minimum is at least the smaller end
        value minus ``max|relative velocity| * h / 2``, plus a second-order
        term for the speed changing across the interval. Adding that bound to
        the required distance at every sample makes the *continuous*
        constraint hold, not merely the sampled one.

        The bound uses the current iterate's own relative speed rather than
        the worst case ``2 * v_max``: on a fleet that mostly flies apart the
        worst case inflates every requirement by half a metre and the
        trajectories pay for a conflict that never happens.
        """
        if not inter_sample:
            return np.zeros_like(relative_speed)
        half = 0.5 * step
        neighbours = np.maximum(
            relative_speed,
            np.maximum(
                np.roll(relative_speed, 1),
                np.roll(relative_speed, -1),
            ),
        )
        return neighbours * half + 0.5 * relative_acceleration * half * half

    default = np.zeros(layout.dimension)
    default[0] = 1.0

    for i, first in enumerate(names):
        _pair_rows(
            rows,
            upper,
            sampler,
            layout,
            names[i + 1 :],
            first,
            positions,
            velocities,
            radii,
            limits,
            clearance,
            dip,
            unit,
            default,
        )
        _obstacle_rows(
            rows,
            upper,
            sampler,
            layout,
            first,
            positions,
            velocities,
            radii,
            limits,
            obstacles,
            clearance,
            dip,
            unit,
            default,
        )
        _limit_rows(
            rows,
            upper,
            sampler,
            layout,
            first,
            velocities,
            accelerations,
            limits,
            unit,
            default,
        )

    if not rows:
        return np.zeros((0, affine.dimension)), np.zeros(0)
    full = np.array(rows)
    bounds = np.array(upper)
    reduced = full @ affine.basis
    return reduced, bounds - full @ affine.particular


def _pair_rows(
    rows,
    upper,
    sampler,
    layout,
    others,
    first,
    positions,
    velocities,
    radii,
    limits,
    clearance,
    dip,
    unit,
    default,
):
    """A supporting half-space per pair per sample: ``n̂ᵀ(p_i - p_j) >= r``."""
    for second in others:
        need = radii[first] + radii[second] + clearance
        delta = positions[first] - positions[second]
        distance = np.linalg.norm(delta, axis=1)
        relative_speed = np.linalg.norm(velocities[first] - velocities[second], axis=1)
        slack = dip(relative_speed, limits[first][1] + limits[second][1])
        for s in range(len(distance)):
            required = need + float(slack[s])
            if distance[s] > 2.5 * required + 1.0:
                # Far apart at this sample, and the trust region bounds how
                # far either can move in one iteration; the true separation is
                # re-checked outside the linearisation.
                continue
            direction = unit(delta[s], default)
            row = np.zeros(layout.size)
            sampler.write(row, first, s, -direction, 0)
            sampler.write(row, second, s, direction, 0)
            rows.append(row)
            upper.append(-required)


def _obstacle_rows(
    rows,
    upper,
    sampler,
    layout,
    first,
    positions,
    velocities,
    radii,
    limits,
    obstacles,
    clearance,
    dip,
    unit,
    default,
):
    """``n̂ᵀ(p - c) >= need``, rearranged as ``-n̂ᵀp <= -need - n̂ᵀc``."""
    own_speed = np.linalg.norm(velocities[first], axis=1)
    for centre, radius in obstacles:
        need = radius + radii[first] + clearance
        delta = positions[first] - np.asarray(centre)
        slack = dip(own_speed, limits[first][1])
        for s in range(len(own_speed)):
            direction = unit(delta[s], default)
            row = np.zeros(layout.size)
            sampler.write(row, first, s, -direction, 0)
            rows.append(row)
            upper.append(
                -need - float(slack[s]) - float(direction @ np.asarray(centre))
            )


#: Half the angle between the vertex the limit polytope puts on the current
#: heading and the vertices either side of it; see :func:`_inscribed_polytope`.
#: Smaller makes the polytope hug the ball more closely around the heading
#: and allows less sideways speed while the heading is still turning.
_LIMIT_HALF_ANGLE = math.radians(15.0)


@functools.lru_cache(maxsize=None)
def _inscribed_polytope(dimension: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(normals, bounds)`` of a polytope inside the unit ball, in the frame
    ``(heading, lateral_1, ..., lateral_{n-1})``.

    With ``a`` the component along the heading, ``b`` the lateral ones,
    ``t = tan φ`` and ``s = sin 2φ``, the polytope is

        ±a + t‖b‖₁ <= 1,      ‖b‖₁ <= s

    and each ``‖·‖₁`` bound is one linear row per sign pattern of ``b``.
    Every point of it lies in the ball: with ``r = ‖b‖₁ >= ‖b‖₂``,
    ``‖v‖² <= (1 - t r)² + r²``, which is convex in ``r`` and equals 1 at both
    ends of ``[0, s]`` -- that is what ``s = sin 2φ = 2t / (1 + t²)`` is. And
    it touches the ball at ``±heading``, so a vehicle flying straight along
    its current heading may use the whole limit.
    """
    lateral = dimension - 1
    t = math.tan(_LIMIT_HALF_ANGLE)
    s = math.sin(2.0 * _LIMIT_HALF_ANGLE)
    signs = list(itertools.product((-1.0, 1.0), repeat=lateral))
    normals: List[Tuple[float, ...]] = []
    bounds: List[float] = []
    for along in (1.0, -1.0):
        for sign in signs:
            normals.append((along,) + tuple(t * x for x in sign))
            bounds.append(1.0)
    if lateral:
        for sign in signs:
            normals.append((0.0,) + sign)
            bounds.append(s)
    return np.array(normals), np.array(bounds)


def _frame(direction: np.ndarray) -> np.ndarray:
    """An orthonormal basis of R^n, as rows, whose first row is ``direction``."""
    n = len(direction)
    basis = np.linalg.qr(np.column_stack([direction, np.eye(n)]))[0].T
    basis[0] = direction  # QR returns it up to sign
    return basis


def _limit_rows(
    rows,
    upper,
    sampler,
    layout,
    first,
    velocities,
    accelerations,
    limits,
    unit,
    default,
):
    """An inscribed polytope for the speed ball and the acceleration ball.

    The obvious linearisation -- the tangent plane ``ĥᵀv <= v_max`` at the
    current heading ``ĥ`` -- is an *outer* approximation: it caps the speed
    along the old heading and nothing sideways, so the next iterate can fly
    faster than ``v_max`` by turning a little. The optimiser finds that and
    exploits it. Given a straight 10 m flight it weaved 0.6 m from side to
    side, oscillated between two shapes, and still needed its clock
    stretched by 11% afterwards.

    These rows are an *inner* approximation instead, aligned with the
    current velocity (or acceleration): see :func:`_inscribed_polytope`. A
    subproblem solution satisfies the true limit at every sample, and on its
    own heading may use all of it, so the SCP's fixed points meet the limit
    exactly. Only the times *between* samples are left to the dilation at
    the end, which is why that dilation is now a fraction of a percent.
    """
    speed_limit, acceleration_limit = limits[first]
    normals, bounds = _inscribed_polytope(layout.dimension)
    for values, limit, derivative in (
        (velocities[first], speed_limit, 1),
        (accelerations[first], acceleration_limit, 2),
    ):
        if not math.isfinite(limit):
            continue
        for s in range(len(values)):
            directions = normals @ _frame(unit(values[s], default))
            for direction, bound in zip(directions, bounds):
                row = np.zeros(layout.size)
                sampler.write(row, first, s, direction, derivative)
                rows.append(row)
                upper.append(limit * bound)
