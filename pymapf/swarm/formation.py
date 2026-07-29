"""Formation control: hold a *shape*, not just a heading.

Flocking asks a swarm to move together; coverage asks it to spread out;
formation control asks for something stricter -- a specified geometry, held
while the group moves. A camera array that must keep its baseline, a lift team
carrying one object, an inspection line sweeping a wall: all of them need the
relative positions to be right, not merely collision-free.

What makes this more than "fly to your assigned point" is **what each agent is
allowed to know**. The three constraint types below need progressively less, and
each buys its weaker sensing with a weaker guarantee -- which is the whole
design space:

``displacement``
    Agents measure relative *positions* in a **common orientation** (a shared
    compass or north reference). Simplest and stiffest; the formation is fixed
    in translation only, so the group can drift but not rotate or scale.

``distance``
    Agents measure only inter-agent *distances* -- no shared frame at all. The
    shape is then defined up to rotation, translation and reflection, and holding
    it requires the graph to be **rigid**: enough distance constraints that no
    continuous deformation preserves them all. This module checks that.

``bearing``
    Agents measure only *directions* to neighbours (what a camera gives you).
    The shape is defined up to translation and **scale**, so a bearing-based
    formation can breathe -- which is a feature when the group must squeeze
    through a gap, and a bug when the baseline matters.

A fourth controller, ``leader_follower``, is the pragmatic one: designated
leaders track whatever trajectory you like and everyone else maintains an offset
from them. It composes with any flocking behavior, because leaders and followers
are just agents with different gains.

Shapes are objects too (:class:`FormationShape`), so "V of nine drones" and
"cube of eight" are one line each, and matching agents to slots is an explicit,
optimal assignment rather than an accident of indexing.

References
----------
* Oh, K.-K.; Park, M.-C.; and Ahn, H.-S. 2015. *A survey of multi-agent
  formation control.* Automatica 53: 424-440.  (the displacement / distance /
  bearing taxonomy this module follows)
* Ren, W.; and Beard, R. W. 2008. *Distributed Consensus in Multi-vehicle
  Cooperative Control.* Springer.  (consensus-based displacement formations)
* Krick, L.; Broucke, M. E.; and Francis, B. A. 2009. *Stabilisation of
  infinitesimally rigid formations of multi-robot networks.* International
  Journal of Control 82(3): 423-439.  (distance-based control and rigidity)
* Zhao, S.; and Zelazo, D. 2016. *Bearing rigidity and almost global
  bearing-only formation stabilization.* IEEE Transactions on Automatic Control
  61(5): 1255-1268.
* Balch, T.; and Arkin, R. C. 1998. *Behavior-based formation control for
  multirobot teams.* IEEE Transactions on Robotics and Automation 14(6):
  926-939.  (leader-follower and behavioural formations)
* Kuhn, H. W. 1955. *The Hungarian method for the assignment problem.* Naval
  Research Logistics Quarterly 2(1-2): 83-97.  (slot assignment)
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

from .base import Behavior, SwarmState, limit, register_behavior

__all__ = [
    "FormationShape",
    "LineFormation",
    "VFormation",
    "CircleFormation",
    "GridFormation",
    "CubeFormation",
    "SphereFormation",
    "CustomFormation",
    "get_shape",
    "SHAPES",
    "assign_slots",
    "formation_error",
    "is_infinitesimally_rigid",
    "DisplacementFormation",
    "DistanceFormation",
    "BearingFormation",
    "LeaderFollower",
]


# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------


class FormationShape(ABC):
    """A desired geometry, as offsets from the formation centre.

    Offsets are expressed in the *formation frame*; the controllers decide how
    that frame relates to the world (fixed for displacement-based control,
    free up to rotation for distance-based, free up to scale for bearing-based).
    """

    name = "abstract"

    #: Fewest spatial dimensions this geometry is defined in. A line exists on
    #: an axis; a V, a circle or a grid needs a plane to be a shape at all.
    minimum_dimension = 2

    def _check(self, dimension: int) -> None:
        """Reject a dimension the geometry is not defined in.

        Without this the shapes that write to a second column raise
        ``IndexError: index 1 is out of bounds``, which says nothing about what
        the caller did wrong.
        """
        if dimension < self.minimum_dimension:
            raise ValueError(
                "%s needs at least %d dimensions, got %d"
                % (type(self).__name__, self.minimum_dimension, dimension)
            )

    @abstractmethod
    def offsets(self, n: int, dimension: int = 2) -> np.ndarray:
        """``(n, dimension)`` desired positions relative to the centroid."""

    def distances(self, n: int, dimension: int = 2) -> np.ndarray:
        """Desired pairwise distances -- what a distance-based controller uses."""
        offsets = self.offsets(n, dimension)
        return np.linalg.norm(offsets[:, None, :] - offsets[None, :, :], axis=2)

    def bearings(self, n: int, dimension: int = 2) -> np.ndarray:
        """Desired unit directions ``g_ij`` -- what a bearing controller uses."""
        offsets = self.offsets(n, dimension)
        deltas = offsets[None, :, :] - offsets[:, None, :]
        norms = np.linalg.norm(deltas, axis=2, keepdims=True)
        return np.divide(deltas, np.maximum(norms, 1e-12))

    def centred(self, n: int, dimension: int = 2) -> np.ndarray:
        offsets = self.offsets(n, dimension)
        return offsets - offsets.mean(axis=0)

    def __repr__(self) -> str:
        return "%s()" % type(self).__name__


class LineFormation(FormationShape):
    """Agents abreast along one axis -- a sweep line for inspection or search."""

    name = "line"
    minimum_dimension = 1

    def __init__(self, spacing: float = 3.0, axis: int = 0):
        self.spacing = spacing
        self.axis = axis

    def offsets(self, n: int, dimension: int = 2) -> np.ndarray:
        self._check(dimension)
        offsets = np.zeros((n, dimension))
        offsets[:, self.axis] = (np.arange(n) - (n - 1) / 2) * self.spacing
        return offsets


class VFormation(FormationShape):
    """The echelon birds actually fly: two trailing arms behind a leader.

    ``sweep`` is the half-angle from the flight axis. The aerodynamic argument
    for it -- each bird sits in the upwash of the one ahead -- is also the
    engineering one for fixed-wing UAVs, and it makes the formation naturally
    collision-tolerant because nobody flies directly behind anybody.
    """

    name = "v"

    def __init__(self, spacing: float = 3.0, sweep: float = math.pi / 4, dihedral: float = 0.0):
        self.spacing = spacing
        self.sweep = sweep
        self.dihedral = dihedral

    def offsets(self, n: int, dimension: int = 2) -> np.ndarray:
        self._check(dimension)
        offsets = np.zeros((n, dimension))
        for index in range(n):
            rank = (index + 1) // 2          # 0 for the leader, then 1, 1, 2, 2...
            side = 1 if index % 2 else -1
            if index == 0:
                continue
            along = -rank * self.spacing * math.cos(self.sweep)
            across = side * rank * self.spacing * math.sin(self.sweep)
            offsets[index, 0] = along
            offsets[index, 1] = across
            if dimension > 2 and self.dihedral:
                offsets[index, 2] = rank * self.spacing * math.sin(self.dihedral)
        return offsets - offsets.mean(axis=0)


class CircleFormation(FormationShape):
    """Agents evenly spaced on a circle -- surveillance around a point."""

    name = "circle"

    def __init__(self, radius: Optional[float] = None, spacing: float = 3.0):
        self.radius = radius
        self.spacing = spacing

    def offsets(self, n: int, dimension: int = 2) -> np.ndarray:
        self._check(dimension)
        # Radius from spacing if not given: chord = 2 r sin(pi/n).
        radius = self.radius
        if radius is None:
            radius = self.spacing / (2 * math.sin(math.pi / max(n, 2))) if n > 1 else 0.0
        angles = 2 * math.pi * np.arange(n) / max(n, 1)
        offsets = np.zeros((n, dimension))
        offsets[:, 0] = radius * np.cos(angles)
        offsets[:, 1] = radius * np.sin(angles)
        return offsets


class GridFormation(FormationShape):
    """A rectangular lattice -- the densest way to cover a footprint."""

    name = "grid"

    def __init__(self, spacing: float = 3.0, columns: Optional[int] = None):
        self.spacing = spacing
        self.columns = columns

    def offsets(self, n: int, dimension: int = 2) -> np.ndarray:
        self._check(dimension)
        columns = self.columns or int(math.ceil(math.sqrt(n)))
        offsets = np.zeros((n, dimension))
        for index in range(n):
            row, column = divmod(index, columns)
            offsets[index, 0] = column * self.spacing
            offsets[index, 1] = row * self.spacing
        return offsets - offsets.mean(axis=0)


class CubeFormation(FormationShape):
    """A 3D lattice. Falls back to a grid when the swarm is planar."""

    name = "cube"

    def __init__(self, spacing: float = 3.0):
        self.spacing = spacing

    def offsets(self, n: int, dimension: int = 2) -> np.ndarray:
        self._check(dimension)
        if dimension < 3:
            return GridFormation(self.spacing).offsets(n, dimension)
        side = int(math.ceil(n ** (1 / 3)))
        coordinates = np.indices((side, side, side)).reshape(3, -1).T[:n].astype(float)
        offsets = coordinates * self.spacing
        return offsets - offsets.mean(axis=0)


class SphereFormation(FormationShape):
    """Agents on a sphere (Fibonacci lattice) -- a shell around a target.

    Takes ``spacing`` like every other shape, and derives the radius from it
    when one is not given: ``n`` points spread over ``4 pi r^2`` occupy about
    ``sqrt(3)/2 s^2`` each under hexagonal packing, so ``r = s sqrt(sqrt(3) n /
    (8 pi))``. Accepting only ``radius`` here made this the one shape that
    ``get_shape(name, spacing=...)`` could not construct, and forced every
    caller into a special case.
    """

    name = "sphere"

    def __init__(self, radius: Optional[float] = None, spacing: float = 3.0):
        self.radius = radius
        self.spacing = spacing

    def _radius(self, n: int) -> float:
        if self.radius is not None:
            return self.radius
        if n < 2:
            return 0.0
        return self.spacing * math.sqrt(math.sqrt(3.0) * n / (8 * math.pi))

    def offsets(self, n: int, dimension: int = 2) -> np.ndarray:
        self._check(dimension)
        if dimension < 3:
            return CircleFormation(radius=self._radius(n)).offsets(n, dimension)
        indices = np.arange(n, dtype=float) + 0.5
        z = 1.0 - 2.0 * indices / max(n, 1)
        r = np.sqrt(np.maximum(0.0, 1.0 - z ** 2))
        golden = math.pi * (3.0 - math.sqrt(5.0))
        theta = golden * indices
        return self._radius(n) * np.stack(
            [r * np.cos(theta), r * np.sin(theta), z], axis=1
        )


class CustomFormation(FormationShape):
    """Whatever geometry you hand it, centred and reused as-is."""

    name = "custom"

    def __init__(self, offsets: np.ndarray):
        self._offsets = np.atleast_2d(np.asarray(offsets, dtype=float))

    def offsets(self, n: int, dimension: int = 2) -> np.ndarray:
        self._check(dimension)
        if n > len(self._offsets):
            raise ValueError(
                "custom formation has %d slots, %d agents asked for one"
                % (len(self._offsets), n)
            )
        chosen = self._offsets[:n]
        if chosen.shape[1] < dimension:
            padded = np.zeros((n, dimension))
            padded[:, : chosen.shape[1]] = chosen
            chosen = padded
        return chosen[:, :dimension] - chosen[:, :dimension].mean(axis=0)


SHAPES: Dict[str, Type[FormationShape]] = {
    "line": LineFormation,
    "v": VFormation,
    "circle": CircleFormation,
    "grid": GridFormation,
    "cube": CubeFormation,
    "sphere": SphereFormation,
}


def register_shape(name: str):
    """Class decorator registering a :class:`FormationShape` under ``name``.

    The same pattern as :func:`~pymapf.swarm.base.register_behavior`: a shape
    defined outside the library becomes available to :func:`get_shape`, and so
    to every controller, without touching this module.
    """

    def decorator(cls: Type[FormationShape]) -> Type[FormationShape]:
        SHAPES[name.lower()] = cls
        return cls

    return decorator


def available_shapes() -> List[str]:
    """Names accepted by :func:`get_shape`, sorted."""
    return sorted(SHAPES)


def get_shape(name, **kwargs) -> FormationShape:
    """Resolve ``name`` to a :class:`FormationShape`.

    Accepts a shape instance (returned unchanged), an ``(n, d)`` array of
    offsets (wrapped in a :class:`CustomFormation`) or a registered name.
    """
    if isinstance(name, FormationShape):
        return name
    if isinstance(name, np.ndarray):
        return CustomFormation(name)
    try:
        factory = SHAPES[str(name).lower()]
    except KeyError:
        raise ValueError(
            "Unknown formation shape %r. Available: %s"
            % (name, ", ".join(available_shapes()))
        )
    return factory(**kwargs)


# --------------------------------------------------------------------------
# assignment and analysis
# --------------------------------------------------------------------------


def assign_slots(positions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Match agents to formation slots, minimising total travel.

    Which agent takes which slot is not cosmetic: assigning by index makes
    agents cross the formation to reach a slot someone else is standing next to,
    which is both slower and the main source of collisions while forming up.

    Solved exactly with the Hungarian algorithm on squared distances
    (Kuhn 1955); ``O(n^3)``, which is nothing at swarm sizes.
    """
    positions = np.atleast_2d(positions)
    targets = np.atleast_2d(targets)
    cost = np.linalg.norm(positions[:, None, :] - targets[None, :, :], axis=2) ** 2
    return _hungarian(cost)


def _hungarian(cost: np.ndarray) -> np.ndarray:
    """Optimal assignment for a square cost matrix (Jonker-Volgenant style).

    Kept dependency-free on purpose -- the whole point of this package is that
    the algorithms are readable and the imports are boring.
    """
    cost = np.asarray(cost, dtype=float)
    n, m = cost.shape
    if n != m:
        raise ValueError("assignment needs a square cost matrix")

    INF = float("inf")
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                current = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if current < minv[j]:
                    minv[j] = current
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    assignment = np.zeros(n, dtype=int)
    for j in range(1, m + 1):
        if p[j]:
            assignment[p[j] - 1] = j - 1
    return assignment


def formation_error(
    positions: np.ndarray,
    shape: FormationShape,
    allow_rotation: bool = True,
    allow_scaling: bool = False,
    allow_reflection: bool = False,
    iterations: int = 8,
    restarts: int = 8,
) -> float:
    """Mean per-agent distance from the best-fitting placement of ``shape``.

    The fit matters, and it is the whole reason this function takes flags. A
    formation that is perfect but rotated 30 degrees is a *success* for a
    distance-based controller and a *failure* for a displacement-based one. One
    that is perfect but twice the size is a success for a bearing-based
    controller and a failure for both of the others. One that is a perfect
    mirror image is a success for a distance-based controller and nothing else,
    because distances cannot see handedness. So the comparison is made under
    exactly the transformation group the controller is entitled to: translation
    always, and rotation, uniform scale and reflection each optionally. Every
    controller declares its own group -- :attr:`~_FormationBase.allow_rotation`,
    :attr:`~_FormationBase.allow_scaling`, :attr:`~_FormationBase.allow_reflection`
    -- and :meth:`_FormationBase.error` uses it, so no controller is ever graded
    on a symmetry it was never given the sensing to fix.

    Fit and correspondence are solved *together*. Procrustes needs to know which
    agent holds which slot, and the assignment needs to know the pose -- solving
    either one first with the other guessed is how a formation that has hit its
    target exactly gets scored as a failure. This alternates the two (Procrustes
    given the assignment, Hungarian given the pose) to a fixed point, which takes
    two or three rounds; ``iterations`` caps it.

    Alternation only finds a *local* optimum, and seeding it from the identity
    pose is a bad start when the formation is substantially rotated: the first
    assignment is then made against a target pointing the wrong way, and the
    pair can lock there. So the descent is restarted from several initial
    correspondences and the best fit wins (see :func:`_seed_assignments`). The
    cost is a handful of extra Hungarian solves on a matrix the size of the
    swarm, which is nothing.

    When ``allow_scaling`` is set the residual is reported relative to the
    fitted size, i.e. as a fraction of the formation's own scale, because an
    absolute residual is meaningless once size is free.
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    n, dimension = positions.shape
    base = shape.centred(n, dimension)
    centred = positions - positions.mean(axis=0)

    best = float("inf")
    for seed in _seed_assignments(centred, base, restarts if allow_rotation else 1):
        residual = _fit_once(
            centred,
            base,
            seed,
            allow_rotation,
            allow_scaling,
            allow_reflection,
            iterations,
        )
        best = min(best, residual)
    return best


def _seed_assignments(
    centred: np.ndarray, base: np.ndarray, restarts: int
) -> List[np.ndarray]:
    """Starting correspondences for the alternating fit, from two sources.

    Neither source is sufficient alone, and they fail on opposite shapes.

    The **radius seed** pairs agents to slots by rank of distance from the
    centroid. That ordering is invariant under any rotation, so it lands the
    correct correspondence immediately whenever the radii are distinct -- a
    sphere or a V. It is useless when they are degenerate: every corner of a
    cube is the same distance from the centre, and the rank order is then
    arbitrary.

    The **rotation seeds** assign against the target under a spread of trial
    poses. In the plane, evenly spaced angles cover SO(2) exactly. In 3D and
    above, SO(d) cannot be covered by a handful of samples, so these are drawn
    pseudo-randomly from a fixed seed -- deterministic, but only a sampling.
    They handle the degenerate-radius shapes the radius seed cannot, and miss
    some of the ones it gets right.

    Using both is what makes the fit exact on every shape in the library: over
    240 random 3D rotations, the worst residual is 0 with both, against 2.6 with
    rotation seeds alone and 1.3 with the radius seed alone.
    """
    count = max(1, int(restarts))
    dimension = base.shape[1]
    seeds = [assign_slots(centred, base)]
    if count == 1 or dimension < 2:
        return seeds

    # Rotation-invariant: rank by distance from the centroid.
    agent_order = np.argsort(np.linalg.norm(centred, axis=1))
    slot_order = np.argsort(np.linalg.norm(base, axis=1))
    radius_seed = np.empty(len(agent_order), dtype=int)
    radius_seed[agent_order] = slot_order
    seeds.append(radius_seed)

    for rotation in _seed_rotations(dimension, count):
        seeds.append(assign_slots(centred, base @ rotation))
    return seeds


def _seed_rotations(dimension: int, restarts: int) -> List[np.ndarray]:
    """Trial poses: exact coverage of SO(2), a fixed pseudo-random sample above."""
    count = max(1, int(restarts))
    if dimension < 2 or count == 1:
        return [np.eye(dimension)]

    if dimension == 2:
        rotations = []
        for index in range(count):
            angle = 2 * math.pi * index / count
            cos, sin = math.cos(angle), math.sin(angle)
            rotations.append(np.array([[cos, -sin], [sin, cos]]))
        return rotations

    # Seeded so the metric stays deterministic: a formation error that changes
    # between runs is worse than one that is occasionally loose.
    rotations = [np.eye(dimension)]
    generator = np.random.default_rng(0)
    while len(rotations) < count:
        matrix, _ = np.linalg.qr(generator.normal(size=(dimension, dimension)))
        if np.linalg.det(matrix) < 0:
            matrix[:, 0] *= -1
        rotations.append(matrix)
    return rotations


def _fit_once(
    centred: np.ndarray,
    base: np.ndarray,
    assignment: np.ndarray,
    allow_rotation: bool,
    allow_scaling: bool,
    allow_reflection: bool,
    iterations: int,
) -> float:
    """One alternating Procrustes/assignment descent from a given start."""
    scale = 1.0
    residual = float("inf")

    for _ in range(max(1, iterations)):
        matched = base[assignment]
        transformed = matched
        rotation = None

        if allow_rotation:
            u, _, vt = np.linalg.svd(matched.T @ centred)
            rotation = u @ vt
            if not allow_reflection and np.linalg.det(rotation) < 0:
                u[:, -1] *= -1
                rotation = u @ vt
            transformed = matched @ rotation

        scale = 1.0
        if allow_scaling:
            denominator = float(np.sum(transformed * transformed))
            if denominator > 1e-12:
                scale = float(np.sum(centred * transformed) / denominator)
            if scale <= 1e-9:  # a collapsed swarm: report the raw residual
                scale = 1.0
            transformed = transformed * scale

        residual = float(np.mean(np.linalg.norm(centred - transformed, axis=1)))

        # Re-assign under the pose just found; stop as soon as it stops moving.
        posed = base if rotation is None else base @ rotation
        if allow_scaling:
            posed = posed * scale
        updated = assign_slots(centred, posed)
        if np.array_equal(updated, assignment):
            break
        assignment = updated

    return residual / scale if allow_scaling else residual


def is_infinitesimally_rigid(
    positions: np.ndarray, edges: Sequence[Tuple[int, int]], tolerance: float = 1e-8
) -> bool:
    """Can this distance graph hold its shape, or can it fold?

    A framework is infinitesimally rigid when the rank of its rigidity matrix is
    ``dn - d(d+1)/2`` -- every motion that preserves all edge lengths is a rigid
    body motion. If it is not, a distance-based controller has a whole manifold
    of configurations that satisfy its constraints and will happily sit on the
    wrong one. Worth checking before trusting a distance formation.
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    n, dimension = positions.shape
    if n < 2 or not len(edges):
        return False

    matrix = np.zeros((len(edges), n * dimension))
    for row, (i, j) in enumerate(edges):
        difference = positions[i] - positions[j]
        matrix[row, i * dimension : (i + 1) * dimension] = difference
        matrix[row, j * dimension : (j + 1) * dimension] = -difference

    expected = dimension * n - dimension * (dimension + 1) // 2
    return int(np.linalg.matrix_rank(matrix, tol=tolerance)) >= expected


# --------------------------------------------------------------------------
# controllers
# --------------------------------------------------------------------------


class _FormationBase(Behavior):
    """Shared machinery: the shape, the slot assignment, and re-assignment."""

    def __init__(
        self,
        shape="v",
        spacing: Optional[float] = None,
        reassign_every: int = 25,
        gain: float = 3.0,
        damping: float = 2.0,
        degree: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        shape_kwargs = {}
        if spacing is not None and isinstance(shape, str) and shape in SHAPES:
            shape_kwargs["spacing"] = spacing
        self.shape = get_shape(shape, **shape_kwargs)
        self.reassign_every = reassign_every
        self.gain = gain
        self.damping = damping
        self.degree = max(0, int(degree))
        self._assignment: Optional[np.ndarray] = None
        self._graph: Optional[List[List[int]]] = None
        self._warned_rigidity = False
        self._assignment_step = -1
        self._steps = 0

    def targets(self, state: SwarmState) -> np.ndarray:
        """Desired absolute positions, in the same order as the agents."""
        offsets = self.shape.centred(state.n, state.dimension)
        centre = self.centre(state)
        return centre + offsets[self.assignment(state, offsets)]

    def centre(self, state: SwarmState) -> np.ndarray:
        """Where the formation sits: the swarm's own centroid.

        Deliberately *not* the waypoint. Placing the slots at the waypoint makes
        the shape term drive the group there as well as :meth:`migration`, and
        two independent terms pulling toward the same point overshoot it -- 2%
        past the target, measured, before settling back. Worse, the shape term
        is unbounded in the distance to the waypoint, so for a distant target it
        saturates the acceleration budget and the clamp scales the formation
        term down to nothing, which is the same failure the flocking behaviors
        hit with an unsaturated waypoint.

        So the division of labour is clean: the shape term holds the formation
        together around wherever the group currently is, and :meth:`migration`
        -- one bounded translation, common to every agent -- carries it to the
        waypoint.
        """
        return state.centroid

    def assignment(self, state: SwarmState, offsets: np.ndarray) -> np.ndarray:
        """Slot per agent, recomputed occasionally rather than every step.

        Re-assigning constantly makes agents chase a slot that keeps changing
        hands; never re-assigning locks in whatever the initial ordering was.

        Two guards matter here. The result is memoised *per step*, because this
        is called once per agent and the answer cannot change within a step --
        without that the Hungarian solve runs ``n`` times per reassignment,
        making the step O(n^4). And controllers with
        :attr:`reassigns` ``= False`` never re-solve at all: for distance and
        bearing control the assignment is baked into the desired distances,
        bearings and interaction graph at reset, so re-solving it later would
        change those targets discontinuously -- or, if the derived quantities
        are cached, silently do nothing at all.
        """
        size_changed = self._assignment is None or len(self._assignment) != state.n
        due = (
            self.reassigns
            and self.reassign_every > 0
            and self._steps % self.reassign_every == 0
            and self._assignment_step != self._steps
        )
        if size_changed or due:
            self._assignment = assign_slots(
                state.positions - state.positions.mean(axis=0), offsets
            )
            self._assignment_step = self._steps
        return self._assignment

    #: Whether the slot assignment may be re-solved while flying. False for
    #: controllers that derive fixed targets from it (see :meth:`assignment`).
    reassigns = True

    # ------------------------------------------------------------------
    # the interaction graph
    # ------------------------------------------------------------------
    def interaction_graph(self, state: SwarmState) -> List[List[int]]:
        """Which pairs constrain each other, taken from the *target* shape.

        Displacement control can use whatever neighbours happen to be in sensing
        range, because every agent knows its own absolute slot. Distance and
        bearing control cannot: their constraints only pin the formation down if
        the constraint graph is rigid, and a range-limited proximity graph is
        not rigid in general -- worse, it *changes* as the swarm moves, so the
        set of constraints being descended shifts underneath the descent. That
        is how a distance controller ends up expanding forever: pairs push apart
        to their desired distance, leave sensing range, and the pairs that were
        supposed to pull them back were never in it.

        So the graph is built once, from the geometry of the target formation,
        and held fixed. ``degree = 0`` (the default) means the complete graph,
        which is trivially rigid and is what small teams should use. A positive
        ``degree`` keeps only that many nearest slots per agent -- sparser and
        more realistic -- and edges are then added back until
        :func:`is_infinitesimally_rigid` accepts the framework, because a
        controller descending a non-rigid set of constraints has a flex mode it
        will happily drift along.
        """
        if self._graph is not None and len(self._graph) == state.n:
            return self._graph

        n = state.n
        offsets = self.shape.centred(n, state.dimension)
        ordered = offsets[self.assignment(state, offsets)]
        distances = np.linalg.norm(ordered[:, None, :] - ordered[None, :, :], axis=2)

        adjacency = np.zeros((n, n), dtype=bool)
        if self.degree <= 0 or self.degree >= n - 1:
            adjacency[:] = True
        else:
            masked = distances.copy()
            np.fill_diagonal(masked, np.inf)
            order = np.argsort(masked, axis=1)
            for i in range(n):
                for j in order[i, : self.degree]:
                    adjacency[i, int(j)] = adjacency[int(j), i] = True
            # Augment until rigid: add the shortest missing edge each round.
            candidates = sorted(
                (
                    (float(distances[i, j]), i, j)
                    for i in range(n)
                    for j in range(i + 1, n)
                    if not adjacency[i, j]
                )
            )
            for _, i, j in candidates:
                edges = [
                    (a, b) for a in range(n) for b in range(a + 1, n) if adjacency[a, b]
                ]
                if is_infinitesimally_rigid(ordered, edges):
                    break
                adjacency[i, j] = adjacency[j, i] = True
        np.fill_diagonal(adjacency, False)
        self._graph = [list(map(int, np.flatnonzero(adjacency[i]))) for i in range(n)]

        if self.requires_rigidity and not self._warned_rigidity:
            edges = [(a, b) for a in range(n) for b in range(a + 1, n) if adjacency[a, b]]
            if not is_infinitesimally_rigid(ordered, edges):
                self._warned_rigidity = True
                warnings.warn(
                    "%s: the target %r formation is not infinitesimally rigid in "
                    "%dD, so this controller has flex modes it cannot see and "
                    "will not converge to the shape. A collinear target in 2D "
                    "and a planar target in 3D are the usual causes."
                    % (type(self).__name__, self.shape.name, state.dimension),
                    RuntimeWarning,
                    stacklevel=2,
                )
        return self._graph

    #: Set on controllers whose constraints only pin the shape down when the
    #: interaction graph is rigid; :meth:`interaction_graph` then warns if it
    #: is not, rather than silently converging to the wrong thing.
    requires_rigidity = False

    def migration(self, state: SwarmState, index: int) -> np.ndarray:
        """Move the *formation* to the waypoint, not each agent individually.

        The inherited rule pulls every agent toward the waypoint separately.
        For a flock that is harmless -- its own repulsion pushes back. For a
        formation it is not: a per-agent pull toward a common point is a
        contraction, and a contraction is a deformation, so the waypoint term
        fights the shape term and the formation settles squashed. Bearing-based
        control fails hardest, because scale is exactly the degree of freedom it
        does not constrain: with nothing resisting, the team collapses onto the
        waypoint and reports a perfectly converged formation of zero size.

        A translation, by contrast, lies in the null space of all four laws --
        every one of them is translation-invariant by construction. So the
        command is computed once from the *centroid* error and issued
        identically to everybody, and the formation slides to the waypoint
        without changing shape at all.
        """
        params = self.params
        if params.migration_point is None:
            return np.zeros(state.dimension)
        target = np.asarray(params.migration_point, dtype=float)
        return limit(
            params.migration_gain * (target - state.centroid),
            params.migration_authority * params.max_acceleration,
        )

    def reset(self, state: SwarmState) -> None:
        self._assignment = None
        self._graph = None
        self._warned_rigidity = False
        self._assignment_step = -1
        self._steps = 0

    def commands(self, state: SwarmState) -> np.ndarray:
        self._steps += 1
        return super().commands(state)

    def error(self, state: SwarmState) -> float:
        """Current formation error, under this controller's own symmetry group."""
        return formation_error(
            state.positions,
            self.shape,
            allow_rotation=self.allow_rotation,
            allow_scaling=self.allow_scaling,
            allow_reflection=self.allow_reflection,
        )

    #: The symmetries this controller's sensing cannot resolve, and which
    #: :meth:`error` therefore quotients out. Overridden per controller.
    allow_rotation = False
    allow_scaling = False
    allow_reflection = False


@register_behavior("formation")
@register_behavior("displacement_formation")
class DisplacementFormation(_FormationBase):
    """Hold the shape using relative positions in a common frame.

    ``u_i = k (p_i^des - p_i) - c v_i``, with the desired position taken from the
    assigned slot. Requires a shared orientation -- a compass, GPS heading, or a
    motion-capture frame -- and in exchange fixes the formation completely: no
    rotation, no scaling, no reflection.

    Collision avoidance is *not* implicit here. Two agents swapping slots will
    fly through each other unless something keeps them apart, so a short-range
    repulsion is included and is the reason ``separation_distance`` is respected
    during the transient.
    """

    allow_rotation = False

    def __init__(self, repulsion: float = 6.0, **kwargs):
        super().__init__(**kwargs)
        self.repulsion = repulsion

    def command(self, state: SwarmState, index: int) -> np.ndarray:
        target = self.targets(state)[index]
        error = target - state.positions[index]
        command = self.gain * error - self.damping * state.velocities[index]

        for j in self.neighbors(state, index):
            offset = state.positions[j] - state.positions[index]
            distance = float(np.linalg.norm(offset))
            if 1e-9 < distance < self.params.separation_distance:
                command -= (
                    self.repulsion
                    * (self.params.separation_distance - distance)
                    / max(distance, 0.2)
                    * (offset / distance)
                )
        return self.finalise(command, state, index)


@register_behavior("distance_formation")
class DistanceFormation(_FormationBase):
    """Hold the shape using only inter-agent *distances* -- no shared frame.

    Two potentials are available, and the difference between them is not
    cosmetic.

    ``potential="squared"`` is the textbook one, Krick, Broucke and Francis
    (2009): descend ``sum_ij (|p_ij|^2 - d_ij^2)^2``, giving

        ``u_i = -k sum_j (|p_ij|^2 - d_ij^2) (p_i - p_j)``

    It is smooth everywhere including ``p_i = p_j``, which is why the stability
    analysis is done on it. It is also *cubic* in the error, so an agent that
    starts one formation-width away asks for a hundred times the acceleration it
    can deliver, and what the vehicle actually executes is a saturated bang-bang
    command that overshoots.

    ``potential="linear"`` (the default here) descends ``sum_ij (|p_ij| -
    d_ij)^2`` instead:

        ``u_i = -k sum_j (|p_ij| - d_ij) (p_i - p_j)/|p_ij|``

    Same equilibria, same rigidity theory, but the demand is proportional to the
    error, so it stays inside the actuator envelope. The cost is a singularity
    at coincident agents, guarded here with a floor on ``|p_ij|``.

    What either one gives up is orientation: the formation converges to the
    right shape in *some* rotation, and possibly reflected, because distances
    cannot tell those apart. What it buys is that no agent needs a compass.
    Whether the shape is determined by its distances at all is a rigidity
    question -- see :meth:`~_FormationBase.interaction_graph` and
    :meth:`rigid`.
    """

    allow_rotation = True
    allow_reflection = True
    requires_rigidity = True
    # The assignment is baked into the desired distances and the interaction
    # graph at reset; re-solving it later would move the targets under the
    # descent for no benefit, since distances fix the shape only up to a
    # relabelling anyway.
    reassigns = False

    def __init__(self, potential: str = "linear", **kwargs):
        super().__init__(**kwargs)
        if potential not in ("linear", "squared"):
            raise ValueError("potential must be 'linear' or 'squared'")
        self.potential = potential
        self._desired: Optional[np.ndarray] = None

    def desired_distances(self, state: SwarmState) -> np.ndarray:
        if self._desired is None or len(self._desired) != state.n:
            offsets = self.shape.centred(state.n, state.dimension)
            assignment = self.assignment(state, offsets)
            ordered = offsets[assignment]
            self._desired = np.linalg.norm(
                ordered[:, None, :] - ordered[None, :, :], axis=2
            )
        return self._desired

    def rigid(self, state: SwarmState) -> bool:
        """Is the interaction graph rigid enough to fix the shape?"""
        graph = self.interaction_graph(state)
        edges = [(i, j) for i in range(state.n) for j in graph[i] if j > i]
        return is_infinitesimally_rigid(state.positions, edges)

    def reset(self, state: SwarmState) -> None:
        super().reset(state)
        self._desired = None

    def command(self, state: SwarmState, index: int) -> np.ndarray:
        desired = self.desired_distances(state)
        neighbors = self.interaction_graph(state)[index]
        command = np.zeros(state.dimension)

        for j in neighbors:
            offset = state.positions[index] - state.positions[j]
            distance = float(np.linalg.norm(offset))
            target = float(desired[index, j])
            if self.potential == "squared":
                command -= self.gain * (distance ** 2 - target ** 2) * offset
            elif distance > 1e-6:
                command -= self.gain * (distance - target) * offset / distance

        command /= max(1, len(neighbors))
        command -= self.damping * state.velocities[index]
        # Distances alone do not pin the group down in space; the waypoint term
        # (bounded, as always) is what translates the finished shape.
        return self.finalise(command, state, index)


@register_behavior("bearing_formation")
class BearingFormation(_FormationBase):
    """Hold the shape using only *directions* to neighbours.

    ``u_i = -k sum_j P(g_ij) (p_i - p_j)`` where ``P(g) = I - g g^T`` projects
    out the component along the desired bearing, so the controller corrects only
    the part of the relative position that points the wrong way -- and says
    nothing about distance.

    That is the defining property: the formation is fixed up to translation and
    **scale**. The team keeps its shape while breathing in and out, which is
    exactly right for a camera swarm (bearings are what cameras measure) and
    exactly wrong when the baseline is the measurement. It is also why
    :attr:`allow_scaling` is set: grading this controller on absolute size would
    be grading it on the one degree of freedom bearings provably cannot see. Set
    ``scale_gain`` if you want the size pinned, at the cost of needing a range
    measurement somewhere.

    Following Zhao and Zelazo (2016), the constraint set is the *bearing* graph
    of the target shape rather than whoever is in sensing range -- same reason
    as :class:`DistanceFormation`, and the reason it converges at all.
    """

    allow_rotation = False
    allow_scaling = True
    requires_rigidity = True
    reassigns = False  # baked into the desired bearings; see DistanceFormation

    def __init__(self, scale_gain: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.scale_gain = scale_gain
        self._bearings: Optional[np.ndarray] = None

    def desired_bearings(self, state: SwarmState) -> np.ndarray:
        if self._bearings is None or len(self._bearings) != state.n:
            offsets = self.shape.centred(state.n, state.dimension)
            ordered = offsets[self.assignment(state, offsets)]
            deltas = ordered[None, :, :] - ordered[:, None, :]
            norms = np.linalg.norm(deltas, axis=2, keepdims=True)
            self._bearings = np.divide(deltas, np.maximum(norms, 1e-12))
        return self._bearings

    def reset(self, state: SwarmState) -> None:
        super().reset(state)
        self._bearings = None

    def command(self, state: SwarmState, index: int) -> np.ndarray:
        bearings = self.desired_bearings(state)
        neighbors = self.interaction_graph(state)[index]
        command = np.zeros(state.dimension)
        identity = np.eye(state.dimension)

        for j in neighbors:
            g = bearings[index, j]
            projector = identity - np.outer(g, g)
            command -= self.gain * projector @ (state.positions[index] - state.positions[j])
        command /= max(1, len(neighbors))

        if self.scale_gain:
            # Optional: pin the scale, which bearings alone leave free.
            offsets = self.shape.centred(state.n, state.dimension)
            desired_spread = float(np.mean(np.linalg.norm(offsets, axis=1)))
            radial = state.positions[index] - state.centroid
            spread = float(np.mean(np.linalg.norm(state.positions - state.centroid, axis=1)))
            if spread > 1e-9:
                command += self.scale_gain * (desired_spread - spread) * radial / spread

        command -= self.damping * state.velocities[index]
        return self.finalise(command / max(1, state.n), state, index)


@register_behavior("leader_follower")
class LeaderFollower(_FormationBase):
    """Leaders go where they are told; followers hold an offset from them.

    The oldest formation scheme and still the most deployed, because it needs no
    consensus at all: a follower tracks one leader, and the leader tracks the
    mission. Its known weakness is equally simple -- error propagates *down* the
    chain and nothing propagates back up, so a follower that falls behind does
    not slow the leader. Setting ``leaders`` to more than one agent bounds the
    chain depth, which is the standard mitigation.
    """

    allow_rotation = False

    def __init__(self, leaders: int = 1, follow_gain: float = 3.0, **kwargs):
        super().__init__(**kwargs)
        self.leaders = max(1, leaders)
        self.follow_gain = follow_gain

    def is_leader(self, index: int) -> bool:
        return index < self.leaders

    def command(self, state: SwarmState, index: int) -> np.ndarray:
        offsets = self.shape.centred(state.n, state.dimension)
        assignment = self.assignment(state, offsets)

        if self.is_leader(index):
            # Leaders answer only to the mission -- the waypoint term inside
            # finalise -- and to nothing else. In particular a leader must not
            # be pulled toward the swarm centroid when there is no waypoint:
            # the centroid depends on the followers, the followers track the
            # leader, and the loop closes on itself and never settles. With no
            # mission a leader simply holds station and lets the team form up
            # around it.
            return self.finalise(
                -self.damping * state.velocities[index], state, index
            )

        # Followers hold their slot relative to the leaders' mean position.
        leader_centre = state.positions[: self.leaders].mean(axis=0)
        leader_slot = offsets[assignment[: self.leaders]].mean(axis=0)
        target = leader_centre + (offsets[assignment[index]] - leader_slot)

        command = self.follow_gain * (target - state.positions[index])
        command -= self.damping * state.velocities[index]
        # Match the leaders' velocity, so the formation travels rather than
        # perpetually catching up.
        command += self.gain * 0.5 * (
            state.velocities[: self.leaders].mean(axis=0) - state.velocities[index]
        )

        for j in self.neighbors(state, index):
            offset = state.positions[j] - state.positions[index]
            distance = float(np.linalg.norm(offset))
            if 1e-9 < distance < self.params.separation_distance:
                command -= (
                    6.0
                    * (self.params.separation_distance - distance)
                    / max(distance, 0.2)
                    * (offset / distance)
                )
        return self.finalise(command, state, index)
