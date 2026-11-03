"""Turn a discrete MAPF plan into a kinematically feasible schedule.

The idea is Hönig et al.'s MAPF-POST (ICAPS 2016): keep the *order* in which
the discrete plan has agents visit each vertex, throw away its *timing*, and
re-derive the timing as the solution of a simple temporal network whose
constraints are physical rather than unit-step. Two kinds of constraint:

* **travel** -- an agent cannot leave a vertex before it has arrived there,
  and a move takes at least as long as its :class:`MotionProfile` says;
* **safety** -- if the discrete plan has agent A at vertex ``v`` before agent
  B, then B may arrive at ``v`` only ``safety_time`` seconds after A *starts to
  leave* it.

Every constraint has the form ``x >= y + w``, so the earliest schedule is the
longest path from a virtual source, found by Bellman-Ford. Using the moment A
*starts* leaving rather than the moment it has fully left is what makes
rotations feasible: three agents cycling around a loop all start at once, and
the spacing between them is the edge length. It also means the safety margin
cannot exceed a move's duration -- ask for more and the network has a positive
cycle, which :class:`InfeasibleScheduleError` reports as exactly that.

The paper's safety margin is one number. Here it is derived per hand-over by
:func:`required_margin`, which simulates A leaving and B arriving under their
actual profiles and the actual angle between the two moves, and finds the
smallest delay at which they never come within ``safety_distance``. That
matters because rest-to-rest profiles are slow at both ends: A creeps away
from the vertex while B creeps onto it, and a margin sized for full speed
lets a right-angle hand-over get far closer than asked. The tests check the
sampled minimum over whole plans against the distance requested.

**Straight runs.** By default every vertex is a halt: the agent comes to rest
at each cell, which is what makes each move's duration independent of the
next. With ``merge_straight=True`` consecutive collinear moves with no wait
between them become one *run* flown under one profile, and the vertices along
it are *passed through* at speed. The network stays linear because a run's
profile is fixed once its length is: the agent passes the k-th vertex of a
run a constant ``tau_k`` after the run departs, so "A passes ``v``" is still
the run's departure plus a constant, and every safety constraint is still a
difference. The hand-over margin is derived from the same simulation, now
with both agents moving through the vertex rather than starting or stopping
on it. This is the form a vehicle that can hover or coast wants: a quadrotor
does not land at every metre of a corridor. A merged run is ordered against
others at vertices it passes long after it departs, which can close a cycle
the halted plan does not have; the plan is then scheduled halted instead.

**Near misses.** Two moves can come close without sharing a vertex: diagonal
moves cross in the middle of a cell, and neighbours fly side by side. The
discrete plan calls ``(0,0)->(1,1)`` against ``(0,1)->(1,0)`` valid, and
scheduled on shared vertices alone the two met 0.004 apart. So a pair of
runs that comes within the safety distance (or crosses, under
``safety_time``) is treated as if it shared its point of closest approach:
whoever the discrete plan has there first leads, and the other is delayed by
the same simulation.

What the model still does not see is an agent *halted* near another's path:
the discrete plan lets them coexist, and no ordering of departures moves
them apart. On a unit grid that gap is at least one cell without diagonals
and ``√2/2`` of one with them, so a ``safety_distance`` up to that holds;
ask for more and :meth:`TrajectorySet.min_separation` is the check, as it is
for anything else this model does not see.

Deviations from the paper: profiles are rest-to-rest per run (the paper
allows any speed within limits, at the cost of a second bound per move and an
LP), and there is no upper bound on dwell, so an agent may wait indefinitely
for a slower one. Both are simplifications that keep the network a plain
longest-path problem and the core free of a solver dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

from .trajectory import (
    MotionProfile,
    Point,
    Segment,
    Stay,
    Trajectory,
    TrajectorySet,
    distance,
    lerp,
)

__all__ = [
    "KinematicLimits",
    "InfeasibleScheduleError",
    "plan_trajectories",
    "required_margin",
    "stays_of",
]


@dataclass(frozen=True)
class KinematicLimits:
    """What one agent can do.

    Args:
        v_max: top speed, in distance units per second.
        a_max: acceleration limit; ``None`` means the agent changes speed
            instantly, which is the classical MAPF-POST assumption.
        safety_distance: the closest another agent may come during a
            hand-over at a shared vertex. Converted to a per-hand-over delay
            by :func:`required_margin`, which accounts for both agents'
            profiles and the angle between their moves.
        safety_time: seconds a vertex must have been left before the next
            agent arrives. A fixed rule, the paper's original; overrides
            ``safety_distance`` when given.
    """

    v_max: float = 1.0
    a_max: Optional[float] = None
    safety_distance: float = 0.0
    safety_time: Optional[float] = None

    def __post_init__(self):
        if self.v_max <= 0:
            raise ValueError("v_max must be positive")
        if self.a_max is not None and self.a_max <= 0:
            raise ValueError("a_max must be positive or None")
        if self.safety_distance < 0:
            raise ValueError("safety_distance cannot be negative")
        if self.safety_time is not None and self.safety_time < 0:
            raise ValueError("safety_time cannot be negative")

    def profile(self, length: float) -> MotionProfile:
        return MotionProfile(length, self.v_max, self.a_max)


class InfeasibleScheduleError(ValueError):
    """No timing satisfies the constraints.

    For a valid discrete plan this only happens when the safety margin is
    longer than some move's duration on a cycle of agents that rotate
    simultaneously: everyone must wait for everyone else, forever.
    """


# --------------------------------------------------------------------------
# discrete path -> stays
# --------------------------------------------------------------------------


def stays_of(path: Sequence[Hashable]) -> List[Tuple[Hashable, int, int]]:
    """Collapse a discrete path into ``(vertex, first_t, last_t)`` stays.

    A wait action in the discrete plan is the same vertex at consecutive
    timesteps; here it is one stay with a longer dwell, since the continuous
    schedule decides dwell for itself.
    """
    if not path:
        raise ValueError("empty path")
    stays: List[Tuple[Hashable, int, int]] = []
    first = 0
    for t in range(1, len(path) + 1):
        if t == len(path) or path[t] != path[first]:
            stays.append((path[first], first, t - 1))
            first = t
    return stays


# --------------------------------------------------------------------------
# runs: one profile over one straight stretch
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Run:
    """One straight move under one profile, with the vertices it passes.

    ``via`` lists ``(vertex, discrete_time, tau)`` for every vertex strictly
    between the endpoints: the agent is there ``tau`` seconds after the run
    departs. Without ``merge_straight`` every run has an empty ``via``.
    """

    origin: Point
    destination: Point
    profile: MotionProfile
    via: Tuple[Tuple[Hashable, int, float], ...] = ()

    @property
    def duration(self) -> float:
        return self.profile.duration

    def at(self, t: float) -> Point:
        """Position ``t`` seconds after departure, clamped to the endpoints."""
        if self.profile.length == 0:
            return self.origin
        fraction = self.profile.travelled(t) / self.profile.length
        return lerp(self.origin, self.destination, fraction)


def _direction(a: Point, b: Point) -> Optional[Point]:
    length = distance(a, b)
    if length == 0:
        return None
    return tuple((y - x) / length for x, y in zip(a, b))


def _same_direction(u: Optional[Point], v: Optional[Point]) -> bool:
    if u is None or v is None:
        return False
    return all(abs(x - y) < 1e-9 for x, y in zip(u, v))


def _runs_of(
    agent_stays: Sequence[Tuple[Hashable, int, int]],
    coords: Mapping[Hashable, Point],
    limits: KinematicLimits,
    merge_straight: bool,
) -> Tuple[List[int], List[_Run]]:
    """Split an agent's stays into halts and the runs between them.

    Returns ``(halts, runs)``: ``halts`` are indices into ``agent_stays`` at
    which the agent is at rest, and ``runs[i]`` goes from halt ``i`` to halt
    ``i + 1``. A stay is merged into a run when it has no discrete wait and
    the moves on either side of it point the same way.
    """
    halts = [0]
    for k in range(1, len(agent_stays) - 1):
        vertex, first, last = agent_stays[k]
        before = _direction(coords[agent_stays[k - 1][0]], coords[vertex])
        after = _direction(coords[vertex], coords[agent_stays[k + 1][0]])
        if merge_straight and first == last and _same_direction(before, after):
            continue
        halts.append(k)
    if len(agent_stays) > 1:
        halts.append(len(agent_stays) - 1)

    runs: List[_Run] = []
    for start, stop in zip(halts, halts[1:]):
        origin = coords[agent_stays[start][0]]
        destination = coords[agent_stays[stop][0]]
        profile = limits.profile(distance(origin, destination))
        via = []
        for k in range(start + 1, stop):
            vertex, first, _last = agent_stays[k]
            via.append(
                (
                    vertex,
                    first,
                    profile.time_to_travel(distance(origin, coords[vertex])),
                )
            )
        runs.append(_Run(origin, destination, profile, tuple(via)))
    return halts, runs


# --------------------------------------------------------------------------
# the hand-over margin
# --------------------------------------------------------------------------


def _pair_delay(
    leader: _Run,
    follower: _Run,
    delay_order: float,
    safety_distance: float,
    resolution: float = 0.025,
) -> float:
    """Smallest delay between two runs' departures that keeps them apart.

    ``leader`` is the run that the discrete plan has first at every vertex
    the two share; ``delay_order`` is the least delay of the follower's
    departure after the leader's that respects that order (the follower
    reaches each shared vertex no earlier than the leader did). From there
    the delay grows until, over the interval in which *both* runs are in
    motion, the two agents never come within ``safety_distance``.

    Only simultaneous motion is simulated. While one agent is halted the
    other's passage through that halt vertex is a hand-over of its own,
    constrained by the runs that touch it; and a halt the other run merely
    passes *near* is the adjacent-vertex gap the module docstring describes.
    Simulating a halt as "parked there forever" would instead forbid runs
    that share an endpoint outright, which the plan may well require.

    The search is a coarse scan refined by bisection, which does not assume
    the separation is monotone in the delay (on an acute angle it need not
    be) -- only that the scan is fine enough to bracket the first crossing.
    Returns at most the leader's duration past ``delay_order``, by which time
    the leader has completed its run before the follower departs; a
    ``safety_distance`` violated even then is outside what the runs alone
    can promise.
    """
    if safety_distance <= 0:
        return delay_order

    def separation(delay: float) -> float:
        lo = max(0.0, delay)
        hi = min(leader.duration, delay + follower.duration)
        if hi < lo:
            return math.inf
        samples = max(24, int(math.ceil((hi - lo) / resolution)))
        best = math.inf
        for k in range(samples + 1):
            t = lo + (hi - lo) * k / samples
            best = min(best, distance(leader.at(t), follower.at(t - delay)))
        return best

    cap = max(delay_order, leader.duration)
    return _first_clearing(separation, safety_distance, delay_order, cap)


def _first_clearing(separation, wanted: float, lo: float, hi: float) -> float:
    """Smallest ``x`` in ``[lo, hi]`` with ``separation(x) >= wanted``, by a
    coarse scan that brackets the first crossing and a bisection that
    refines it; ``hi`` when nothing in the range clears."""
    if separation(lo) >= wanted:
        return lo
    if hi <= lo:
        return hi
    steps = 32
    previous = lo
    for k in range(1, steps + 1):
        candidate = lo + (hi - lo) * k / steps
        if separation(candidate) >= wanted:
            lo, hi = previous, candidate
            break
        previous = candidate
    else:
        return hi
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if separation(mid) >= wanted:
            hi = mid
        else:
            lo = mid
    return hi


def required_margin(
    vertex: Point,
    leaving_to: Point,
    leaving: MotionProfile,
    arriving_from: Point,
    arriving: MotionProfile,
    safety_distance: float,
    samples: int = 48,
) -> float:
    """How long after A starts leaving ``vertex`` may B arrive there, so that
    the two are never closer than ``safety_distance``.

    A scalar rule cannot answer this. A leaves from rest, so at first it is
    barely moving; B arrives *to* rest, so it spends the end of its approach
    almost on the vertex. Whether that is dangerous depends on the angle
    between the two moves -- straight following is benign, a right-angle
    hand-over is the worst case on a grid -- and on both profiles. So the
    interaction is simulated: for a candidate delay, sample the distance over
    B's whole approach with A moving as scheduled, and search for the smallest
    delay whose minimum clears ``safety_distance``.

    This is the rest-to-rest case of the pairwise simulation the scheduler
    runs on every pair of runs that share a vertex (:func:`_pair_delay`);
    ``samples`` sets its time resolution over one second of motion.
    """
    a = _Run(vertex, leaving_to, leaving)
    b = _Run(arriving_from, vertex, arriving)
    # B arriving exactly as A departs is delay_order = -duration(B) between
    # departures; the margin is measured from A's departure to B's arrival.
    delay = _pair_delay(a, b, -b.duration, safety_distance, resolution=1.0 / samples)
    return delay + b.duration


def _resolve_positions(paths, graph, positions, cell_size) -> Dict[Hashable, Point]:
    """Coordinates for every vertex that appears in any path."""
    vertices = {v for path in paths.values() for v in path}
    if positions is not None:
        table = {v: tuple(float(x) for x in positions[v]) for v in vertices}
    elif graph is not None and getattr(graph, "positions", None):
        table = {v: tuple(float(x) for x in graph.positions[v]) for v in vertices}
    else:
        # Grid cells are their own coordinates.
        try:
            table = {v: tuple(float(x) * cell_size for x in v) for v in vertices}
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "vertices are not coordinate tuples; pass positions= or a graph "
                "with a .positions layout"
            ) from exc
    return table


# --------------------------------------------------------------------------
# the temporal network
# --------------------------------------------------------------------------


def plan_trajectories(
    solution,
    graph=None,
    positions: Optional[Mapping[Hashable, Sequence[float]]] = None,
    limits: Optional[KinematicLimits] = None,
    limits_by_agent: Optional[Mapping[str, KinematicLimits]] = None,
    cell_size: float = 1.0,
    merge_straight: bool = False,
) -> TrajectorySet:
    """Schedule a discrete :class:`pymapf.Solution` under kinematic limits.

    Args:
        solution: a *valid* discrete plan. Validity is what makes the visit
            order at each vertex well-defined; an invalid plan is rejected.
        graph: the map the plan was made on, used for vertex coordinates when
            it carries a ``positions`` layout (an ``ExplicitGraph``).
        positions: explicit ``{vertex: (x, y, ...)}`` coordinates, taking
            precedence over ``graph``. Grid cells need neither.
        limits: limits shared by every agent (default: ``v_max = 1``).
        limits_by_agent: per-agent overrides, for heterogeneous fleets.
        cell_size: metres per grid cell when cells are used as coordinates.
        merge_straight: fly consecutive collinear moves as one run under one
            profile, passing the vertices between them at speed, instead of
            coming to rest at every vertex. The trajectory then has a stay
            only where the agent turns or waits; the vertices passed through
            are listed on the segment's ``via``. If the merged runs order
            each other into a cycle that the unmerged plan does not have,
            the plan is scheduled unmerged instead: a slower schedule, not
            a refusal of a valid plan.

    Returns:
        A :class:`TrajectorySet`, earliest-possible under the constraints.

    Raises:
        InfeasibleScheduleError: see the class docstring.
        ValueError: on an invalid plan or unresolvable coordinates.
    """
    if not solution.is_valid():
        conflict = solution.first_conflict()
        raise ValueError(
            "the discrete plan has a %s conflict between %r and %r at t=%d; "
            "schedule only valid plans"
            % (conflict.kind, conflict.a, conflict.b, conflict.t)
        )
    paths = solution.paths
    if not paths:
        raise ValueError("the solution has no agents")

    shared = limits or KinematicLimits()
    overrides = dict(limits_by_agent or {})

    def limits_for(agent: str) -> KinematicLimits:
        return overrides.get(agent, shared)

    coords = _resolve_positions(paths, graph, positions, cell_size)
    if not merge_straight:
        return _schedule(paths, coords, limits_for, False)
    try:
        return _schedule(paths, coords, limits_for, True)
    except InfeasibleScheduleError:
        # Merging can manufacture a cycle the plan does not have. A merged
        # run is ordered against other runs at vertices it passes long after
        # it departs, so it can be made to wait for a run that departs
        # *later* than it does, and a few such constraints close a loop. On
        # one 7x7 instance with six agents a run departing at t=3 led one
        # departing at t=2. Unmerged, every constraint points forward in
        # discrete time and the only cycles are genuine simultaneous
        # rotations -- so if this one is genuine too, the call below says so.
        return _schedule(paths, coords, limits_for, False)


def _schedule(paths, coords, limits_for, merge_straight) -> TrajectorySet:
    """Build the temporal network for one choice of runs and solve it."""
    # -- variables: one per run departure ------------------------------------
    # dep[(agent, i)] is when the agent's i-th run leaves halt i. Arrival at
    # halt i+1 is dep[(agent, i)] + the run's duration, and passing the k-th
    # vertex of the run is dep[(agent, i)] + tau_k -- neither is a variable,
    # which is what keeps every constraint a plain difference.
    stays: Dict[str, List[Tuple[Hashable, int, int]]] = {
        agent: stays_of(path) for agent, path in paths.items()
    }
    halts: Dict[str, List[int]] = {}
    runs: Dict[str, List[_Run]] = {}
    for agent, agent_stays in stays.items():
        halts[agent], runs[agent] = _runs_of(
            agent_stays, coords, limits_for(agent), merge_straight
        )
    variables = [(agent, i) for agent in runs for i in range(len(runs[agent]))]
    index = {var: i for i, var in enumerate(variables)}

    # Edges (src, dst, w) meaning  t[dst] >= t[src] + w.  src=None is the
    # virtual source at time 0.
    edges: List[Tuple[Optional[int], int, float]] = [
        (None, index[var], 0.0) for var in variables
    ]
    # travel + dwell: leave halt i+1 no earlier than arriving there.
    for agent, agent_runs in runs.items():
        for i in range(len(agent_runs) - 1):
            edges.append(
                (index[(agent, i)], index[(agent, i + 1)], agent_runs[i].duration)
            )
    edges.extend(_safety_edges(stays, halts, runs, index, limits_for))

    times = _longest_path(len(variables), edges)
    return _assemble(stays, halts, runs, coords, times, index)


def _passages(stays, halts, runs):
    """Every time a run is at a vertex: ``vertex -> [(time, agent, run, tau)]``.

    A run is at its origin when it departs (the halt's last discrete time),
    at its destination when it arrives (the halt's first), and at each
    vertex it passes through at that vertex's discrete time.
    """
    passages: Dict[Hashable, List[Tuple[int, str, int, float]]] = {}
    for agent, agent_halts in halts.items():
        agent_stays = stays[agent]
        for i, run in enumerate(runs[agent]):
            origin, _first, last = agent_stays[agent_halts[i]]
            destination, first, _last = agent_stays[agent_halts[i + 1]]
            passages.setdefault(origin, []).append((last, agent, i, 0.0))
            passages.setdefault(destination, []).append((first, agent, i, run.duration))
            for vertex, time, tau in run.via:
                passages.setdefault(vertex, []).append((time, agent, i, tau))
    return passages


def _check_parking(stays, passages) -> None:
    """A vertex an agent parks on for good must not be visited afterwards."""
    for agent, agent_stays in stays.items():
        vertex, first, _last = agent_stays[-1]
        for time, other, _run, _tau in passages.get(vertex, ()):
            if other != agent and time > first:
                raise ValueError(
                    "%r parks at %r but %r visits it later; plan is not valid"
                    % (agent, vertex, other)
                )


def _safety_edges(
    stays, halts, runs, index, limits_for
) -> List[Tuple[int, int, float]]:
    """One safety constraint per pair of runs that share a vertex, and one
    per pair that comes too close without sharing one (:func:`_near_misses`).

    The discrete plan says which of the two is first at the shared vertices
    (it is the same one at all of them: two runs sharing several vertices
    are collinear, and opposite directions would be a swap). The follower's
    departure is then delayed after the leader's by :func:`_pair_delay`, so
    ``dep(follower) >= dep(leader) + delay``.
    """
    passages = _passages(stays, halts, runs)
    _check_parking(stays, passages)

    # Shared vertices, grouped by the pair of runs that share them.
    pairs: Dict[
        Tuple[Tuple[str, int], Tuple[str, int]], List[Tuple[int, int, float, float]]
    ] = {}
    for vertex, visitors in passages.items():
        for x in range(len(visitors)):
            time_a, agent_a, run_a, tau_a = visitors[x]
            for y in range(x + 1, len(visitors)):
                time_b, agent_b, run_b, tau_b = visitors[y]
                if agent_a == agent_b:
                    continue
                key_a, key_b = (agent_a, run_a), (agent_b, run_b)
                if key_a <= key_b:
                    pairs.setdefault((key_a, key_b), []).append(
                        (time_a, time_b, tau_a, tau_b)
                    )
                else:
                    pairs.setdefault((key_b, key_a), []).append(
                        (time_b, time_a, tau_b, tau_a)
                    )

    edges: List[Tuple[int, int, float]] = []
    for (key_a, key_b), shared in pairs.items():
        a_first = [time_a < time_b for time_a, time_b, _, _ in shared]
        if all(a_first):
            leader, follower = key_a, key_b
            order = max(tau_a - tau_b for _, _, tau_a, tau_b in shared)
        elif not any(a_first):
            leader, follower = key_b, key_a
            order = max(tau_b - tau_a for _, _, tau_a, tau_b in shared)
        else:
            raise ValueError(
                "%r and %r pass each other head-on between shared vertices; "
                "plan is not valid" % (key_a[0], key_b[0])
            )
        edges.append(_ordered(leader, follower, order, runs, index, limits_for))
    for leader, follower, order, tied in _near_misses(
        stays, halts, runs, limits_for, pairs
    ):
        # A simultaneous crossing has no order in the discrete plan, and the
        # wrong guess can close a cycle that the other one avoids: three
        # agents trading places at once, two of them crossing diagonally.
        # So it follows the order the constraints so far already imply.
        if tied and _reaches(edges, index[follower], index[leader]):
            leader, follower, order = follower, leader, -order
        edges.append(_ordered(leader, follower, order, runs, index, limits_for))
    return edges


def _reaches(edges, source: int, target: int) -> bool:
    """Whether ``target`` is constrained, through ``edges``, to follow ``source``."""
    after: Dict[int, List[int]] = {}
    for src, dst, _w in edges:
        after.setdefault(src, []).append(dst)
    seen, frontier = {source}, [source]
    while frontier:
        for nxt in after.get(frontier.pop(), ()):
            if nxt == target:
                return True
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return False


def _ordered(leader, follower, order, runs, index, limits_for):
    """The edge ``dep(follower) >= dep(leader) + delay`` for one ordered pair."""
    lim_l, lim_f = limits_for(leader[0]), limits_for(follower[0])
    explicit = [
        lim.safety_time for lim in (lim_l, lim_f) if lim.safety_time is not None
    ]
    if explicit:
        delay = order + max(explicit)
    else:
        wanted = max(lim_l.safety_distance, lim_f.safety_distance)
        delay = _pair_delay(
            runs[leader[0]][leader[1]],
            runs[follower[0]][follower[1]],
            order,
            wanted,
        )
    return index[leader], index[follower], delay


def _closest_approach(p1: Point, q1: Point, p2: Point, q2: Point):
    """``(distance, s, t)``: how close segments ``p1q1`` and ``p2q2`` come, at
    fractions ``s`` and ``t`` along each (Ericson, *Real-Time Collision
    Detection*, 5.1.9). Neither segment may be a point."""

    def dot(u, v):
        return sum(x * y for x, y in zip(u, v))

    def clamp(x):
        return min(1.0, max(0.0, x))

    d1 = [y - x for x, y in zip(p1, q1)]
    d2 = [y - x for x, y in zip(p2, q2)]
    r = [x - y for x, y in zip(p1, p2)]
    a, b, e = dot(d1, d1), dot(d1, d2), dot(d2, d2)
    c, f = dot(d1, r), dot(d2, r)
    denominator = a * e - b * b
    s = clamp((b * f - c * e) / denominator) if denominator > 1e-12 * a * e else 0.0
    t = (b * s + f) / e
    if t < 0.0:
        t, s = 0.0, clamp(-c / a)
    elif t > 1.0:
        t, s = 1.0, clamp((b - c) / a)
    return distance(lerp(p1, q1, s), lerp(p2, q2, t)), s, t


def _near_misses(stays, halts, runs, limits_for, shared):
    """Pairs of runs that come too close without sharing a vertex.

    Yields ``(leader, follower, order, tied)`` as a shared vertex would,
    with the point of closest approach standing in for the vertex: the
    leader is whichever run the discrete plan has there first (its discrete
    time is interpolated along the run), and ``order`` is the follower's
    least delay that keeps it second there. ``tied`` marks a simultaneous
    crossing, which the discrete plan does not order at all. "Too close" is the pair's
    safety distance; under an explicit ``safety_time``, which is a rule
    about shared points, only an actual crossing counts.

    Candidates come from a sweep over the runs' bounding boxes along the
    first axis, so a plan whose runs are far apart costs one sort.
    """
    reach = max((limits_for(agent).safety_distance for agent in runs), default=0.0)
    boxes = []
    for agent, agent_runs in runs.items():
        for i, run in enumerate(agent_runs):
            low = [min(x, y) - reach for x, y in zip(run.origin, run.destination)]
            high = [max(x, y) + reach for x, y in zip(run.origin, run.destination)]
            departs = stays[agent][halts[agent][i]][2]
            arrives = stays[agent][halts[agent][i + 1]][1]
            boxes.append((low, high, (agent, i), departs, arrives - departs))
    boxes.sort(key=lambda box: box[0][0])
    for x, (low_a, high_a, key_a, dep_a, steps_a) in enumerate(boxes):
        for low_b, high_b, key_b, dep_b, steps_b in boxes[x + 1 :]:
            if low_b[0] > high_a[0]:
                break
            if key_a[0] == key_b[0] or (min(key_a, key_b), max(key_a, key_b)) in shared:
                continue
            if any(lo > hi for lo, hi in zip(low_b, high_a)) or any(
                lo > hi for lo, hi in zip(low_a, high_b)
            ):
                continue
            lim_a, lim_b = limits_for(key_a[0]), limits_for(key_b[0])
            run_a, run_b = runs[key_a[0]][key_a[1]], runs[key_b[0]][key_b[1]]
            gap, s, t = _closest_approach(
                run_a.origin, run_a.destination, run_b.origin, run_b.destination
            )
            if lim_a.safety_time is not None or lim_b.safety_time is not None:
                if gap > 1e-9:
                    continue
            elif gap >= max(lim_a.safety_distance, lim_b.safety_distance) - 1e-9:
                continue
            tau_a = run_a.profile.time_to_travel(s * run_a.profile.length)
            tau_b = run_b.profile.time_to_travel(t * run_b.profile.length)
            # Equal times there go to the earlier departure, so that every
            # edge points forward in discrete time; a full tie, to the name.
            there_a = (dep_a + s * steps_a, dep_a, key_a)
            there_b = (dep_b + t * steps_b, dep_b, key_b)
            tied = there_a[:2] == there_b[:2]
            if there_a <= there_b:
                yield key_a, key_b, tau_a - tau_b, tied
            else:
                yield key_b, key_a, tau_b - tau_a, tied


def _assemble(stays, halts, runs, coords, times, index) -> TrajectorySet:
    """Turn departure times back into per-agent stays and segments."""
    trajectories: Dict[str, Trajectory] = {}
    for agent, agent_halts in halts.items():
        agent_runs = runs[agent]
        stay_objects: List[Stay] = []
        segments: List[Segment] = []
        arrive = 0.0
        for h, k in enumerate(agent_halts):
            vertex = stays[agent][k][0]
            last = h == len(agent_runs)
            depart = math.inf if last else times[index[(agent, h)]]
            stay_objects.append(Stay(vertex, coords[vertex], arrive, depart))
            if not last:
                run = agent_runs[h]
                end = depart + run.duration
                segments.append(
                    Segment(
                        depart,
                        end,
                        run.origin,
                        run.destination,
                        run.profile,
                        via=tuple(v for v, _first, _tau in run.via),
                    )
                )
                arrive = end
        trajectories[agent] = Trajectory(agent, stay_objects, segments)
    return TrajectorySet(trajectories)


def _longest_path(n: int, edges) -> List[float]:
    """Earliest times satisfying every ``t[dst] >= t[src] + w``.

    Bellman-Ford, maximising. Converges in at most ``n`` passes; a change on
    the pass after that is a positive cycle, i.e. no finite schedule.
    """
    times = [0.0] * n
    for _ in range(n + 1):
        changed = False
        for src, dst, w in edges:
            base = 0.0 if src is None else times[src]
            if times[dst] < base + w - 1e-12:
                times[dst] = base + w
                changed = True
        if not changed:
            return times
    raise InfeasibleScheduleError(
        "the safety margin exceeds a move's duration on a cycle of agents "
        "that must rotate simultaneously; reduce safety_time/safety_distance "
        "or raise v_max"
    )
