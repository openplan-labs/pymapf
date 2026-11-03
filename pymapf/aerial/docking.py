"""Centralised, kinematics-aware docking of a quadrotor fleet.

A fleet is hovering; a set of docking stations sits on the ground; every
quadrotor must land on one without any two coming closer than their bodies
allow. Three optimisations, stacked:

1. **Assignment** -- which quadrotor takes which station. Solved exactly
   with the Hungarian algorithm on flight distance, so the fleet as a whole
   travels least and nobody crosses the formation to reach a pad someone
   else is above (Turpin, Michael and Kumar's CAPT makes the case that this
   choice is what removes most conflicts before any trajectory is planned).
2. **Routing** -- a MAPF solver plans conflict-free routes through the
   :class:`~pymapf.core.voxel.VoxelGrid` the airspace is discretised into.
   CBS by default, so the routes are optimal for the assignment.
3. **Timing** -- :func:`~pymapf.kinodynamic.plan_trajectories` turns the
   routes into trajectories under the fleet's speed and acceleration limits,
   flying straight stretches as single runs and deriving the delay at every
   hand-over from the two vehicles' actual motion. The result is a set of
   continuous trajectories with a guaranteed separation at every shared
   waypoint, and a sampled separation over the whole flight as the check.

The ground layer of the airspace is blocked everywhere except the pads, so a
route can only touch the ground at a station: the last leg of every
trajectory is a vertical descent onto its pad, and no vehicle ever passes
over a pad at ground level on its way somewhere else.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..core.assignment import hungarian
from ..core.solver import Agent, MAPFProblem, Solution
from ..core.voxel import VoxelGrid
from ..kinodynamic import KinematicLimits, TrajectorySet, plan_trajectories

__all__ = [
    "Airspace",
    "DockingPlan",
    "DockingStation",
    "Quadrotor",
    "assign_stations",
    "hovering_fleet",
    "plan_docking",
]

Point = Tuple[float, float, float]
Voxel = Tuple[int, int, int]


# --------------------------------------------------------------------------
# the world
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DockingStation:
    """A pad on the ground. ``position`` is ``(x, y, z)`` in metres, at the
    centre of a voxel of the airspace it is used in (``Airspace.position``).
    """

    name: str
    position: Point


@dataclass(frozen=True)
class Quadrotor:
    """A vehicle: where it hovers, how big it is, and optionally its own
    limits (a heavier vehicle is slower; the fleet default applies otherwise).
    :func:`plan_docking` needs ``position`` at a voxel centre
    (``Airspace.position``) and refuses one that is not.
    """

    name: str
    position: Point
    radius: float = 0.3
    limits: Optional[KinematicLimits] = None


@dataclass(frozen=True)
class Airspace:
    """A box of air discretised into voxels the solvers can plan on.

    A voxel ``(layer, row, col)`` is centred at ``(col, row, layer) *
    cell_size`` metres: ``x`` runs along columns, ``y`` along rows and ``z``
    along layers, with layer 0 on the ground. Build one with
    :meth:`Airspace.build`.

    Attributes:
        grid: the :class:`~pymapf.core.voxel.VoxelGrid` -- ground blocked
            except at the pads, obstacles blocked at every layer they span.
        cell_size: metres per voxel edge. The separation guarantee at
            shared waypoints extends to vehicles on *adjacent* waypoints only
            when the safety distance is at most one cell, which
            :func:`plan_docking` enforces.
        pads: the ground voxels that are docking stations, in order.
        floor: the lowest layer of free flight; below it only the approach
            corridors above the pads are open.
    """

    grid: VoxelGrid
    cell_size: float
    pads: Tuple[Voxel, ...]
    floor: int = 1

    @classmethod
    def build(
        cls,
        cells: Tuple[int, int, int],
        cell_size: float = 1.0,
        pads: Iterable[Tuple[int, int]] = (),
        obstacles: Iterable[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = (),
        floor: int = 1,
    ) -> "Airspace":
        """Args:
        cells: ``(width, depth, height)`` -- columns along x, rows along y,
            layers along z including the ground layer.
        cell_size: metres per voxel.
        pads: ``(row, col)`` ground cells that are docking stations. Every
            other ground cell is blocked.
        obstacles: axis-aligned boxes ``((row0, col0, layer0), (row1, col1,
            layer1))``, inclusive, blocked at every voxel they cover -- a
            building, a mast, a tree line.
        floor: the lowest layer of free flight. Layers below it are open
            only directly above a pad, so with ``floor=2`` every vehicle
            transits at two cells or more and descends onto its pad through
            a vertical approach corridor -- the rule an operator wants near
            people and parked aircraft. ``1`` (the default) lets vehicles fly
            just above the ground.
        """
        width, depth, height = cells
        if width < 1 or depth < 1 or height < 2:
            raise ValueError(
                "an airspace needs at least one column, one row and two layers"
            )
        if not 1 <= floor < height:
            raise ValueError("floor must be a layer between 1 and %d" % (height - 1))
        pads = tuple(dict.fromkeys((int(r), int(c)) for r, c in pads))
        for r, c in pads:
            if not (0 <= r < depth and 0 <= c < width):
                raise ValueError(
                    "pad (%d, %d) is outside the %dx%d footprint" % (r, c, depth, width)
                )
        grid = VoxelGrid(_blocked_voxels(width, depth, height, pads, obstacles, floor))
        for r, c in pads:
            if not grid.is_free((0, r, c)):
                raise ValueError("pad (%d, %d) is inside an obstacle" % (r, c))
        return cls(grid, float(cell_size), tuple((0, r, c) for r, c in pads), floor)

    # -- geometry -------------------------------------------------------
    def position(self, voxel: Voxel) -> Point:
        """Metres of a voxel's centre."""
        layer, row, col = voxel
        return (col * self.cell_size, row * self.cell_size, layer * self.cell_size)

    def voxel(self, point: Sequence[float]) -> Voxel:
        """The voxel whose centre is nearest to ``point`` (metres)."""
        x, y, z = point
        return (
            int(round(z / self.cell_size)),
            int(round(y / self.cell_size)),
            int(round(x / self.cell_size)),
        )

    @property
    def positions(self) -> Dict[Voxel, Point]:
        """Every voxel's centre, the layout :func:`plan_trajectories` needs."""
        return {voxel: self.position(voxel) for voxel in self.grid.cells()}

    @property
    def extent(self) -> Point:
        """``(x, y, z)`` size in metres, measured between outer voxel centres."""
        depth, height, width = self.grid.shape
        return (
            (width - 1) * self.cell_size,
            (height - 1) * self.cell_size,
            (depth - 1) * self.cell_size,
        )

    def stations(self) -> List[DockingStation]:
        """One :class:`DockingStation` per pad, named ``pad-0``, ``pad-1``…"""
        return [
            DockingStation("pad-%d" % k, self.position(pad))
            for k, pad in enumerate(self.pads)
        ]


def _blocked_voxels(width, depth, height, pads, obstacles, floor):
    """``voxels[layer][row][col]`` truthy where flight is not allowed."""
    blocked = [
        [[layer < floor for _c in range(width)] for _r in range(depth)]
        for layer in range(height)
    ]
    for r, c in pads:
        for layer in range(floor):
            blocked[layer][r][c] = False
    for (r0, c0, z0), (r1, c1, z1) in obstacles:
        layers = range(max(0, min(z0, z1)), min(height - 1, max(z0, z1)) + 1)
        rows = range(max(0, min(r0, r1)), min(depth - 1, max(r0, r1)) + 1)
        cols = range(max(0, min(c0, c1)), min(width - 1, max(c0, c1)) + 1)
        for z in layers:
            for r in rows:
                for c in cols:
                    blocked[z][r][c] = True
    return blocked


def hovering_fleet(
    airspace: Airspace,
    n: int,
    seed: int = 0,
    min_layer: int = 2,
    radius: float = 0.3,
) -> List[Quadrotor]:
    """``n`` quadrotors hovering at distinct free voxels at or above ``min_layer``."""
    depth, height, width = airspace.grid.shape
    candidates = [
        (z, r, c)
        for z in range(min_layer, depth)
        for r in range(height)
        for c in range(width)
        if airspace.grid.is_free((z, r, c))
    ]
    if n > len(candidates):
        raise ValueError(
            "%d quadrotors do not fit in %d free voxels" % (n, len(candidates))
        )
    rng = random.Random(seed)
    chosen = rng.sample(candidates, n)
    return [
        Quadrotor("q%d" % k, airspace.position(voxel), radius=radius)
        for k, voxel in enumerate(chosen)
    ]


# --------------------------------------------------------------------------
# assignment
# --------------------------------------------------------------------------


def _flight_distances(airspace: Airspace, goal: Voxel) -> Dict[Voxel, float]:
    from ..algorithms.search import distance_table

    table = distance_table(airspace.grid, goal, allow_diagonals=False)
    return {voxel: steps * airspace.cell_size for voxel, steps in table.items()}


def assign_stations(
    quadrotors: Sequence[Quadrotor],
    stations: Sequence[DockingStation],
    airspace: Optional[Airspace] = None,
) -> Dict[str, str]:
    """The assignment of quadrotors to stations that minimises total flight
    distance -- exactly, with the Hungarian algorithm.

    With an ``airspace`` the distance is the length of the shortest route
    through it (obstacles and the blocked ground included), so a pad behind
    a building costs what it really costs; without one it is the straight
    line. There must be at least as many stations as quadrotors; surplus
    stations stay empty.

    Returns ``{quadrotor name: station name}``.
    """
    if len(stations) < len(quadrotors):
        raise ValueError(
            "%d quadrotors but only %d stations" % (len(quadrotors), len(stations))
        )
    if airspace is None:
        cost = [
            [math.dist(q.position, s.position) for s in stations] for q in quadrotors
        ]
    else:
        cost = []
        tables = [
            _flight_distances(airspace, airspace.voxel(s.position)) for s in stations
        ]
        for q in quadrotors:
            start = airspace.voxel(q.position)
            cost.append([table.get(start, math.inf) for table in tables])
    try:
        columns = hungarian(cost)
    except ValueError as exc:
        raise ValueError("some quadrotor cannot reach any station: %s" % exc) from exc
    return {q.name: stations[j].name for q, j in zip(quadrotors, columns)}


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------


@dataclass
class DockingPlan:
    """Assignment, routes and trajectories for one docking operation.

    ``trajectories[name]`` is the quadrotor's continuous motion in metres and
    seconds; ``solution`` the discrete routes it was scheduled from;
    ``assignment`` who lands where.
    """

    airspace: Airspace
    quadrotors: List[Quadrotor]
    stations: List[DockingStation]
    assignment: Dict[str, str]
    solution: Solution
    trajectories: TrajectorySet
    limits: KinematicLimits
    assignment_cost: float = 0.0
    metadata: Dict[str, float] = field(default_factory=dict)

    @property
    def makespan(self) -> float:
        """Seconds until the last quadrotor is docked."""
        return self.trajectories.makespan

    def arrival_times(self) -> Dict[str, float]:
        return {
            name: self.trajectories[name].arrival_time
            for name in self.trajectories.agents
        }

    def required_separation(self) -> float:
        """The closest two vehicles may come: the largest pair of radii."""
        radii = sorted((q.radius for q in self.quadrotors), reverse=True)
        return sum(radii[:2]) if len(radii) >= 2 else 0.0

    def min_separation(self, dt: float = 0.05) -> Tuple[float, float, Tuple[str, str]]:
        """Sampled closest approach: ``(distance, time, (a, b))``."""
        return self.trajectories.min_separation(dt)

    def is_safe(self, dt: float = 0.05) -> bool:
        """Whether the sampled closest approach clears the vehicles' bodies."""
        return self.min_separation(dt)[0] >= self.required_separation() - 1e-9

    def station_of(self, name: str) -> DockingStation:
        wanted = self.assignment[name]
        return next(s for s in self.stations if s.name == wanted)

    def summary(self) -> Dict[str, float]:
        separation, when, _pair = self.min_separation()
        return {
            "quadrotors": len(self.quadrotors),
            "stations": len(self.stations),
            "assignment_cost": self.assignment_cost,
            "route_cost": self.solution.sum_of_costs * self.airspace.cell_size,
            "makespan": self.makespan,
            "sum_of_arrival_times": self.trajectories.sum_of_arrival_times,
            "min_separation": separation,
            "min_separation_at": when,
            "required_separation": self.required_separation(),
            "max_speed": self.trajectories.max_speed(),
            **self.metadata,
        }


def _centred_voxel(airspace: Airspace, point: Sequence[float], who: str) -> Voxel:
    """The voxel ``point`` is the centre of; refuse a point that is not one.

    The routes start and end at voxel centres. Snapping to the nearest one
    quietly moved a quadrotor hovering at (2.2, 2.2, 3.7) to (1.5, 1.5, 3.0)
    -- a metre away, through air no constraint had looked at -- and the plan
    began there as if it had always been.
    """
    voxel = airspace.voxel(point)
    centre = airspace.position(voxel)
    off = math.dist(tuple(float(x) for x in point), centre)
    if off > 1e-6 * max(1.0, airspace.cell_size):
        raise ValueError(
            "%s at %s, %.3f m from the centre %s of voxel %s; plans start and end "
            "at voxel centres (Airspace.position), so place it on one or fly it "
            "there first" % (who, tuple(point), off, centre, voxel)
        )
    return voxel


def plan_docking(
    quadrotors: Sequence[Quadrotor],
    airspace: Airspace,
    stations: Optional[Sequence[DockingStation]] = None,
    limits: Optional[KinematicLimits] = None,
    clearance: float = 0.2,
    algorithm: str = "cbs",
    time_limit: float = 30.0,
    assignment: Optional[Mapping[str, str]] = None,
    merge_straight: bool = True,
) -> DockingPlan:
    """Assign, route and schedule a fleet onto the ground.

    Args:
        quadrotors: the fleet, hovering at free voxels of ``airspace``.
        airspace: the discretised volume; its pads are the stations unless
            ``stations`` is given.
        limits: fleet-wide ``v_max``/``a_max`` (default 2 m/s, 1.5 m/s²).
            The safety distance is derived from the bodies: a quadrotor's
            own is ``2 * radius + clearance``, and a hand-over between two
            vehicles honours the larger of theirs. A quadrotor with its own
            ``limits`` keeps its speed and acceleration.
        clearance: metres of air required between two bodies at a hand-over.
        algorithm: any registered solver; CBS gives routes optimal for the
            assignment, LaCAM is the fast fallback for large fleets.
        time_limit: seconds for the solver.
        assignment: ``{quadrotor: station}`` to use instead of the optimal one.
        merge_straight: fly straight stretches as single runs (see
            :func:`~pymapf.kinodynamic.plan_trajectories`).

    Raises:
        ValueError: on an ill-posed instance -- a vehicle outside the free
            airspace or off the centre of its voxel, two on the same voxel, a
            safety distance larger than a voxel, more vehicles than stations.
        RuntimeError: when the solver finds no conflict-free routes in time.
    """
    stations = list(stations if stations is not None else airspace.stations())
    fleet = list(quadrotors)
    if not fleet:
        raise ValueError("no quadrotors")
    shared = limits or KinematicLimits(v_max=2.0, a_max=1.5)

    def limits_for(q: Quadrotor) -> KinematicLimits:
        own = q.limits or shared
        return KinematicLimits(
            own.v_max, own.a_max, safety_distance=2 * q.radius + clearance
        )

    per_agent = {q.name: limits_for(q) for q in fleet}
    widest = max(lim.safety_distance for lim in per_agent.values())
    if widest > airspace.cell_size + 1e-9:
        raise ValueError(
            "a safety distance of %.2f m exceeds the %.2f m voxel: vehicles on "
            "adjacent waypoints could be closer than that. Use a coarser airspace "
            "or a smaller clearance." % (widest, airspace.cell_size)
        )

    starts: Dict[Voxel, str] = {}
    for q in fleet:
        voxel = _centred_voxel(airspace, q.position, "%s hovers" % q.name)
        if not airspace.grid.is_free(voxel):
            raise ValueError(
                "%s hovers at %s, which is not free airspace" % (q.name, voxel)
            )
        if voxel in starts:
            raise ValueError(
                "%s and %s share voxel %s" % (starts[voxel], q.name, voxel)
            )
        starts[voxel] = q.name

    if assignment is None:
        assignment = assign_stations(fleet, stations, airspace)
    by_name = {s.name: s for s in stations}
    goals = {
        q.name: _centred_voxel(
            airspace,
            by_name[assignment[q.name]].position,
            "station %s is" % assignment[q.name],
        )
        for q in fleet
    }
    if len(set(goals.values())) != len(goals):
        raise ValueError("the assignment sends two quadrotors to the same station")
    cost = sum(
        _flight_distances(airspace, goals[q.name]).get(
            airspace.voxel(q.position), math.inf
        )
        for q in fleet
    )

    problem = MAPFProblem(
        airspace.grid,
        [Agent(q.name, airspace.voxel(q.position), goals[q.name]) for q in fleet],
        allow_diagonals=False,
    )
    from .. import solve

    solution = solve(problem, algorithm, time_limit=time_limit)
    if solution is None:
        raise RuntimeError(
            "%s found no conflict-free routes within %.1f s" % (algorithm, time_limit)
        )
    trajectories = plan_trajectories(
        solution,
        positions=airspace.positions,
        limits=shared,
        limits_by_agent=per_agent,
        merge_straight=merge_straight,
    )
    return DockingPlan(
        airspace=airspace,
        quadrotors=fleet,
        stations=stations,
        assignment=dict(assignment),
        solution=solution,
        trajectories=trajectories,
        limits=shared,
        assignment_cost=cost,
        metadata={
            "expansions": float(solution.expansions),
            "solver_runtime": solution.runtime,
        },
    )
