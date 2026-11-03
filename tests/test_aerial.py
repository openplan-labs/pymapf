"""Quadrotor docking: assignment, routing and scheduling on a volume."""

import math
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pytest  # noqa: E402

from pymapf.aerial import (  # noqa: E402
    Airspace,
    DockingStation,
    Quadrotor,
    assign_stations,
    hovering_fleet,
    plan_docking,
)
from pymapf.kinodynamic import KinematicLimits  # noqa: E402


def small_airspace(**kwargs):
    defaults = dict(
        cells=(7, 6, 5),
        cell_size=1.0,
        pads=[(1, 1), (1, 3), (1, 5), (4, 1), (4, 3), (4, 5)],
    )
    defaults.update(kwargs)
    return Airspace.build(**defaults)


# --------------------------------------------------------------------------
# the world
# --------------------------------------------------------------------------


def test_ground_is_blocked_except_at_the_pads():
    airspace = small_airspace()
    depth, height, width = airspace.grid.shape
    assert (depth, height, width) == (5, 6, 7)
    for r in range(height):
        for c in range(width):
            assert airspace.grid.is_free((0, r, c)) == (
                (r, c) in {(1, 1), (1, 3), (1, 5), (4, 1), (4, 3), (4, 5)}
            )
            assert airspace.grid.is_free((1, r, c))


def test_obstacles_block_every_layer_they_span_and_pads_cannot_be_inside_one():
    airspace = small_airspace(obstacles=[((2, 2, 1), (3, 4, 3))])
    assert not airspace.grid.is_free((2, 3, 3))
    assert airspace.grid.is_free((4, 3, 3))  # above the block
    assert airspace.grid.is_free((1, 1, 3))
    with pytest.raises(ValueError, match="inside an obstacle"):
        small_airspace(obstacles=[((1, 1, 0), (1, 1, 0))])


def test_a_floor_opens_only_the_approach_corridors_below_it():
    airspace = small_airspace(floor=2)
    assert airspace.floor == 2
    for layer in (0, 1):
        assert airspace.grid.is_free((layer, 1, 3))  # above a pad
        assert not airspace.grid.is_free((layer, 2, 3))  # not above a pad
    assert airspace.grid.is_free((2, 2, 3))
    fleet = hovering_fleet(airspace, n=3, seed=0, min_layer=2)
    plan = plan_docking(fleet, airspace, algorithm="lacam")
    for q in fleet:
        trajectory = plan.trajectories[q.name]
        pad = plan.station_of(q.name).position
        t = 0.0
        while t < trajectory.arrival_time:
            x, y, z = trajectory.position_at(t)
            if z < 2 * airspace.cell_size - 1e-9:
                assert (x, y) == pytest.approx(pad[:2])
            t += 0.05
    with pytest.raises(ValueError, match="floor"):
        small_airspace(floor=5)


def test_voxel_and_position_are_inverse_and_scale_with_cell_size():
    airspace = small_airspace(cell_size=2.5)
    for voxel in [(0, 1, 1), (3, 5, 6), (2, 0, 4)]:
        assert airspace.voxel(airspace.position(voxel)) == voxel
    assert airspace.position((2, 1, 3)) == (7.5, 2.5, 5.0)
    assert airspace.extent == (15.0, 12.5, 10.0)
    assert [s.position for s in airspace.stations()][:2] == [
        (2.5, 2.5, 0.0),
        (7.5, 2.5, 0.0),
    ]


def test_hovering_fleet_is_distinct_free_and_airborne():
    airspace = small_airspace()
    fleet = hovering_fleet(airspace, n=6, seed=3, min_layer=2)
    voxels = [airspace.voxel(q.position) for q in fleet]
    assert len(set(voxels)) == 6
    assert all(airspace.grid.is_free(v) and v[0] >= 2 for v in voxels)
    assert hovering_fleet(airspace, n=6, seed=3) == fleet  # seeded
    with pytest.raises(ValueError):
        hovering_fleet(airspace, n=10_000)


# --------------------------------------------------------------------------
# assignment
# --------------------------------------------------------------------------


def test_assignment_is_optimal_and_refuses_too_few_stations():
    stations = [DockingStation("s%d" % k, (float(k * 3), 0.0, 0.0)) for k in range(3)]
    # Index order would send q0 across q1; the optimum is the crossing-free one.
    fleet = [Quadrotor("q0", (6.0, 2.0, 4.0)), Quadrotor("q1", (0.0, 2.0, 4.0))]
    assert assign_stations(fleet, stations) == {"q0": "s2", "q1": "s0"}
    with pytest.raises(ValueError, match="only 1 stations"):
        assign_stations(fleet, stations[:1])


def test_assignment_through_the_airspace_accounts_for_obstacles():
    # A wall across the middle with a gap at one end: the straight line lies,
    # the flight distance does not.
    airspace = Airspace.build(
        cells=(7, 7, 4),
        pads=[(6, 0), (6, 6)],
        obstacles=[((3, 0, 0), (3, 5, 3))],
    )
    stations = airspace.stations()
    fleet = [Quadrotor("left", (0.0, 0.0, 2.0)), Quadrotor("right", (6.0, 0.0, 2.0))]
    straight = assign_stations(fleet, stations)
    routed = assign_stations(fleet, stations, airspace)
    assert straight == {"left": "pad-0", "right": "pad-1"}
    # Through the gap at column 6, both go right first; the left vehicle's
    # cheapest pad is still pad-0 only if it detours; the optimal fleet plan
    # keeps the total lowest, which the cost check below pins.
    from pymapf.aerial.docking import _flight_distances

    cost = {
        name: {
            s.name: _flight_distances(airspace, airspace.voxel(s.position))[
                airspace.voxel(q.position)
            ]
            for s in stations
        }
        for q in fleet
        for name in [q.name]
    }
    best = min(
        cost["left"][a] + cost["right"][b]
        for a in ("pad-0", "pad-1")
        for b in ("pad-0", "pad-1")
        if a != b
    )
    assert cost["left"][routed["left"]] + cost["right"][
        routed["right"]
    ] == pytest.approx(best)


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def docked():
    airspace = small_airspace()
    fleet = hovering_fleet(airspace, n=6, seed=1, min_layer=2, radius=0.3)
    plan = plan_docking(
        fleet, airspace, limits=KinematicLimits(v_max=2.0, a_max=1.5), clearance=0.2
    )
    return airspace, fleet, plan


def test_every_quadrotor_lands_on_its_assigned_pad(docked):
    airspace, fleet, plan = docked
    assert set(plan.assignment) == {q.name for q in fleet}
    assert len(set(plan.assignment.values())) == len(fleet)
    for q in fleet:
        trajectory = plan.trajectories[q.name]
        assert trajectory.position_at(0.0) == pytest.approx(q.position)
        station = plan.station_of(q.name)
        assert trajectory.position_at(1e9) == pytest.approx(station.position)
        assert station.position[2] == 0.0


def test_the_ground_is_only_touched_at_the_pad(docked):
    airspace, fleet, plan = docked
    for q in fleet:
        trajectory = plan.trajectories[q.name]
        pad = plan.station_of(q.name).position
        t = 0.0
        while t < trajectory.arrival_time:
            x, y, z = trajectory.position_at(t)
            if z < 0.5 * airspace.cell_size:
                # Below the first flight layer only on the final descent,
                # straight above the assigned pad.
                assert (x, y) == pytest.approx(pad[:2])
            t += 0.05
        # The last leg is a vertical descent.
        last = trajectory.segments[-1]
        assert last.origin[:2] == pytest.approx(last.destination[:2])
        assert last.origin[2] > last.destination[2]


def test_separation_speed_and_acceleration_hold(docked):
    airspace, fleet, plan = docked
    assert plan.is_safe()
    separation = plan.min_separation(0.02)[0]
    assert separation >= 0.6 - 1e-9  # two bodies of 0.3
    assert plan.trajectories.max_speed(0.02) <= 2.0 + 1e-9
    # Acceleration by finite differences.
    for q in fleet:
        trajectory = plan.trajectories[q.name]
        t, dt = 0.0, 0.02
        while t < trajectory.arrival_time:
            v0 = trajectory.velocity_at(t)
            v1 = trajectory.velocity_at(t + dt)
            accel = math.dist(v0, v1) / dt
            assert accel <= 1.5 + 0.2  # profile corners sampled across
            t += dt


def test_straight_stretches_are_single_runs(docked):
    _, fleet, plan = docked
    total_moves = sum(len(plan.solution.paths[q.name]) - 1 for q in fleet)
    total_segments = sum(len(plan.trajectories[q.name].segments) for q in fleet)
    assert total_segments < total_moves
    assert any(
        segment.via for q in fleet for segment in plan.trajectories[q.name].segments
    )


def test_summary_reports_the_three_optimisations(docked):
    _, _, plan = docked
    summary = plan.summary()
    assert summary["quadrotors"] == 6 and summary["stations"] == 6
    assert summary["assignment_cost"] > 0
    assert summary["min_separation"] >= summary["required_separation"]
    assert summary["makespan"] == plan.makespan
    assert "expansions" in summary


def test_an_explicit_assignment_is_honoured_and_a_clash_refused():
    airspace = small_airspace()
    fleet = hovering_fleet(airspace, n=2, seed=4)
    forced = {"q0": "pad-5", "q1": "pad-0"}
    plan = plan_docking(fleet, airspace, assignment=forced, algorithm="lacam")
    assert plan.assignment == forced
    with pytest.raises(ValueError, match="same station"):
        plan_docking(fleet, airspace, assignment={"q0": "pad-0", "q1": "pad-0"})


def test_ill_posed_instances_are_refused():
    airspace = small_airspace()
    fleet = hovering_fleet(airspace, n=2, seed=0)
    with pytest.raises(ValueError, match="exceeds"):
        plan_docking(fleet, airspace, clearance=1.0)  # 0.6 + 1.0 > one 1 m voxel
    grounded = [Quadrotor("g", (2.0, 2.0, 0.0))]  # ground, not a pad
    with pytest.raises(ValueError, match="not free airspace"):
        plan_docking(grounded, airspace)
    twins = [Quadrotor("a", (2.0, 2.0, 2.0)), Quadrotor("b", (2.0, 2.0, 2.0))]
    with pytest.raises(ValueError, match="share voxel"):
        plan_docking(twins, airspace)
    with pytest.raises(ValueError, match="stations"):
        plan_docking(hovering_fleet(airspace, n=7, seed=0), airspace)


def test_a_vehicle_off_its_voxel_centre_is_refused_not_snapped():
    """Routes start at voxel centres; the vehicle at (2.2, 2.2, 3.7) used to
    start its trajectory at (1.5, 1.5, 3.0), a metre from where it was."""
    airspace = Airspace.build(cells=(6, 6, 4), cell_size=1.5, pads=[(1, 1), (4, 4)])
    fleet = [Quadrotor("q0", (2.2, 2.2, 3.7)), Quadrotor("q1", (6.0, 6.0, 4.5))]
    with pytest.raises(ValueError, match="q0 hovers .* from the centre"):
        plan_docking(fleet, airspace)
    centred = [Quadrotor("q0", (1.5, 1.5, 3.0)), fleet[1]]
    plan = plan_docking(centred, airspace)
    assert plan.trajectories["q0"].position_at(0.0) == pytest.approx((1.5, 1.5, 3.0))
    pads = [DockingStation("near", (1.5, 1.5, 0.2)), DockingStation("far", (6, 6, 0))]
    with pytest.raises(ValueError, match="station near is"):
        plan_docking(centred, airspace, stations=pads)


def test_a_heavier_vehicle_keeps_its_own_limits():
    airspace = small_airspace()
    fleet = hovering_fleet(airspace, n=3, seed=2)
    slow = Quadrotor(
        fleet[0].name, fleet[0].position, radius=0.3, limits=KinematicLimits(v_max=0.5)
    )
    plan = plan_docking([slow] + fleet[1:], airspace, algorithm="lacam")
    speeds = {
        q.name: max(
            math.dist(plan.trajectories[q.name].velocity_at(t * 0.05), (0, 0, 0))
            for t in range(int(plan.makespan / 0.05) + 1)
        )
        for q in plan.quadrotors
    }
    assert speeds[slow.name] <= 0.5 + 1e-9
    assert max(speeds.values()) > 0.5
