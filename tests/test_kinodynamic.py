"""From discrete plans to kinematically feasible trajectories."""

import math
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pytest

import pymapf
from pymapf import ExplicitGraph, Solution
from pymapf.kinodynamic import (
    InfeasibleScheduleError,
    KinematicLimits,
    MotionProfile,
    plan_trajectories,
    stays_of,
)

# --------------------------------------------------------------------------
# motion profiles
# --------------------------------------------------------------------------


def test_kinematic_profile_is_a_straight_line_at_v_max():
    profile = MotionProfile(3.0, v_max=1.5)
    assert profile.duration == pytest.approx(2.0)
    assert profile.travelled(1.0) == pytest.approx(1.5)
    assert profile.speed(1.0) == pytest.approx(1.5)
    assert profile.time_to_travel(1.5) == pytest.approx(1.0)


def test_trapezoid_reaches_cruise_and_respects_both_limits():
    profile = MotionProfile(4.0, v_max=1.0, a_max=2.0)
    assert profile.peak_speed == pytest.approx(1.0)
    assert profile.ramp_time == pytest.approx(0.5)
    # 0.25 accelerating, 3.5 cruising, 0.25 decelerating.
    assert profile.duration == pytest.approx(0.5 + 3.5 + 0.5)
    assert profile.travelled(profile.duration) == pytest.approx(4.0)
    # Finite-difference acceleration never exceeds a_max.
    dt = 1e-3
    for k in range(int(profile.duration / dt)):
        t = k * dt
        acc = (profile.speed(t + dt) - profile.speed(t)) / dt
        assert abs(acc) <= 2.0 + 1e-6
        assert profile.speed(t) <= 1.0 + 1e-9


def test_short_move_becomes_a_triangle():
    profile = MotionProfile(0.5, v_max=10.0, a_max=1.0)
    assert profile.peak_speed == pytest.approx(math.sqrt(0.5))
    assert profile.peak_speed < 10.0
    assert profile.duration == pytest.approx(2 * math.sqrt(0.5))
    assert profile.travelled(profile.duration) == pytest.approx(0.5)
    assert profile.travelled(profile.duration / 2) == pytest.approx(0.25)


@pytest.mark.parametrize("a_max", [None, 0.5, 2.0])
def test_time_to_travel_inverts_travelled(a_max):
    profile = MotionProfile(2.0, v_max=1.0, a_max=a_max)
    for s in [0.0, 0.1, 0.5, 1.0, 1.7, 2.0]:
        assert profile.travelled(profile.time_to_travel(s)) == pytest.approx(
            s, abs=1e-9
        )


def test_zero_length_move_takes_no_time():
    profile = MotionProfile(0.0, 1.0, 1.0)
    assert profile.duration == 0.0
    assert profile.travelled(0.3) == 0.0


@pytest.mark.parametrize("bad", [dict(v_max=0), dict(v_max=1, a_max=0), dict(v_max=-1)])
def test_profile_rejects_nonpositive_limits(bad):
    with pytest.raises(ValueError):
        MotionProfile(1.0, **bad)


# --------------------------------------------------------------------------
# stays
# --------------------------------------------------------------------------


def test_waits_collapse_into_one_stay():
    assert stays_of([(0, 0), (0, 0), (0, 1), (0, 1), (0, 1), (0, 2)]) == [
        ((0, 0), 0, 1),
        ((0, 1), 2, 4),
        ((0, 2), 5, 5),
    ]


def test_single_cell_path_is_one_stay():
    assert stays_of([(3, 3)]) == [((3, 3), 0, 0)]


# --------------------------------------------------------------------------
# scheduling real plans
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def warehouse():
    # The scheduler needs a *valid* plan, not an optimal one. CBS on this
    # instance takes ~5 s here and did not finish inside 20 s on a CI runner
    # under coverage; LaCAM is complete and returns in milliseconds.
    scenario = pymapf.build_scenario("warehouse", n_agents=8, seed=3)
    solution = pymapf.solve(scenario.to_problem(), "lacam", time_limit=10.0)
    assert solution is not None and solution.is_valid()
    return scenario, solution


def test_trajectories_start_and_end_where_the_plan_says(warehouse):
    scenario, solution = warehouse
    trajectories = plan_trajectories(solution, limits=KinematicLimits(v_max=1.0))
    for agent in scenario.agents:
        trajectory = trajectories[agent.name]
        assert trajectory.position_at(0.0) == tuple(map(float, agent.start))
        assert trajectory.position_at(1e9) == tuple(map(float, agent.goal))
        # The realised vertex sequence is the discrete path minus repeats.
        assert trajectory.vertices() == [
            v for v, _, _ in stays_of(solution.paths[agent.name])
        ]


def test_unit_speed_never_exceeds_the_discrete_makespan(warehouse):
    _, solution = warehouse
    trajectories = plan_trajectories(solution, limits=KinematicLimits(v_max=1.0))
    # One cell per second with no margin: the discrete plan's own timing is
    # feasible, so the earliest schedule cannot be slower than it -- and it is
    # faster whenever the plan contains waits nothing forces, which a
    # suboptimal planner's plan does. Every agent settles no later than its
    # own discrete path length, and at least one move takes its full second.
    assert trajectories.makespan <= solution.makespan + 1e-9
    for agent, discrete_path in solution.paths.items():
        assert trajectories[agent].arrival_time <= len(discrete_path) - 1 + 1e-9
    assert trajectories.makespan >= 1.0


def test_unit_speed_reproduces_an_optimal_plan_exactly():
    # CBS emits no wait that is not forced, so its timing *is* the earliest
    # schedule and the two makespans coincide. Small enough to be instant.
    scenario = pymapf.build_scenario("corner_swap", size=7, n_agents=4, seed=0)
    solution = pymapf.solve(scenario.to_problem(), "cbs", time_limit=10.0)
    assert solution is not None
    trajectories = plan_trajectories(solution, limits=KinematicLimits(v_max=1.0))
    assert trajectories.makespan == pytest.approx(solution.makespan)


def test_speed_limit_is_never_exceeded(warehouse):
    _, solution = warehouse
    for limits in [KinematicLimits(0.7), KinematicLimits(2.0, a_max=1.0)]:
        trajectories = plan_trajectories(solution, limits=limits)
        assert trajectories.max_speed(dt=0.02) <= limits.v_max + 1e-9


def test_acceleration_limit_holds_along_every_trajectory(warehouse):
    _, solution = warehouse
    limits = KinematicLimits(v_max=1.0, a_max=1.5)
    trajectories = plan_trajectories(solution, limits=limits)
    dt = 1e-3
    for trajectory in trajectories.values():
        previous = trajectory.velocity_at(0.0)
        t = dt
        while t < trajectory.arrival_time:
            current = trajectory.velocity_at(t)
            acc = math.sqrt(sum((c - p) ** 2 for c, p in zip(current, previous))) / dt
            assert acc <= limits.a_max + 1e-6, "%.3f m/s^2 at t=%.3f" % (acc, t)
            previous, t = current, t + dt


def test_slower_agents_take_proportionally_longer(warehouse):
    _, solution = warehouse
    fast = plan_trajectories(solution, limits=KinematicLimits(2.0)).makespan
    slow = plan_trajectories(solution, limits=KinematicLimits(0.5)).makespan
    assert slow == pytest.approx(4 * fast)


def test_larger_safety_margin_never_makes_the_schedule_faster(warehouse):
    _, solution = warehouse
    makespans = [
        plan_trajectories(
            solution, limits=KinematicLimits(1.0, safety_distance=d)
        ).makespan
        for d in (0.0, 0.25, 0.5, 0.9)
    ]
    assert makespans == sorted(makespans)


def test_safety_distance_is_realised_in_space(warehouse):
    """The margin is derived per hand-over from the real geometry, so the
    sampled minimum over a whole plan must clear the distance asked for --
    with and without an acceleration limit, which is where a full-speed rule
    fails. Zero margin is the control: the follower arrives exactly as the
    leader starts to leave.
    """
    _, solution = warehouse
    bare = plan_trajectories(solution, limits=KinematicLimits(1.0)).min_separation(
        0.02
    )[0]
    assert bare == pytest.approx(0.0, abs=0.05)
    for limits in [
        KinematicLimits(1.0, safety_distance=0.5),
        KinematicLimits(1.0, a_max=2.0, safety_distance=0.5),
        KinematicLimits(0.5, a_max=0.5, safety_distance=0.5),
    ]:
        separation = plan_trajectories(solution, limits=limits).min_separation(0.02)[0]
        assert separation >= 0.5 - 0.02, limits


def test_heterogeneous_fleet_uses_each_agents_limits(warehouse):
    scenario, solution = warehouse
    slow_agent = scenario.agents[0].name
    trajectories = plan_trajectories(
        solution,
        limits=KinematicLimits(2.0),
        limits_by_agent={slow_agent: KinematicLimits(0.25)},
    )
    fast_agent = scenario.agents[1].name
    fast_length = trajectories[fast_agent].path_length
    if fast_length > 0:
        assert trajectories[fast_agent].arrival_time >= fast_length / 2.0 - 1e-9
    # The slow agent takes at least its length at its own top speed.
    slow_length = trajectories[slow_agent].path_length
    assert trajectories[slow_agent].arrival_time >= slow_length / 0.25 - 1e-9


# --------------------------------------------------------------------------
# hand-built plans, where the answer is known exactly
# --------------------------------------------------------------------------


def test_follower_waits_exactly_the_margin():
    lead = [(0, 0), (0, 1), (0, 2), (0, 3)]
    tail = [(1, 0), (0, 0), (0, 1), (0, 2)]
    solution = Solution({"lead": lead, "tail": tail})
    assert solution.is_valid()
    trajectories = plan_trajectories(
        solution, limits=KinematicLimits(1.0, safety_time=0.4)
    )
    # lead leaves (0,0) at 0; tail may arrive at (0,0) at 0.4, so it leaves
    # (1,0) at max(0, 0.4 - 1.0) = 0 and arrives at 1.0 -- no wait needed.
    assert trajectories["tail"].stays[1].arrive == pytest.approx(1.0)
    # Straight-line following keeps at least the margin times the speed.
    assert trajectories.min_separation(0.01)[0] >= 0.4 - 1e-6


def test_rotation_on_a_cycle_is_feasible_when_the_margin_fits_a_move():
    graph = ExplicitGraph.undirected(
        [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")],
        positions={"a": (0, 0), "b": (0, 1), "c": (1, 1), "d": (1, 0)},
    )
    rotation = Solution(
        {"1": ["a", "b"], "2": ["b", "c"], "3": ["c", "d"], "4": ["d", "a"]}
    )
    assert rotation.is_valid()
    trajectories = plan_trajectories(
        rotation, graph=graph, limits=KinematicLimits(1.0, safety_time=1.0)
    )
    # Everyone moves at once; the ring keeps its unit spacing.
    assert trajectories.makespan == pytest.approx(1.0)
    assert trajectories.min_separation(0.01)[0] == pytest.approx(
        math.sqrt(2) / 2, abs=0.02
    )


def test_rotation_with_a_margin_longer_than_the_move_is_infeasible():
    graph = ExplicitGraph.undirected(
        [("a", "b"), ("b", "c"), ("c", "a")],
        positions={"a": (0, 0), "b": (0, 1), "c": (1, 0)},
    )
    rotation = Solution({"1": ["a", "b"], "2": ["b", "c"], "3": ["c", "a"]})
    with pytest.raises(InfeasibleScheduleError):
        plan_trajectories(
            rotation, graph=graph, limits=KinematicLimits(1.0, safety_time=1.5)
        )


def test_invalid_plans_are_refused():
    clash = Solution({"a": [(0, 0), (0, 1)], "b": [(0, 1), (0, 0)]})
    assert not clash.is_valid()
    with pytest.raises(ValueError, match="conflict"):
        plan_trajectories(clash)


def test_graph_vertices_need_coordinates():
    solution = Solution({"a": ["x", "y"]})
    with pytest.raises(ValueError, match="positions"):
        plan_trajectories(solution)
    trajectories = plan_trajectories(solution, positions={"x": (0, 0), "y": (3, 4)})
    assert trajectories["a"].path_length == pytest.approx(5.0)


def test_cell_size_scales_distances():
    solution = Solution({"a": [(0, 0), (0, 1), (0, 2)]})
    metre = plan_trajectories(solution, cell_size=1.0, limits=KinematicLimits(1.0))
    half = plan_trajectories(solution, cell_size=0.5, limits=KinematicLimits(1.0))
    assert metre.makespan == pytest.approx(2.0)
    assert half.makespan == pytest.approx(1.0)


def test_position_is_continuous_across_pieces():
    solution = Solution({"a": [(0, 0), (0, 0), (0, 1), (1, 1)]})
    trajectory = plan_trajectories(solution, limits=KinematicLimits(1.0, a_max=1.0))[
        "a"
    ]
    previous = trajectory.position_at(0.0)
    t, dt = 0.0, 0.01
    while t < trajectory.arrival_time + 1.0:
        current = trajectory.position_at(t)
        jump = math.sqrt(sum((c - p) ** 2 for c, p in zip(current, previous)))
        assert jump <= 1.0 * dt + 1e-9  # never faster than v_max, never a jump
        previous, t = current, t + dt


def test_limits_validate_their_inputs():
    with pytest.raises(ValueError):
        KinematicLimits(v_max=0)
    with pytest.raises(ValueError):
        KinematicLimits(a_max=-1)
    with pytest.raises(ValueError):
        KinematicLimits(safety_distance=-0.1)


# --------------------------------------------------------------------------
# the margin itself
# --------------------------------------------------------------------------


def test_required_margin_depends_on_geometry_and_profile():
    from pymapf.kinodynamic import required_margin

    linear = MotionProfile(1.0, v_max=1.0)
    trapezoid = MotionProfile(1.0, v_max=1.0, a_max=2.0)
    vertex = (0.0, 0.0)
    ahead, behind, side = (0.0, 1.0), (0.0, -1.0), (1.0, 0.0)

    straight = required_margin(vertex, ahead, linear, behind, linear, 0.5)
    corner = required_margin(vertex, side, linear, behind, linear, 0.5)
    # Straight following at unit speed needs exactly the distance in seconds;
    # a right-angle hand-over needs more, since the gap opens diagonally.
    assert straight == pytest.approx(0.5, abs=0.01)
    assert corner > straight
    # Leaving from rest opens the gap more slowly than leaving at full speed.
    assert required_margin(vertex, ahead, trapezoid, behind, trapezoid, 0.5) > straight
    # No distance asked for, no delay.
    assert required_margin(vertex, ahead, linear, behind, linear, 0.0) == 0.0


def test_explicit_safety_time_bypasses_the_geometric_margin():
    solution = Solution(
        {"lead": [(0, 0), (0, 1), (0, 2)], "tail": [(1, 0), (0, 0), (0, 1)]}
    )
    fixed = plan_trajectories(solution, limits=KinematicLimits(1.0, safety_time=0.3))
    derived = plan_trajectories(
        solution, limits=KinematicLimits(1.0, safety_distance=0.3)
    )
    # Straight following at unit speed: the two rules agree.
    assert fixed.makespan == pytest.approx(derived.makespan, abs=0.02)


# --------------------------------------------------------------------------
# straight runs: collinear moves flown as one profile
# --------------------------------------------------------------------------


def test_merged_runs_keep_endpoints_and_list_the_vertices_passed(warehouse):
    scenario, solution = warehouse
    limits = KinematicLimits(v_max=1.5, a_max=3.0, safety_distance=0.5)
    merged = plan_trajectories(solution, limits=limits, merge_straight=True)
    halted = plan_trajectories(solution, limits=limits, merge_straight=False)
    for agent in scenario.agents:
        assert merged[agent.name].position_at(0.0) == halted[agent.name].position_at(
            0.0
        )
        assert merged[agent.name].position_at(1e9) == halted[agent.name].position_at(
            1e9
        )
        # Every vertex of the discrete path is a stay or on a segment's via.
        seen = list(merged[agent.name].vertices())
        for segment in merged[agent.name].segments:
            seen.extend(segment.via)
        assert set(seen) == set(solution.paths[agent.name])
    assert sum(len(t.segments) for t in merged.values()) < sum(
        len(t.segments) for t in halted.values()
    )


def test_merged_runs_are_faster_and_just_as_safe(warehouse):
    _, solution = warehouse
    limits = KinematicLimits(v_max=1.5, a_max=3.0, safety_distance=0.5)
    merged = plan_trajectories(solution, limits=limits, merge_straight=True)
    halted = plan_trajectories(solution, limits=limits, merge_straight=False)
    # Not stopping at every cell saves the ramps: strictly faster under a_max.
    assert merged.makespan < halted.makespan
    assert merged.sum_of_arrival_times < halted.sum_of_arrival_times
    assert merged.min_separation(0.02)[0] >= 0.5 - 5e-3
    assert merged.max_speed(0.02) <= 1.5 + 1e-9


def test_a_wait_or_a_turn_breaks_a_run():
    # Straight, then a wait, then straight again: two runs. A turn: two runs.
    waits = Solution({"a": [(0, 0), (0, 1), (0, 1), (0, 2), (0, 3)]})
    turns = Solution({"b": [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]})
    straight = Solution({"c": [(0, 0), (0, 1), (0, 2), (0, 3)]})
    for solution, expected in ((waits, 2), (turns, 2), (straight, 1)):
        trajectories = plan_trajectories(solution, merge_straight=True)
        (trajectory,) = trajectories.values()
        assert len(trajectory.segments) == expected
    (only,) = plan_trajectories(straight, merge_straight=True).values()
    assert only.segments[0].via == ((0, 1), (0, 2))
    # One run of length 3 at unit speed: passes (0,1) at 1 s, (0,2) at 2 s.
    assert only.position_at(1.0) == pytest.approx((0.0, 1.0))
    assert only.position_at(2.5) == pytest.approx((0.0, 2.5))


def test_following_along_a_run_keeps_the_margin_at_every_passed_vertex():
    lead = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
    tail = [(1, 0), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    solution = Solution({"lead": lead, "tail": tail})
    limits = KinematicLimits(v_max=1.0, safety_distance=0.4)
    trajectories = plan_trajectories(solution, limits=limits, merge_straight=True)
    assert len(trajectories["lead"].segments) == 1
    assert trajectories.min_separation(0.01)[0] >= 0.4 - 1e-6
    # The tail follows one margin behind, not one full stop-and-go behind.
    assert trajectories["tail"].arrival_time <= 6.0 + 1e-9


def test_crossing_runs_hand_over_the_shared_vertex_safely():
    # Two runs at right angles through (2, 2): first at t=2, second at t=4.
    across = [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)]
    down = [(0, 2), (0, 2), (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]
    solution = Solution({"across": across, "down": down})
    assert solution.is_valid()
    limits = KinematicLimits(v_max=1.0, a_max=2.0, safety_distance=0.5)
    trajectories = plan_trajectories(solution, limits=limits, merge_straight=True)
    assert len(trajectories["across"].segments) == 1
    assert len(trajectories["down"].segments) == 1
    assert trajectories.min_separation(0.01)[0] >= 0.5 - 5e-3
    # 'down' passes (2, 2) at speed after 'across' has cleared it.
    across_at = trajectories["across"].stays[0].depart + trajectories[
        "across"
    ].segments[0].profile.time_to_travel(2.0)
    down_at = trajectories["down"].stays[0].depart + trajectories["down"].segments[
        0
    ].profile.time_to_travel(2.0)
    assert down_at > across_at


def test_merging_runs_never_makes_a_valid_plan_infeasible():
    """A CBS plan (random_obstacles 7x7, seed 15) that schedules without
    merging. Merged, a run is ordered against others at vertices it passes
    long after it departs, and the constraints closed a loop: the plan was
    reported infeasible, as if four agents had to rotate at once."""
    solution = Solution(
        {
            "A": [(2, 3), (2, 4), (2, 5), (1, 5)],
            "B": [(3, 3), (3, 3), (4, 3), (4, 4), (4, 5)],
            "C": [(5, 5), (4, 5), (3, 5), (3, 4)],
            "D": [(2, 5), (3, 5), (3, 4), (2, 4), (2, 3), (1, 3)],
            "E": [(2, 4), (3, 4), (3, 3), (4, 3)],
            "F": [(4, 2), (4, 3), (4, 4), (4, 5), (3, 5)],
        }
    )
    assert solution.is_valid()
    limits = KinematicLimits(v_max=1.3, a_max=0.7, safety_distance=0.9)
    halted = plan_trajectories(solution, limits=limits)
    merged = plan_trajectories(solution, limits=limits, merge_straight=True)
    assert merged.makespan == pytest.approx(halted.makespan)  # scheduled unmerged
    assert merged.min_separation(0.01)[0] >= 0.9 - 5e-3


def test_diagonal_moves_that_cross_between_vertices_keep_their_distance():
    """Diagonal moves can cross mid-cell without sharing a vertex, so the
    shared-vertex constraints never saw them: CBS returns this swap as valid,
    and the schedule flew both agents through (0.5, 0.5) at once, 0.004
    apart for a 0.5 requirement."""
    grid = pymapf.GridMap([[0, 0], [0, 0]])
    agents = [pymapf.Agent("a", (0, 0), (1, 1)), pymapf.Agent("b", (0, 1), (1, 0))]
    problem = pymapf.MAPFProblem(grid, agents, allow_diagonals=True)
    solution = pymapf.solve(problem, "cbs")
    assert solution.paths == {"a": [(0, 0), (1, 1)], "b": [(0, 1), (1, 0)]}
    limits = KinematicLimits(v_max=1.0, a_max=1.0, safety_distance=0.5)
    assert plan_trajectories(solution, limits=limits).min_separation(0.01)[0] >= (
        0.5 - 5e-3
    )


def test_a_simultaneous_crossing_takes_the_order_the_plan_implies():
    """C and D cross diagonally while A trades places with both. Which of C
    and D goes first is free, and the wrong choice closes a cycle with A's
    hand-overs (from an 8-connected CBS plan, random_obstacles 7x7 seed 4)."""
    solution = Solution(
        {
            "A": [(5, 3), (5, 4), (4, 5)],
            "B": [(2, 4), (2, 3), (2, 2), (2, 1), (1, 1)],
            "C": [(4, 4), (5, 3), (4, 2), (3, 2)],
            "D": [(5, 4), (4, 3), (5, 2), (4, 2)],
            "E": [(2, 5), (2, 4), (2, 3), (1, 3)],
            "F": [(3, 5), (2, 5), (2, 4), (2, 3)],
        }
    )
    assert solution.is_valid()
    for safety in (0.5, 0.7):
        limits = KinematicLimits(v_max=1.3, safety_distance=safety)
        trajectories = plan_trajectories(solution, limits=limits)
        assert trajectories.min_separation(0.01)[0] >= safety - 5e-3


def test_runs_that_share_an_endpoint_are_allowed():
    # Both leave (0, 0) along the same row, one after the other: sharing the
    # origin must not be mistaken for a permanent collision there.
    first = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    second = [(1, 0), (1, 0), (0, 0), (0, 1), (0, 2), (0, 3)]
    solution = Solution({"first": first, "second": second})
    assert solution.is_valid()
    limits = KinematicLimits(v_max=1.0, a_max=2.0, safety_distance=0.5)
    trajectories = plan_trajectories(solution, limits=limits, merge_straight=True)
    assert trajectories.min_separation(0.01)[0] >= 0.5 - 5e-3
    assert trajectories.makespan < 12.0
