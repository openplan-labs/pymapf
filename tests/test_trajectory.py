"""Joint trajectory optimisation in continuous space."""

import math
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pytest  # noqa: E402

np = pytest.importorskip("numpy")

from pymapf.trajectory import (  # noqa: E402
    PiecewisePolynomial,
    Vehicle,
    cost_matrix,
    derivative_row,
    plan_joint_trajectories,
    solve_qp,
    waypoints_from_solution,
)


def _scipy():
    return pytest.importorskip("scipy.optimize")


# --------------------------------------------------------------------------
# the polynomial layer
# --------------------------------------------------------------------------


@pytest.mark.parametrize("duration", [0.5, 1.0, 3.7])
@pytest.mark.parametrize("d", [0, 1, 2, 3])
def test_derivative_rows_match_finite_differences(duration, d):
    order = 7
    rng = np.random.default_rng(0)
    coefficients = rng.normal(size=order + 1)

    def value(t, derivative):
        return float(
            derivative_row(order, duration, t / duration, derivative) @ coefficients
        )

    t, h = 0.37 * duration, 1e-6 * duration
    analytic = value(t, d + 1)
    numeric = (value(t + h, d) - value(t - h, d)) / (2 * h)
    assert analytic == pytest.approx(numeric, rel=1e-4, abs=1e-6)


@pytest.mark.parametrize("d", [2, 3, 4])
def test_cost_matrix_is_the_exact_integral(d):
    order, duration = 7, 2.3
    rng = np.random.default_rng(1)
    c = rng.normal(size=order + 1)
    exact = float(c @ cost_matrix(order, duration, d) @ c)
    times = np.linspace(0.0, duration, 20001)
    values = np.array(
        [derivative_row(order, duration, t / duration, d) @ c for t in times]
    )
    numeric = float(np.trapezoid(values**2, times))
    assert exact == pytest.approx(numeric, rel=1e-6)


def test_piecewise_polynomial_evaluates_clamps_and_reports_shape():
    coefficients = np.zeros((2, 2, 4))
    coefficients[0, 0] = [0.0, 1.0, 0.0, 0.0]  # x = s on the first segment
    coefficients[1, 0] = [1.0, 1.0, 0.0, 0.0]  # x = 1 + s on the second
    trajectory = PiecewisePolynomial((1.0, 1.0), coefficients)
    assert (trajectory.segments, trajectory.dimension, trajectory.order) == (2, 2, 3)
    assert trajectory.duration == 2.0
    assert trajectory(0.0)[0] == pytest.approx(0.0)
    assert trajectory(0.5)[0] == pytest.approx(0.5)
    assert trajectory(1.5)[0] == pytest.approx(1.5)
    assert trajectory(2.0)[0] == pytest.approx(2.0)
    # outside: parked at the endpoints, and not moving
    assert trajectory(-1.0)[0] == pytest.approx(0.0)
    assert trajectory(99.0)[0] == pytest.approx(2.0)
    assert np.allclose(trajectory(99.0, 1), 0.0)
    assert np.allclose(trajectory.knots, [0.0, 1.0, 2.0])


def test_dilation_scales_the_derivatives_and_keeps_the_shape():
    rng = np.random.default_rng(2)
    trajectory = PiecewisePolynomial((1.5, 2.5), rng.normal(size=(2, 3, 6)))
    slower = trajectory.dilated(2.0)
    assert slower.duration == pytest.approx(2 * trajectory.duration)
    for fraction in (0.0, 0.3, 0.61, 1.0):
        t = fraction * trajectory.duration
        assert np.allclose(slower(2 * t), trajectory(t))
        assert np.allclose(slower(2 * t, 1), trajectory(t, 1) / 2.0, atol=1e-9)
        assert np.allclose(slower(2 * t, 2), trajectory(t, 2) / 4.0, atol=1e-9)
    assert slower.path_length() == pytest.approx(trajectory.path_length(), rel=1e-6)
    with pytest.raises(ValueError):
        trajectory.dilated(0.0)


def test_a_malformed_trajectory_is_refused():
    with pytest.raises(ValueError, match="segments, dimension"):
        PiecewisePolynomial((1.0,), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="durations"):
        PiecewisePolynomial((1.0,), np.zeros((2, 2, 4)))
    with pytest.raises(ValueError, match="positive time"):
        PiecewisePolynomial((0.0,), np.zeros((1, 2, 4)))


# --------------------------------------------------------------------------
# the quadratic program
# --------------------------------------------------------------------------


def test_the_unconstrained_and_equality_cases_are_one_solve():
    hessian = np.array([[2.0, 0.0], [0.0, 4.0]])
    gradient = np.array([-2.0, -8.0])
    x, converged = solve_qp(hessian, gradient)[:2]
    assert converged and np.allclose(x, [1.0, 2.0])
    a, b = np.array([[1.0, 1.0]]), np.array([1.0])
    x, converged = solve_qp(hessian, gradient, a, b)[:2]
    assert converged and a @ x == pytest.approx(b)


def test_inconsistent_equalities_are_reported_rather_than_hidden():
    a = np.array([[1.0, 0.0], [1.0, 0.0]])
    b = np.array([0.0, 1.0])
    result = solve_qp(np.eye(2), np.zeros(2), a, b)
    assert not result.converged
    assert result.violation > 0.1


def test_a_semidefinite_cost_still_has_a_solution():
    """The snap cost ignores each segment's constant and linear terms, so the
    Hessian the trajectory layer hands over is only positive *semi*-definite;
    the solver must not need a strictly convex one."""
    hessian = np.diag([1.0, 1.0, 0.0, 0.0])
    inequality = np.vstack([np.eye(4), -np.eye(4)])
    rhs = np.full(8, 2.0)
    a, b = np.array([[1.0, 1.0, 1.0, 1.0]]), np.array([3.0])
    x, converged = solve_qp(hessian, np.zeros(4), a, b, inequality, rhs)[:2]
    assert converged
    assert a @ x == pytest.approx(b)
    assert (inequality @ x <= rhs + 1e-9).all()


@pytest.mark.parametrize("seed", range(12))
def test_matches_slsqp_on_random_dense_problems(seed):
    minimize = _scipy().minimize
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 14))
    n_eq = int(rng.integers(0, 4))
    n_ineq = int(rng.integers(1, 10))
    m = rng.normal(size=(n, n))
    hessian = m @ m.T + 0.05 * np.eye(n)
    gradient = rng.normal(size=n)
    interior = rng.normal(size=n)  # a point every constraint admits
    a = rng.normal(size=(n_eq, n)) if n_eq else None
    b = a @ interior if n_eq else None
    c = rng.normal(size=(n_ineq, n))
    d = c @ interior + rng.random(n_ineq)

    x, converged = solve_qp(hessian, gradient, a, b, c, d)[:2]
    assert converged
    assert (c @ x <= d + 1e-6).all()
    if n_eq:
        assert np.abs(a @ x - b).max() < 1e-6

    constraints = [{"type": "ineq", "fun": lambda z: d - c @ z}]
    if n_eq:
        constraints.append({"type": "eq", "fun": lambda z: a @ z - b})
    reference = minimize(
        lambda z: 0.5 * z @ hessian @ z + gradient @ z,
        interior,
        jac=lambda z: hessian @ z + gradient,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 800, "ftol": 1e-12},
    )
    mine = 0.5 * x @ hessian @ x + gradient @ x
    theirs = 0.5 * reference.x @ hessian @ reference.x + gradient @ reference.x
    assert mine <= theirs + 1e-4 * max(1.0, abs(theirs))


# --------------------------------------------------------------------------
# joint optimisation: what the trajectories satisfy
# --------------------------------------------------------------------------


def crossing(n=2, radius=0.35, span=10.0):
    """Vehicles on a line, each swapping with the one opposite."""
    return [
        Vehicle(
            "v%d" % k,
            (span * k / max(1, n - 1), 0.0),
            (span * (n - 1 - k) / max(1, n - 1), span),
            radius=radius,
        )
        for k in range(n)
    ]


def ring(n, radius=8.0, dimension=2, body=0.35):
    fleet = []
    for k in range(n):
        angle = 2 * math.pi * k / n
        start = [radius * math.cos(angle), radius * math.sin(angle)]
        goal = [-radius * math.cos(angle), -radius * math.sin(angle)]
        if dimension == 3:
            start.append(0.0)
            goal.append(0.0)
        fleet.append(Vehicle("v%d" % k, start, goal, radius=body))
    return fleet


@pytest.fixture(scope="module")
def swap():
    fleet = [
        Vehicle("a", (0.0, 0.0), (10.0, 10.0), radius=0.35),
        Vehicle("b", (10.0, 0.0), (0.0, 10.0), radius=0.35),
    ]
    return fleet, plan_joint_trajectories(
        fleet, v_max=2.0, a_max=4.0, segments=4, samples=50, iterations=20
    )


def test_endpoints_are_exact_and_the_fleet_starts_and_ends_at_rest(swap):
    fleet, plan = swap
    for vehicle in fleet:
        trajectory = plan.trajectories[vehicle.name]
        assert trajectory(0.0) == pytest.approx(np.asarray(vehicle.start), abs=1e-6)
        assert trajectory(plan.duration) == pytest.approx(
            np.asarray(vehicle.goal), abs=1e-6
        )
        for derivative in (1, 2, 3):
            assert np.linalg.norm(trajectory(0.0, derivative)) < 1e-5
            assert np.linalg.norm(trajectory(plan.duration, derivative)) < 1e-5


def test_the_trajectories_are_continuous_across_their_knots(swap):
    _, plan = swap
    for trajectory in plan.trajectories.values():
        for knot in trajectory.knots[1:-1]:
            for derivative in range(5):
                before = trajectory(knot - 1e-7, derivative)
                after = trajectory(knot + 1e-7, derivative)
                assert np.allclose(before, after, atol=1e-4)


def test_the_pair_never_comes_closer_than_their_bodies(swap):
    fleet, plan = swap
    distance, when, pair = plan.min_separation(0.005)
    assert distance >= plan.required_separation(*pair) - 1e-9
    assert 0.0 <= when <= plan.duration
    assert plan.is_safe()
    assert plan.separation_margin() <= 0.0


def test_speed_and_acceleration_limits_hold(swap):
    _, plan = swap
    assert plan.max_speed(0.005) <= 2.0 + 1e-6
    assert plan.max_acceleration(0.005) <= 4.0 + 1e-6


def test_the_optimiser_actually_bends_the_straight_line(swap):
    """Both vehicles would collide head-on flying straight; the cost of the
    detour is the point of the exercise, so it must be small but non-zero."""
    fleet, plan = swap
    for vehicle in fleet:
        straight = np.linalg.norm(np.asarray(vehicle.goal) - np.asarray(vehicle.start))
        flown = plan.path_length(vehicle.name)
        assert flown > straight + 1e-3
        assert flown < 1.35 * straight


@pytest.mark.parametrize("n", [3, 5])
def test_a_ring_swap_stays_apart(n):
    fleet = ring(n)
    plan = plan_joint_trajectories(
        fleet, v_max=2.0, a_max=4.0, segments=4, samples=50, iterations=20
    )
    assert plan.converged
    assert plan.is_safe()
    assert plan.max_speed(0.01) <= 2.0 + 1e-6


def test_three_dimensions_work_the_same_way():
    fleet = ring(4, dimension=3)
    plan = plan_joint_trajectories(
        fleet, v_max=2.0, a_max=4.0, segments=4, samples=50, iterations=20
    )
    assert plan.trajectories["v0"].dimension == 3
    assert plan.is_safe()
    # The fleet is free to use the third axis and, unlike on a plane, should:
    # lifting is cheaper than the detour a planar swap needs.
    heights = [
        abs(plan.trajectories[v.name].sample(plan.times(0.05))[:, 2]).max()
        for v in fleet
    ]
    assert max(heights) > 0.1


def test_obstacles_are_avoided_by_the_required_margin():
    fleet = [
        Vehicle("a", (-6.0, 0.0), (6.0, 0.0), radius=0.3),
        Vehicle("b", (6.0, 0.5), (-6.0, 0.5), radius=0.3),
    ]
    obstacles = [((0.0, 0.0), 1.5)]
    plan = plan_joint_trajectories(
        fleet,
        obstacles=obstacles,
        v_max=2.0,
        a_max=4.0,
        segments=5,
        samples=60,
        iterations=25,
    )
    assert plan.obstacle_margin(0.005) <= 0.0
    assert plan.is_safe()
    for name in ("a", "b"):
        points = plan.trajectories[name].sample(plan.times(0.01))
        assert np.linalg.norm(points, axis=1).min() >= 1.5 + 0.3 - 1e-6


def test_unequal_vehicles_keep_unequal_distances():
    fleet = [
        Vehicle("small", (0.0, 0.0), (8.0, 0.0), radius=0.2),
        Vehicle("large", (8.0, 0.0), (0.0, 0.0), radius=0.9),
    ]
    plan = plan_joint_trajectories(
        fleet, v_max=2.0, a_max=4.0, segments=4, samples=50, iterations=20
    )
    assert plan.required_separation("small", "large") == pytest.approx(1.1)
    assert plan.min_separation(0.005)[0] >= 1.1 - 1e-9


def test_a_slower_vehicle_keeps_its_own_limit():
    fleet = [
        Vehicle("fast", (0.0, 0.0), (10.0, 0.0), radius=0.3),
        Vehicle("slow", (0.0, 3.0), (10.0, 3.0), radius=0.3, v_max=0.6),
    ]
    plan = plan_joint_trajectories(
        fleet,
        duration=20.0,
        v_max=2.0,
        a_max=4.0,
        segments=3,
        samples=50,
        iterations=20,
    )
    times = plan.times(0.01)
    speeds = {
        name: float(np.linalg.norm(trajectory.sample(times, 1), axis=1).max())
        for name, trajectory in plan.trajectories.items()
    }
    assert speeds["slow"] <= 0.6 + 1e-6
    assert speeds["fast"] > 0.6


# --------------------------------------------------------------------------
# the two design decisions that are worth a test each
# --------------------------------------------------------------------------


def test_without_the_inter_sample_bound_a_pair_dips_below_the_requirement():
    """Enforcing separation at sample times only is not enough, and this is
    the measurement that says so: the same instance is feasible at every
    sample and violated between two of them."""
    fleet = [
        Vehicle("a", (0.0, 0.0), (6.0, 0.0), radius=0.4),
        Vehicle("b", (6.0, 0.0), (0.0, 0.0), radius=0.4),
    ]
    common = dict(
        duration=6.0,
        segments=2,
        order=5,
        samples=40,
        v_max=3.0,
        a_max=6.0,
        boundary=2,
        continuity=3,
        iterations=40,
        enforce_limits=False,
    )
    sampled = plan_joint_trajectories(fleet, inter_sample=False, **common)
    bounded = plan_joint_trajectories(fleet, inter_sample=True, **common)

    coarse = np.linspace(0.0, 6.0, 40)
    at_samples = np.linalg.norm(
        sampled.trajectories["a"].sample(coarse)
        - sampled.trajectories["b"].sample(coarse),
        axis=1,
    ).min()
    assert at_samples >= 0.8 - 1e-6  # feasible where it was asked to be
    assert sampled.min_separation(0.002)[0] < 0.8 - 1e-3  # and not in between

    assert bounded.min_separation(0.002)[0] >= 0.8 - 1e-9
    assert bounded.cost > sampled.cost  # the guarantee is not free


def test_time_dilation_meets_the_limits_and_preserves_every_separation():
    fleet = ring(4)
    fast = plan_joint_trajectories(
        fleet,
        duration=6.0,
        v_max=1.0,
        a_max=1.0,
        segments=4,
        samples=50,
        iterations=20,
        enforce_limits=False,
    )
    slow = plan_joint_trajectories(
        fleet,
        duration=6.0,
        v_max=1.0,
        a_max=1.0,
        segments=4,
        samples=50,
        iterations=20,
        enforce_limits=True,
    )
    assert slow.time_scale > 1.0
    assert slow.duration == pytest.approx(fast.duration * slow.time_scale)
    assert slow.max_speed(0.01) <= 1.0 + 1e-6
    assert slow.max_acceleration(0.01) <= 1.0 + 1e-6
    # The geometry is untouched: same closest approach, at the scaled time.
    fast_distance, fast_time, pair = fast.min_separation(0.005)
    slow_distance, slow_time, _ = slow.min_separation(0.005 * slow.time_scale)
    assert slow_distance == pytest.approx(fast_distance, rel=1e-3)
    assert slow_time == pytest.approx(fast_time * slow.time_scale, rel=1e-2)
    assert pair


def test_the_reported_cost_is_the_cost_of_the_returned_trajectories():
    """It used to be the optimiser's cost, before the dilation stretched the
    flight -- 16x too high here, where the limits need a 1.5x slower clock
    and snap cost scales as time to the -7th."""
    plan = plan_joint_trajectories(
        [Vehicle("a", (0.0, 0.0), (10.0, 0.0))], duration=5.0, samples=30
    )
    assert plan.time_scale > 1.1
    trajectory = plan.trajectories["a"]
    snap = sum(
        float(c @ cost_matrix(trajectory.order, duration, 4) @ c)
        for duration, segment in zip(trajectory.durations, trajectory.coefficients)
        for c in segment
    )
    assert plan.cost == pytest.approx(0.5 * snap, rel=1e-9)


def test_the_speed_limit_is_not_bought_with_a_sideways_weave():
    """A lone vehicle has no reason to leave a straight line. With the speed
    limit linearised as a tangent plane it did: turning a little let it fly
    faster than the limit along its old heading, so it weaved 0.6 m from side
    to side, 12% over the limit until the clock was stretched to hide it."""
    fleet = [Vehicle("a", (0.0, 0.0), (10.0, 0.0))]
    raw = plan_joint_trajectories(fleet, v_max=2.0, a_max=4.0, enforce_limits=False)
    assert raw.max_speed(0.005) <= 2.0 * 1.005  # only between the samples
    plan = plan_joint_trajectories(fleet, v_max=2.0, a_max=4.0)
    sideways = np.abs(plan.trajectories["a"].sample(plan.times(0.01))[:, 1]).max()
    assert sideways < 0.02
    assert plan.path_length("a") == pytest.approx(10.0, abs=1e-3)
    assert plan.time_scale < 1.005


# --------------------------------------------------------------------------
# seeding from a discrete plan
# --------------------------------------------------------------------------


def test_waypoints_from_a_solution_keep_the_corners_and_drop_the_rest():
    pymapf = pytest.importorskip("pymapf")
    solution = pymapf.Solution(
        {
            "a": [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3)],
            "b": [(5, 5), (5, 5), (5, 6)],
        }
    )
    waypoints = waypoints_from_solution(solution)
    assert np.allclose(waypoints["a"], [(0, 0), (0, 3), (2, 3)])  # one corner
    assert np.allclose(waypoints["b"], [(5, 5), (5, 6)])  # the wait is dropped
    dense = waypoints_from_solution(solution, simplify=False)
    assert len(dense["a"]) == 6
    scaled = waypoints_from_solution(solution, cell_size=0.5)
    assert np.allclose(scaled["a"], [(0, 0), (0, 1.5), (1, 1.5)])


def test_a_wait_on_a_corner_keeps_the_corner_and_a_parked_agent_has_a_leg():
    pymapf = pytest.importorskip("pymapf")
    solution = pymapf.Solution(
        {
            "a": [(0, 0), (0, 1), (0, 2), (0, 3), (0, 3), (1, 3), (2, 3)],
            "b": [(4, 4)],
        }
    )
    waypoints = waypoints_from_solution(solution)
    # The wait used to hide the corner from the collinearity test, and the
    # seed cut straight across whatever the discrete plan went around.
    assert np.allclose(waypoints["a"], [(0, 0), (0, 3), (2, 3)])
    # One vertex used to be one waypoint, which the planner refuses.
    assert np.allclose(waypoints["b"], [(4, 4), (4, 4)])
    plan = plan_joint_trajectories(
        [Vehicle("a", (0, 0), (2, 3)), Vehicle("b", (4, 4), (4, 4))],
        waypoints=waypoints,
        samples=30,
        iterations=5,
    )
    assert plan.trajectories["b"](0.5 * plan.duration) == pytest.approx([4.0, 4.0])


def test_a_discrete_plan_seeds_the_optimiser_through_an_obstacle_field():
    """The fleet stays apart and keeps its endpoints. The grid's walls are
    *not* checked: they are not constraints of the continuous problem (only
    ``obstacles=`` spheres are), and on this instance three of the four
    smoothed trajectories cut across a blocked cell -- see the module
    docstring."""
    pymapf = pytest.importorskip("pymapf")
    scenario = pymapf.build_scenario(
        "random_obstacles", height=10, width=10, n_agents=4, seed=1
    )
    solution = pymapf.solve(scenario.to_problem(), "lacam", time_limit=10.0)
    assert solution is not None
    fleet = [
        Vehicle(
            a.name,
            np.asarray(a.start, dtype=float),
            np.asarray(a.goal, dtype=float),
            radius=0.3,
        )
        for a in scenario.agents
    ]
    plan = plan_joint_trajectories(
        fleet,
        waypoints=waypoints_from_solution(solution),
        v_max=2.0,
        a_max=4.0,
        samples=60,
        iterations=25,
    )
    assert plan.is_safe()
    for agent in scenario.agents:
        trajectory = plan.trajectories[agent.name]
        assert trajectory(0.0) == pytest.approx(
            np.asarray(agent.start, dtype=float), abs=1e-6
        )
        assert trajectory(plan.duration) == pytest.approx(
            np.asarray(agent.goal, dtype=float), abs=1e-6
        )


def test_a_straight_seed_still_leaves_the_optimiser_room_to_avoid():
    """CBS makes ``a`` wait so that ``b`` crosses first. Both paths are
    straight, so each seed simplified to a single leg -- one segment of
    order 7, every coefficient fixed by the boundary conditions -- and the
    pair finished 0.024 m apart for a 0.6 m requirement, with no error."""
    pymapf = pytest.importorskip("pymapf")
    grid = pymapf.GridMap([[0] * 5 for _ in range(5)])
    agents = [pymapf.Agent("a", (2, 0), (2, 4)), pymapf.Agent("b", (0, 2), (4, 2))]
    solution = pymapf.solve(pymapf.MAPFProblem(grid, agents), "cbs")
    fleet = [Vehicle(a.name, a.start, a.goal, radius=0.3) for a in agents]
    plan = plan_joint_trajectories(
        fleet, waypoints=waypoints_from_solution(solution), v_max=2.0, a_max=4.0
    )
    assert all(len(t.durations) >= 4 for t in plan.trajectories.values())
    assert plan.converged
    assert plan.min_separation(0.005)[0] >= 0.6 - 1e-9
    # Asked for outright, a trajectory with no freedom is refused, not flown.
    with pytest.raises(ValueError, match="no freedom"):
        plan_joint_trajectories(fleet, segments=1)


# --------------------------------------------------------------------------
# optimality, and the errors
# --------------------------------------------------------------------------


def test_the_joint_optimum_matches_slsqp_on_a_small_instance():
    """The whole layer, end to end, against a general-purpose optimiser on a
    problem small enough for one: same cost, same trajectories."""
    minimize = _scipy().minimize
    order, segments, dimension, duration = 5, 2, 2, 6.0
    boundary, continuity, minimise, samples = 2, 3, 4, 40
    fleet = [
        Vehicle("a", (0.0, 0.0), (6.0, 0.0), radius=0.4),
        Vehicle("b", (6.0, 0.0), (0.0, 0.0), radius=0.4),
    ]
    mine = plan_joint_trajectories(
        fleet,
        duration=duration,
        segments=segments,
        order=order,
        samples=samples,
        v_max=3.0,
        a_max=6.0,
        iterations=60,
        boundary=boundary,
        continuity=continuity,
        minimise=minimise,
        enforce_limits=False,
        inter_sample=False,
        tolerance=1e-7,
        cost_tolerance=1e-6,
    )
    durations = [duration / segments] * segments

    def unpack(z):
        return z.reshape(2, segments, dimension, order + 1)

    def value(block, k, s, d):
        return block[k] @ derivative_row(order, durations[k], s, d)

    def cost(z):
        blocks = unpack(z)
        total = 0.0
        for v in range(2):
            for k in range(segments):
                q = cost_matrix(order, durations[k], minimise)
                for m in range(dimension):
                    total += blocks[v, k, m] @ q @ blocks[v, k, m]
        return 0.5 * total

    constraints = []
    for v, vehicle in enumerate(fleet):
        for m in range(dimension):
            for d in range(boundary + 1):
                constraints.append(
                    {
                        "type": "eq",
                        "fun": (
                            lambda z, v=v, m=m, d=d, vehicle=vehicle: value(
                                unpack(z)[v], 0, 0.0, d
                            )[m]
                            - (vehicle.start[m] if d == 0 else 0.0)
                        ),
                    }
                )
                constraints.append(
                    {
                        "type": "eq",
                        "fun": (
                            lambda z, v=v, m=m, d=d, vehicle=vehicle: value(
                                unpack(z)[v], segments - 1, 1.0, d
                            )[m]
                            - (vehicle.goal[m] if d == 0 else 0.0)
                        ),
                    }
                )
            for k in range(segments - 1):
                for d in range(continuity + 1):
                    constraints.append(
                        {
                            "type": "eq",
                            "fun": (
                                lambda z, v=v, m=m, d=d, k=k: value(
                                    unpack(z)[v], k, 1.0, d
                                )[m]
                                - value(unpack(z)[v], k + 1, 0.0, d)[m]
                            ),
                        }
                    )

    times = np.linspace(0.0, duration, samples)

    def separation(z):
        blocks = unpack(z)
        out = []
        for t in times:
            k = min(int(t / durations[0]), segments - 1)
            s = (t - durations[0] * k) / durations[0]
            out.append(
                np.linalg.norm(value(blocks[0], k, s, 0) - value(blocks[1], k, s, 0))
                - 0.8
            )
        return np.array(out)

    constraints.append({"type": "ineq", "fun": separation})
    guess = np.concatenate(
        [mine.trajectories[n].coefficients.ravel() for n in ("a", "b")]
    )
    reference = minimize(
        cost,
        guess,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 600, "ftol": 1e-12},
    )
    assert reference.status == 0
    assert mine.cost == pytest.approx(reference.fun, rel=1e-5)


def test_a_deterministic_instance_gives_a_deterministic_plan():
    fleet = crossing(3)
    first = plan_joint_trajectories(
        fleet, v_max=2.0, a_max=4.0, segments=3, samples=40, iterations=12
    )
    second = plan_joint_trajectories(
        fleet, v_max=2.0, a_max=4.0, segments=3, samples=40, iterations=12
    )
    for name in first.trajectories:
        assert np.allclose(
            first.trajectories[name].coefficients,
            second.trajectories[name].coefficients,
        )
    assert first.cost == pytest.approx(second.cost)


def test_ill_posed_fleets_are_refused():
    with pytest.raises(ValueError, match="no vehicles"):
        plan_joint_trajectories([])
    with pytest.raises(ValueError, match="unique"):
        plan_joint_trajectories(
            [
                Vehicle("a", (0.0, 0.0), (1.0, 1.0)),
                Vehicle("a", (1.0, 1.0), (0.0, 0.0)),
            ]
        )
    with pytest.raises(ValueError, match="same dimension"):
        plan_joint_trajectories(
            [
                Vehicle("a", (0.0, 0.0), (1.0, 1.0)),
                Vehicle("b", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ]
        )
    with pytest.raises(ValueError, match="different dimensions"):
        Vehicle("a", (0.0, 0.0), (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="too few for"):
        plan_joint_trajectories(
            [Vehicle("a", (0.0, 0.0), (1.0, 1.0))], order=3, boundary=3
        )
    with pytest.raises(ValueError, match="infeasible endpoint"):
        plan_joint_trajectories(
            [
                Vehicle("a", (0.0, 0.0), (5.0, 0.0), radius=1.0),
                Vehicle("b", (0.5, 0.0), (5.0, 5.0), radius=1.0),
            ]
        )
    with pytest.raises(ValueError, match="obstacle centre"):
        plan_joint_trajectories(
            [Vehicle("a", (0.0, 0.0), (1.0, 1.0))], obstacles=[((0.0, 0.0, 0.0), 1.0)]
        )


def test_a_parked_fleet_and_a_tiny_hop_are_planned_not_refused():
    # Every vehicle already at its goal: the makespan used to floor at 1 ms,
    # and the equalities of a 1 ms segment were reported as "inconsistent".
    parked = plan_joint_trajectories(
        [Vehicle("a", (1.0, 1.0), (1.0, 1.0)), Vehicle("b", (3.0, 0.0), (3.0, 0.0))]
    )
    assert parked.converged
    assert parked.trajectories["a"](0.5 * parked.duration) == pytest.approx([1, 1])
    # A 1 cm hop sizes a 7 ms flight; it is planned, and then slowed down
    # until it meets the limits, which a fixed 10 ms check step never saw.
    hop = plan_joint_trajectories([Vehicle("a", (0.0, 0.0), (0.01, 0.0))])
    assert hop.trajectories["a"](hop.duration) == pytest.approx([0.01, 0.0])
    step = hop.duration / 5000
    assert hop.max_speed(step) <= 2.0 + 1e-6
    assert hop.max_acceleration(step) <= 4.0 + 1e-6
    # An infinite speed limit cannot size the flight; say so.
    with pytest.raises(ValueError, match="pass a duration"):
        plan_joint_trajectories([Vehicle("a", (0.0, 0.0), (4.0, 0.0))], v_max=math.inf)


def test_the_summary_reports_what_a_caller_needs():
    fleet = crossing(2)
    plan = plan_joint_trajectories(
        fleet, v_max=2.0, a_max=4.0, segments=3, samples=40, iterations=12
    )
    summary = plan.summary()
    assert summary["vehicles"] == 2
    assert summary["duration"] > 0
    assert summary["min_separation"] >= 0.0
    assert summary["max_speed"] <= 2.0 + 1e-6
    assert summary["path_length"] > 0
    assert "JointPlan" in repr(plan)
    assert set(plan.positions(0.5)) == {"v0", "v1"}
