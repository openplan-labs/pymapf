"""Formation control: shapes, assignment, rigidity and the four control laws.

The theme running through these tests is that a formation controller must be
graded under the symmetry group its *sensing* leaves free. A distance-based
controller that lands on a mirror image has succeeded; a bearing-based one that
lands on a half-size copy has succeeded. Testing them against a fixed pose
measures the test's assumptions, not the controller.
"""

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pytest

np = pytest.importorskip("numpy")

from pymapf.swarm import (
    BearingFormation,
    CircleFormation,
    CubeFormation,
    CustomFormation,
    DisplacementFormation,
    DistanceFormation,
    GridFormation,
    LeaderFollower,
    LineFormation,
    SphereFormation,
    SwarmParams,
    SwarmSimulator,
    VFormation,
    assign_slots,
    available_behaviors,
    available_shapes,
    formation_error,
    get_shape,
    is_infinitesimally_rigid,
    register_shape,
)
from pymapf.swarm.formation import SHAPES

CONTROLLERS = [
    "displacement_formation",
    "distance_formation",
    "bearing_formation",
    "leader_follower",
]

# Shapes that are infinitesimally rigid in the plane, i.e. the ones every
# controller here is entitled to converge on. "line" is deliberately absent --
# see the rigidity tests below.
RIGID_2D_SHAPES = ["v", "circle", "grid"]


# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SHAPES))
@pytest.mark.parametrize("dimension", [2, 3])
def test_every_shape_produces_centred_offsets_of_the_right_size(name, dimension):
    offsets = get_shape(name).centred(7, dimension)
    assert offsets.shape == (7, dimension)
    assert np.all(np.isfinite(offsets))
    assert np.allclose(offsets.mean(axis=0), 0.0, atol=1e-9)


def test_shape_registry_lists_what_get_shape_accepts():
    assert available_shapes() == sorted(SHAPES)
    for name in available_shapes():
        assert get_shape(name).name == name


def test_get_shape_rejects_an_unknown_name():
    with pytest.raises(ValueError):
        get_shape("dodecahedron")


def test_get_shape_passes_through_instances_and_wraps_arrays():
    shape = CircleFormation(radius=4.0)
    assert get_shape(shape) is shape

    offsets = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    wrapped = get_shape(offsets)
    assert isinstance(wrapped, CustomFormation)
    assert np.allclose(wrapped.centred(3, 2), offsets - offsets.mean(axis=0))


def test_register_shape_adds_a_shape_to_the_registry():
    @register_shape("test_pair")
    class PairFormation(LineFormation):
        name = "test_pair"

    try:
        assert "test_pair" in available_shapes()
        assert isinstance(get_shape("test_pair"), PairFormation)
    finally:
        SHAPES.pop("test_pair", None)


def test_line_spacing_is_the_gap_between_adjacent_agents():
    offsets = LineFormation(spacing=2.5).offsets(5, 2)
    gaps = np.linalg.norm(np.diff(offsets, axis=0), axis=1)
    assert gaps == pytest.approx(2.5)


def test_circle_radius_follows_from_spacing_when_not_given():
    offsets = CircleFormation(spacing=2.0).offsets(6, 2)
    radii = np.linalg.norm(offsets, axis=1)
    assert radii == pytest.approx(radii[0])
    # Six agents on a circle: the chord equals the radius.
    assert float(np.linalg.norm(offsets[1] - offsets[0])) == pytest.approx(2.0)


def test_v_formation_is_symmetric_about_the_flight_axis():
    offsets = VFormation(spacing=3.0).offsets(7, 2)
    assert float(offsets[:, 1].sum()) == pytest.approx(0.0, abs=1e-9)
    # Ranks trail progressively further back along the flight axis.
    assert offsets[1, 0] > offsets[3, 0] > offsets[5, 0]


def test_sphere_puts_every_agent_on_the_shell():
    offsets = SphereFormation(radius=5.0).offsets(12, 3)
    assert np.linalg.norm(offsets, axis=1) == pytest.approx(5.0)


def test_three_dimensional_shapes_fall_back_gracefully_in_the_plane():
    assert CubeFormation().offsets(6, 2).shape == (6, 2)
    assert SphereFormation().offsets(6, 2).shape == (6, 2)


def test_custom_formation_refuses_to_invent_slots():
    shape = CustomFormation(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        shape.offsets(5, 2)


def test_shape_distances_and_bearings_agree_with_the_offsets():
    shape = get_shape("grid", spacing=2.0)
    offsets = shape.offsets(6, 2)
    distances = shape.distances(6, 2)
    bearings = shape.bearings(6, 2)
    for i in range(6):
        for j in range(6):
            delta = offsets[j] - offsets[i]
            assert distances[i, j] == pytest.approx(float(np.linalg.norm(delta)))
            if i != j:
                assert bearings[i, j] == pytest.approx(delta / np.linalg.norm(delta))


# --------------------------------------------------------------------------
# assignment
# --------------------------------------------------------------------------


def test_assignment_recovers_a_known_permutation():
    targets = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0]])
    permutation = np.array([2, 0, 3, 1])
    assignment = assign_slots(targets[permutation], targets)
    assert np.array_equal(assignment, permutation)


def test_assignment_beats_matching_by_index():
    rng = np.random.default_rng(7)
    targets = rng.normal(size=(8, 2)) * 4
    positions = rng.normal(size=(8, 2)) * 4
    assignment = assign_slots(positions, targets)

    assert sorted(assignment.tolist()) == list(range(8))  # a permutation
    optimal = np.sum(np.linalg.norm(positions - targets[assignment], axis=1) ** 2)
    by_index = np.sum(np.linalg.norm(positions - targets, axis=1) ** 2)
    assert optimal <= by_index + 1e-9


def test_assignment_requires_a_square_cost():
    with pytest.raises(ValueError):
        assign_slots(np.zeros((3, 2)), np.zeros((4, 2)))


# --------------------------------------------------------------------------
# the error metric and its symmetry groups
# --------------------------------------------------------------------------


def test_error_is_zero_on_the_shape_itself():
    shape = get_shape("v", spacing=3.0)
    assert formation_error(shape.centred(7, 2), shape) == pytest.approx(0.0, abs=1e-9)


def test_error_ignores_translation():
    shape = get_shape("grid", spacing=2.0)
    moved = shape.centred(9, 2) + np.array([40.0, -17.0])
    assert formation_error(moved, shape, allow_rotation=False) == pytest.approx(
        0.0, abs=1e-9
    )


def test_rotation_counts_only_when_the_controller_cannot_see_it():
    shape = get_shape("v", spacing=3.0)
    angle = 0.7
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rotated = shape.centred(7, 2) @ rotation.T

    assert formation_error(rotated, shape, allow_rotation=True) == pytest.approx(
        0.0, abs=1e-9
    )
    assert formation_error(rotated, shape, allow_rotation=False) > 0.5


def test_reflection_counts_only_when_the_controller_cannot_see_it():
    shape = get_shape("v", spacing=3.0)
    mirrored = shape.centred(7, 2) * np.array([1.0, -1.0])

    assert formation_error(
        mirrored, shape, allow_rotation=True, allow_reflection=True
    ) == pytest.approx(0.0, abs=1e-9)
    # A V is symmetric about its own axis, so mirroring it is invisible; a grid
    # with an odd column count is not, which is what makes the flag observable.
    grid = get_shape("grid", spacing=2.0)
    skew = grid.centred(6, 2) @ np.array([[1.0, 0.4], [0.0, 1.0]])
    flipped = skew * np.array([1.0, -1.0])
    assert formation_error(
        flipped, grid, allow_rotation=True, allow_reflection=True
    ) <= formation_error(flipped, grid, allow_rotation=True) + 1e-9


def test_scale_counts_only_when_the_controller_cannot_see_it():
    shape = get_shape("circle", spacing=3.0)
    doubled = shape.centred(8, 2) * 2.0

    assert formation_error(doubled, shape, allow_scaling=True) == pytest.approx(
        0.0, abs=1e-9
    )
    assert formation_error(doubled, shape, allow_scaling=False) > 1.0


def test_error_fits_pose_and_correspondence_together():
    """A rotated *and* permuted formation is still a perfect formation.

    Fitting the pose before knowing the correspondence uses the identity
    pairing, which is wrong, and the resulting rotation is then wrong too. This
    is the case that catches it.
    """
    shape = get_shape("grid", spacing=3.0)
    angle = 1.1
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    scrambled = (shape.centred(9, 2) @ rotation.T)[[4, 0, 7, 2, 8, 1, 5, 3, 6]]
    assert formation_error(scrambled, shape, allow_rotation=True) == pytest.approx(
        0.0, abs=1e-9
    )


# --------------------------------------------------------------------------
# rigidity
# --------------------------------------------------------------------------


def _complete_edges(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def test_a_square_needs_its_diagonal():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    sides = [(0, 1), (1, 2), (2, 3), (3, 0)]
    assert not is_infinitesimally_rigid(square, sides)
    assert is_infinitesimally_rigid(square, sides + [(0, 2)])


def test_a_collinear_formation_is_never_rigid_in_the_plane():
    line = LineFormation(spacing=2.0).centred(6, 2)
    assert not is_infinitesimally_rigid(line, _complete_edges(6))


def test_a_planar_formation_is_not_rigid_in_three_dimensions():
    flat = GridFormation(spacing=2.0).centred(6, 3)
    assert not is_infinitesimally_rigid(flat, _complete_edges(6))
    assert is_infinitesimally_rigid(
        CubeFormation(spacing=2.0).centred(8, 3), _complete_edges(8)
    )


def test_rigidity_needs_edges_and_agents():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert not is_infinitesimally_rigid(square, [])
    assert not is_infinitesimally_rigid(np.array([[0.0, 0.0]]), [(0, 0)])


@pytest.mark.parametrize("name", RIGID_2D_SHAPES)
def test_the_shapes_the_controllers_are_tested_on_are_rigid(name):
    target = get_shape(name, spacing=3.0).centred(9, 2)
    assert is_infinitesimally_rigid(target, _complete_edges(9))


# --------------------------------------------------------------------------
# the controllers
# --------------------------------------------------------------------------


def test_the_controllers_are_registered():
    for name in CONTROLLERS + ["formation"]:
        assert name in available_behaviors()
    # "formation" is the alias for the displacement law, not a fifth controller.
    assert isinstance(SwarmSimulator("formation", n_agents=4).behavior, DisplacementFormation)


@pytest.mark.parametrize("controller", CONTROLLERS)
@pytest.mark.parametrize("shape", RIGID_2D_SHAPES)
def test_every_controller_converges_on_a_rigid_planar_shape(controller, shape):
    simulator = SwarmSimulator(
        controller, n_agents=9, dimension=2, shape=shape, spacing=3.0
    )
    errors = []
    simulator.run(steps=800, observer=lambda step, state: errors.append(
        simulator.behavior.error(state)
    ))
    assert np.all(np.isfinite(errors))
    assert errors[-1] < 0.05 * errors[0]
    assert errors[-1] < 0.1


@pytest.mark.parametrize("controller", CONTROLLERS)
@pytest.mark.parametrize("shape", ["cube", "sphere"])
def test_every_controller_converges_in_three_dimensions(controller, shape):
    simulator = SwarmSimulator(
        controller, n_agents=8, dimension=3, shape=shape, spacing=3.0
    )
    errors = []
    simulator.run(steps=800, observer=lambda step, state: errors.append(
        simulator.behavior.error(state)
    ))
    assert errors[-1] < 0.05 * errors[0]


@pytest.mark.parametrize("controller", CONTROLLERS)
def test_no_controller_drives_agents_into_each_other(controller):
    result = SwarmSimulator(
        controller, n_agents=9, dimension=2, shape="circle", spacing=3.0
    ).run(steps=600)
    # The steady-state window only: a dense spawn is the initial condition's
    # doing, not the controller's.
    assert result.metrics.summary()["steady_collisions"] == 0


@pytest.mark.parametrize("controller", CONTROLLERS)
def test_the_formation_travels_to_a_waypoint(controller):
    params = SwarmParams(seed=3, migration_point=(60.0, 45.0))
    simulator = SwarmSimulator(
        controller, n_agents=9, dimension=2, params=params, shape="v", spacing=3.0
    )
    result = simulator.run(steps=900)
    start = float(np.linalg.norm(result.history[0].positions.mean(axis=0) - params.migration_point))
    end = float(np.linalg.norm(result.final.positions.mean(axis=0) - params.migration_point))
    assert end < 0.25 * start
    assert simulator.behavior.error(result.final) < 0.5


def test_slot_assignment_is_recomputed_but_not_every_step():
    simulator = SwarmSimulator(
        "displacement_formation", n_agents=6, shape="circle", reassign_every=10
    )
    state = simulator.initial_state()
    simulator.behavior.reset(state)
    offsets = simulator.behavior.shape.centred(6, 2)

    first = simulator.behavior.assignment(state, offsets).copy()
    simulator.behavior._steps = 3  # not a multiple of reassign_every
    assert np.array_equal(simulator.behavior.assignment(state, offsets), first)


def test_reset_clears_the_cached_assignment_and_graph():
    simulator = SwarmSimulator("distance_formation", n_agents=6, shape="grid")
    state = simulator.initial_state()
    simulator.behavior.reset(state)
    simulator.behavior.interaction_graph(state)
    assert simulator.behavior._graph is not None

    simulator.behavior.reset(state)
    assert simulator.behavior._graph is None
    assert simulator.behavior._assignment is None


# --------------------------------------------------------------------------
# what each sensing model can and cannot fix
# --------------------------------------------------------------------------


def test_displacement_control_fixes_the_orientation_too():
    simulator = SwarmSimulator(
        "displacement_formation", n_agents=9, shape="v", spacing=3.0
    )
    result = simulator.run(steps=600)
    # Graded without any rotation freedom at all: a shared frame is exactly
    # what this controller assumes it has.
    assert formation_error(
        result.final.positions, simulator.behavior.shape, allow_rotation=False
    ) < 0.1


def test_bearing_control_leaves_the_scale_free():
    simulator = SwarmSimulator(
        "bearing_formation", n_agents=9, shape="circle", spacing=3.0
    )
    result = simulator.run(steps=900)
    shape = simulator.behavior.shape

    assert formation_error(result.final.positions, shape, allow_scaling=True) < 0.05
    achieved = float(np.mean(np.linalg.norm(
        result.final.positions - result.final.positions.mean(axis=0), axis=1
    )))
    desired = float(np.mean(np.linalg.norm(shape.centred(9, 2), axis=1)))
    # Free, not merely unenforced: the size it settles on is its own business.
    assert not np.isclose(achieved, desired, rtol=0.02)


def test_a_scale_gain_pins_the_size_that_bearings_leave_free():
    simulator = SwarmSimulator(
        "bearing_formation", n_agents=9, shape="circle", spacing=3.0, scale_gain=4.0
    )
    result = simulator.run(steps=900)
    shape = simulator.behavior.shape
    achieved = float(np.mean(np.linalg.norm(
        result.final.positions - result.final.positions.mean(axis=0), axis=1
    )))
    desired = float(np.mean(np.linalg.norm(shape.centred(9, 2), axis=1)))
    assert achieved == pytest.approx(desired, rel=0.15)


def test_distance_control_reaches_the_target_distances_exactly():
    simulator = SwarmSimulator(
        "distance_formation", n_agents=9, shape="v", spacing=3.0
    )
    result = simulator.run(steps=1200)
    state = result.final
    desired = simulator.behavior.desired_distances(state)
    achieved = np.linalg.norm(
        state.positions[:, None, :] - state.positions[None, :, :], axis=2
    )
    graph = simulator.behavior.interaction_graph(state)
    residuals = [
        abs(achieved[i, j] - desired[i, j]) for i in range(state.n) for j in graph[i]
    ]
    assert max(residuals) < 1e-3


def test_distance_control_warns_on_a_target_it_cannot_hold():
    with pytest.warns(RuntimeWarning, match="not infinitesimally rigid"):
        SwarmSimulator(
            "distance_formation", n_agents=6, dimension=2, shape="line"
        ).run(steps=5)


def test_a_rigid_target_raises_no_warning():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        SwarmSimulator(
            "distance_formation", n_agents=6, dimension=2, shape="grid"
        ).run(steps=5)


def test_the_interaction_graph_is_augmented_until_it_is_rigid():
    # Two nearest neighbours per agent is nowhere near enough on its own; the
    # graph must grow before the controller will run on it.
    simulator = SwarmSimulator(
        "distance_formation", n_agents=9, dimension=2, shape="grid", degree=2
    )
    state = simulator.initial_state()
    simulator.behavior.reset(state)
    graph = simulator.behavior.interaction_graph(state)

    edges = [(i, j) for i in range(9) for j in graph[i] if j > i]
    assert len(edges) > 9  # more than the k-nearest rule alone produced
    assert len(edges) < len(_complete_edges(9))  # but still sparse
    assert simulator.behavior.rigid(state) or len(edges) == len(_complete_edges(9))


def test_the_default_graph_is_complete():
    simulator = SwarmSimulator("distance_formation", n_agents=6, shape="grid")
    state = simulator.initial_state()
    simulator.behavior.reset(state)
    graph = simulator.behavior.interaction_graph(state)
    assert all(len(graph[i]) == 5 for i in range(6))


@pytest.mark.parametrize("potential", ["linear", "squared"])
def test_both_distance_potentials_share_their_equilibria(potential):
    simulator = SwarmSimulator(
        "distance_formation", n_agents=9, shape="grid", spacing=3.0,
        potential=potential,
    )
    result = simulator.run(steps=1200)
    assert simulator.behavior.error(result.final) < 0.1


def test_an_unknown_potential_is_rejected():
    with pytest.raises(ValueError):
        DistanceFormation(potential="quartic")


def test_leaders_are_not_dragged_by_their_followers():
    """With no mission, a leader holds station rather than chasing the centroid.

    A leader pulled toward the swarm centroid closes a loop through its own
    followers -- they track it, it tracks them -- and the formation never
    settles. This pins the fix.
    """
    simulator = SwarmSimulator(
        "leader_follower", n_agents=9, shape="v", spacing=3.0, leaders=1
    )
    result = simulator.run(steps=600)
    start = result.history[0].positions[0]
    end = result.final.positions[0]
    assert float(np.linalg.norm(end - start)) < 1.0
    assert simulator.behavior.error(result.final) < 0.1


def test_more_leaders_shorten_the_dependency_chain():
    errors = {}
    for leaders in (1, 3):
        simulator = SwarmSimulator(
            "leader_follower", n_agents=9, shape="grid", spacing=3.0, leaders=leaders
        )
        result = simulator.run(steps=600)
        errors[leaders] = simulator.behavior.error(result.final)
    assert all(value < 0.2 for value in errors.values())


def test_a_custom_shape_can_be_flown():
    offsets = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.5], [2.0, -3.5]])
    simulator = SwarmSimulator(
        "displacement_formation", n_agents=4, shape=CustomFormation(offsets)
    )
    result = simulator.run(steps=600)
    assert simulator.behavior.error(result.final) < 0.05


def test_controllers_expose_the_symmetries_they_cannot_resolve():
    assert not DisplacementFormation().allow_rotation
    assert not DisplacementFormation().allow_scaling

    assert DistanceFormation().allow_rotation
    assert DistanceFormation().allow_reflection
    assert DistanceFormation().requires_rigidity

    assert BearingFormation().allow_scaling
    assert BearingFormation().requires_rigidity

    assert not LeaderFollower().allow_rotation


# --------------------------------------------------------------------------
# regressions
# --------------------------------------------------------------------------


def test_error_fits_rotations_about_any_axis_in_three_dimensions():
    """Seeding the fit with planar rotations only misses most of SO(3).

    A cube rotated about an arbitrary axis reported an error of 1.30 when the
    trial poses were all rotations in the first two axes.
    """
    generator = np.random.default_rng(1)
    shape = get_shape("cube", spacing=3.0)
    worst = 0.0
    for _ in range(40):
        rotation, _ = np.linalg.qr(generator.normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        placed = (shape.centred(8, 3) @ rotation.T)[generator.permutation(8)]
        worst = max(worst, formation_error(placed, shape, allow_rotation=True))
    assert worst < 1e-6


@pytest.mark.parametrize("name,n", [("sphere", 12), ("cube", 8), ("v", 7), ("grid", 9)])
def test_error_is_exact_on_every_shape_under_rotation(name, n):
    """The radius seed and the rotation seeds fail on opposite shapes.

    Distinct radii (sphere, V) are solved by rank-matching and missed by a
    handful of SO(3) samples; degenerate radii (cube, grid) are the reverse.
    Only using both is exact everywhere.
    """
    generator = np.random.default_rng(2)
    shape = get_shape(name)
    base = shape.centred(n, 3)
    for _ in range(20):
        rotation, _ = np.linalg.qr(generator.normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        placed = (base @ rotation.T)[generator.permutation(n)]
        assert formation_error(placed, shape, allow_rotation=True) < 1e-6


def test_the_error_metric_is_deterministic():
    shape = get_shape("sphere")
    placed = shape.centred(10, 3) @ np.linalg.qr(np.arange(9).reshape(3, 3) + np.eye(3))[0]
    first = formation_error(placed, shape, allow_rotation=True)
    assert all(
        formation_error(placed, shape, allow_rotation=True) == first for _ in range(5)
    )


def test_the_assignment_is_solved_once_per_step_not_once_per_agent():
    """It is called per agent and cannot change within a step; re-solving makes
    the step O(n^4)."""
    import pymapf.swarm.formation as formation_module

    calls = []
    original = formation_module.assign_slots
    formation_module.assign_slots = lambda *a, **k: (
        calls.append(1), original(*a, **k)
    )[1]
    try:
        SwarmSimulator(
            "displacement_formation", n_agents=12, shape="grid", reassign_every=25
        ).run(steps=50)
    finally:
        formation_module.assign_slots = original
    assert len(calls) <= 5  # one initial solve plus 50/25 reassignments


@pytest.mark.parametrize("controller", ["distance_formation", "bearing_formation"])
def test_controllers_with_derived_targets_freeze_their_assignment(controller):
    """The assignment is baked into the desired distances, bearings and the
    interaction graph at reset. Re-solving it would move those targets under the
    descent -- and while the derived values are cached it silently does nothing,
    which is worse. It is now explicit rather than accidental."""
    assert SwarmSimulator(controller, n_agents=6).behavior.reassigns is False

    finals = [
        SwarmSimulator(
            controller, n_agents=9, shape="v", spacing=3.0, reassign_every=every
        ).run(steps=400).final.positions
        for every in (5, 10_000)
    ]
    assert np.allclose(finals[0], finals[1])


def test_displacement_control_reassigns_while_flying():
    assert SwarmSimulator("displacement_formation", n_agents=6).behavior.reassigns


@pytest.mark.parametrize(
    "controller", ["displacement_formation", "distance_formation", "bearing_formation"]
)
def test_the_shape_term_never_drives_the_centroid(controller):
    """The invariant behind "a waypoint is a translation, not an attraction".

    Each of these laws is translation-invariant, so its shape term must sum to
    zero over the swarm: it arranges agents *relative to each other* and leaves
    the group's position to exactly one other term. Placing the slots at the
    waypoint broke that -- the shape term then drove the group there too, and
    two terms pulling at one point overshoot it.

    Leader-follower is excluded on purpose: followers chase leaders and leaders
    do not chase back, so it is asymmetric by construction.

    The acceleration clamp is lifted for the same reason the velocities are
    zeroed: both are per-agent nonlinearities that legitimately break the sum,
    and neither is what this test is about.
    """
    params = SwarmParams(max_acceleration=1e6, max_speed=1e6)
    simulator = SwarmSimulator(
        controller, n_agents=9, params=params, shape="v", spacing=3.0
    )
    state = simulator.initial_state()
    state.velocities = np.zeros_like(state.velocities)  # isolate the shape term
    simulator.behavior.reset(state)
    simulator.behavior._steps = 1

    commands = np.array(
        [simulator.behavior.command(state, i) for i in range(state.n)]
    )
    assert np.abs(commands).max() < 1e5  # nothing clamped
    assert np.allclose(commands.sum(axis=0), 0.0, atol=1e-8)


def test_the_formation_reaches_the_waypoint_without_running_past_it():
    """Displacement and distance control settle on the waypoint monotonically.

    Bearing control is checked separately: it is the slowest law here by a
    factor of ten, so its shape term is still small while the (saturated)
    migration term is building speed, and it arrives as an ordinary
    lightly-damped second-order system -- overshooting and settling back. The
    others are saturated by their own shape term early on, which incidentally
    brakes them.
    """
    params = SwarmParams(seed=3, migration_point=(60.0, 45.0))
    for controller in ["displacement_formation", "distance_formation",
                       "leader_follower"]:
        simulator = SwarmSimulator(
            controller, n_agents=9, params=params, shape="v", spacing=3.0
        )
        result = simulator.run(steps=900)
        centroids = np.array([state.positions.mean(axis=0) for state in result.history])
        start = centroids[0]
        heading = np.asarray(params.migration_point, dtype=float) - start
        progress = (centroids - start) @ heading / (heading @ heading)
        assert progress.max() <= 1.005, controller
        assert progress[-1] == pytest.approx(1.0, abs=0.01), controller


def test_bearing_control_overshoots_the_waypoint_but_settles_on_it():
    params = SwarmParams(seed=3, migration_point=(60.0, 45.0))
    simulator = SwarmSimulator(
        "bearing_formation", n_agents=9, params=params, shape="v", spacing=3.0
    )
    result = simulator.run(steps=1200)
    centroids = np.array([state.positions.mean(axis=0) for state in result.history])
    start = centroids[0]
    heading = np.asarray(params.migration_point, dtype=float) - start
    progress = (centroids - start) @ heading / (heading @ heading)

    assert 1.0 < progress.max() < 1.2          # overshoots, but bounded
    assert progress[-1] == pytest.approx(1.0, abs=0.01)   # and settles on it
    # More damping trades the overshoot away, confirming what it is.
    damped = SwarmSimulator(
        "bearing_formation", n_agents=9, params=params, shape="v", spacing=3.0,
        damping=6.0,
    ).run(steps=1200)
    damped_centroids = np.array([s.positions.mean(axis=0) for s in damped.history])
    damped_progress = (damped_centroids - start) @ heading / (heading @ heading)
    assert damped_progress.max() < progress.max()


def test_the_formation_centre_is_the_swarm_not_the_waypoint():
    simulator = SwarmSimulator(
        "displacement_formation", n_agents=6,
        params=SwarmParams(migration_point=(50.0, 50.0)),
    )
    state = simulator.initial_state()
    simulator.behavior.reset(state)
    assert np.allclose(simulator.behavior.centre(state), state.centroid)
