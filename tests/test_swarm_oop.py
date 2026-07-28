"""The object-oriented swarm layer: behaviors, neighbourhoods, domains, coverage."""

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pytest

np = pytest.importorskip("numpy")

from pymapf.swarm import (
    AnnulusDomain,
    Behavior,
    CompositeBehavior,
    ConeNeighborhood,
    CoverageSimulator,
    DiskDomain,
    GaussianKernelNeighborhood,
    GaussianMixtureDensity,
    HemisphereDomain,
    MeshDomain,
    MetricNeighborhood,
    MixtureAssignment,
    PlanarDomain,
    SphereDomain,
    SwarmParams,
    SwarmSimulator,
    SwarmState,
    TimeVaryingDensity,
    UniformDensity,
    available_behaviors,
    available_coverage,
    get_behavior,
    get_coverage,
    get_domain,
    get_neighborhood,
    register_behavior,
    TopologicalNeighborhood,
)

FLOCKING = [
    "boids",
    "vicsek",
    "cucker_smale",
    "olfati_saber",
    "proximal",
    "active_elastic",
    "acceleration",
    "gaussian_kernel",
    "minimalistic",
    "distributed_3d",
]


# ------------------------------------------------------------------ core ---


def test_every_flocking_model_is_registered():
    assert set(FLOCKING) <= set(available_behaviors())


@pytest.mark.parametrize("name", FLOCKING)
def test_commands_are_finite_and_within_limits(name):
    params = SwarmParams()
    state = SwarmState.lattice(9, 2)
    behavior = get_behavior(name, params=params)
    behavior.reset(state)
    for index in range(state.n):
        command = behavior.command(state, index)
        assert command.shape == (2,)
        assert np.all(np.isfinite(command))
        ceiling = params.max_speed if behavior.output == "velocity" else params.max_acceleration
        assert np.linalg.norm(command) <= ceiling + 1e-6


@pytest.mark.parametrize("name", FLOCKING)
@pytest.mark.parametrize("dimension", [2, 3])
def test_models_are_dimension_agnostic(name, dimension):
    state = SwarmState.lattice(8, dimension)
    behavior = get_behavior(name)
    behavior.reset(state)
    assert behavior.command(state, 0).shape == (dimension,)


def test_unknown_behavior_lists_the_available_ones():
    with pytest.raises(ValueError) as error:
        get_behavior("murmuration")
    assert "acceleration" in str(error.value)


def test_a_new_behavior_can_be_registered_from_outside():
    @register_behavior("test-drift")
    class Drift(Behavior):
        def command(self, state, index):
            return self.finalise(np.ones(state.dimension), state, index)

    assert "test-drift" in available_behaviors()
    assert isinstance(get_behavior("test-drift"), Drift)
    result = SwarmSimulator("test-drift", n_agents=4).run(steps=5)
    assert len(result.history) == 6


def test_composite_sums_its_parts():
    params = SwarmParams()
    state = SwarmState.lattice(6, 2)
    a = get_behavior("vicsek", params=params)
    b = get_behavior("acceleration", params=params)
    composite = CompositeBehavior([(a, 1.0), (b, 1.0)], params=params)
    composite.reset(state)
    expected = a.command(state, 0) + b.command(state, 0)
    assert np.allclose(
        composite.command(state, 0),
        expected if np.linalg.norm(expected) <= params.max_acceleration
        else expected * params.max_acceleration / np.linalg.norm(expected),
    )


def test_composite_requires_parts():
    with pytest.raises(ValueError):
        CompositeBehavior([])


def test_state_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        SwarmState(np.zeros((3, 2)), np.zeros((4, 2)))


def test_lattice_spawn_is_legal():
    params = SwarmParams()
    state = SwarmState.lattice(16, 2, spacing=params.reference_distance)
    distances = np.linalg.norm(
        state.positions[:, None, :] - state.positions[None, :, :], axis=2
    )
    np.fill_diagonal(distances, np.inf)
    assert distances.min() >= params.separation_distance


# --------------------------------------------------------- neighbourhoods ---


def test_topological_neighbourhood_has_fixed_degree():
    state = SwarmState.lattice(16, 2)
    params = SwarmParams()
    assert np.all(TopologicalNeighborhood(k=5).degree(state, params) == 5)


def test_metric_neighbourhood_respects_its_radius():
    state = SwarmState.lattice(16, 2, spacing=3.0)
    params = SwarmParams()
    tight = MetricNeighborhood(radius=3.5).degree(state, params)
    wide = MetricNeighborhood(radius=10.0).degree(state, params)
    assert np.all(tight <= wide)
    assert tight.mean() < wide.mean()


def test_cone_neighbourhood_excludes_agents_behind():
    state = SwarmState(
        np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]),
        np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
    )
    neighbours = ConeNeighborhood(half_angle=np.pi / 4).of(state, 0, SwarmParams())
    assert 1 in neighbours and 2 not in neighbours


def test_gaussian_kernel_weights_decay_with_distance():
    state = SwarmState(
        np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]), np.zeros((3, 2))
    )
    kernel = GaussianKernelNeighborhood(sigma=2.0)
    params = SwarmParams()
    neighbours = kernel.of(state, 0, params)
    weights = kernel.weights(state, 0, params)
    assert len(weights) == len(neighbours)
    # the nearer neighbour must carry more influence
    order = np.argsort(state.distances_from(0)[neighbours])
    assert weights[order[0]] > weights[order[-1]]


def test_unknown_neighborhood_is_rejected():
    with pytest.raises(ValueError):
        get_neighborhood("telepathic")


def test_a_behavior_accepts_any_neighbourhood():
    state = SwarmState.lattice(12, 2)
    for neighborhood in (
        MetricNeighborhood(),
        TopologicalNeighborhood(k=3),
        ConeNeighborhood(),
        GaussianKernelNeighborhood(),
    ):
        behavior = get_behavior("acceleration", neighborhood=neighborhood)
        behavior.reset(state)
        assert np.all(np.isfinite(behavior.command(state, 0)))


# -------------------------------------------------------------- dynamics ---


def test_acceleration_model_sustains_motion_without_a_waypoint():
    """Self-propulsion is the distinguishing feature: the flock keeps flying."""
    summary = SwarmSimulator(
        "acceleration", n_agents=16, params=SwarmParams(seed=1)
    ).run(steps=250).metrics.summary()
    assert summary["mean_speed"] > 1.0
    assert summary["order"] > 0.8
    assert summary["steady_collisions"] == 0


def test_active_elastic_aligns_without_sensing_any_velocity():
    """Ferrante's claim: alignment emerges from elasticity alone."""
    result = SwarmSimulator(
        "active_elastic", n_agents=20, params=SwarmParams(seed=1)
    ).run(steps=300)
    summary = result.metrics.summary()
    assert summary["order"] > 0.7
    assert summary["steady_collisions"] == 0


def test_active_elastic_uses_only_relative_positions():
    """If it read velocities, scrambling them would change the command."""
    state = SwarmState.lattice(10, 2)
    behavior = get_behavior("active_elastic")
    behavior.reset(state)
    force = behavior.elastic_force(state, 0)

    scrambled = state.copy()
    scrambled.velocities = np.random.default_rng(3).normal(size=state.velocities.shape)
    assert np.allclose(force, behavior.elastic_force(scrambled, 0))


def test_olfati_saber_settles_at_the_reference_spacing():
    params = SwarmParams(seed=1)
    summary = SwarmSimulator("olfati_saber", n_agents=16, params=params).run(
        steps=250
    ).metrics.summary()
    assert summary["min_distance"] > 0.8 * params.reference_distance
    assert summary["steady_collisions"] == 0


def test_cucker_smale_reaches_velocity_consensus():
    result = SwarmSimulator("cucker_smale", n_agents=12, params=SwarmParams(seed=4)).run(
        steps=200
    )
    velocities = result.final.velocities
    spread = np.linalg.norm(velocities - velocities.mean(axis=0), axis=1).mean()
    assert spread < 0.5


def test_bounds_reflect_agents_back_inside():
    params = SwarmParams(seed=1, bounds=(-10.0, -10.0, 10.0, 10.0))
    result = SwarmSimulator("acceleration", n_agents=10, params=params).run(steps=200)
    positions = result.final.positions
    assert np.all(positions >= -10.5) and np.all(positions <= 10.5)


def test_simulation_is_reproducible():
    a = SwarmSimulator("acceleration", n_agents=8, params=SwarmParams(seed=9)).run(steps=50)
    b = SwarmSimulator("acceleration", n_agents=8, params=SwarmParams(seed=9)).run(steps=50)
    assert np.allclose(a.final.positions, b.final.positions)


def test_observer_sees_every_step():
    seen = []
    SwarmSimulator("boids", n_agents=5).run(steps=10, observer=lambda i, s: seen.append(i))
    assert seen == list(range(11))


# --------------------------------------------------------------- domains ---


@pytest.mark.parametrize(
    "domain",
    [PlanarDomain(), DiskDomain(), SphereDomain(), HemisphereDomain(), AnnulusDomain()],
)
def test_domains_sample_project_and_measure(domain):
    points = domain.sample(400)
    assert points.shape[1] == domain.dimension
    projected = domain.project(points.copy())
    assert np.allclose(projected, domain.project(projected))  # idempotent
    assert domain.measure > 0


def test_sphere_uses_great_circle_distance():
    domain = SphereDomain(radius=1.0)
    north = np.array([[0.0, 0.0, 1.0]])
    equator = np.array([[1.0, 0.0, 0.0]])
    assert np.isclose(domain.distance(north, equator)[0, 0], np.pi / 2)


def test_hemisphere_keeps_agents_above_the_equator():
    domain = HemisphereDomain(radius=5.0)
    below = np.array([[0.0, 0.0, -5.0], [3.0, 0.0, -4.0]])
    projected = domain.project(below)
    assert np.all(projected[:, 2] >= -1e-9)
    assert np.allclose(np.linalg.norm(projected, axis=1), 5.0)


def test_annulus_rejects_inverted_radii():
    with pytest.raises(ValueError):
        AnnulusDomain(inner=10.0, outer=5.0)


def test_mesh_domain_snaps_to_its_points():
    points = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
    domain = MeshDomain(points)
    assert np.allclose(domain.project(np.array([[4.0, 0.4]])), [[5.0, 0.0]])


def test_unknown_domain_is_rejected():
    with pytest.raises(ValueError):
        get_domain("hyperbolic")


# ------------------------------------------------------------- densities ---


def test_mixture_density_peaks_at_its_components():
    mixture = GaussianMixtureDensity(means=[(0.0, 0.0), (10.0, 0.0)], covariances=[1.0, 1.0])
    values = mixture(np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]))
    assert values[0] > values[1] and values[2] > values[1]


def test_mixture_responsibilities_sum_to_one():
    mixture = GaussianMixtureDensity(means=[(0.0, 0.0), (10.0, 0.0)], covariances=[2.0, 2.0])
    responsibilities = mixture.responsibilities(np.array([[1.0, 0.0], [9.0, 0.0]]))
    assert np.allclose(responsibilities.sum(axis=1), 1.0)
    assert responsibilities[0, 0] > 0.5 and responsibilities[1, 1] > 0.5


def test_em_recovers_the_generating_components():
    truth = GaussianMixtureDensity(
        means=[(0.0, 0.0), (12.0, 8.0)], covariances=[1.0, 1.0], weights=[0.5, 0.5]
    )
    samples = truth.sample(600, np.random.default_rng(0))
    fitted = GaussianMixtureDensity.fit(samples, k=2, seed=1)
    recovered = np.sort(fitted.means, axis=0)
    assert np.allclose(recovered, np.array([[0.0, 0.0], [12.0, 8.0]]), atol=1.0)


def test_mixture_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        GaussianMixtureDensity(means=[(0.0, 0.0)], weights=[0.5, 0.5])


def test_time_varying_density_moves_its_peak():
    from pymapf.swarm.density import GaussianDensity

    moving = TimeVaryingDensity(GaussianDensity((0.0, 0.0), sigma=1.0), motion=lambda t: (t, 0.0))
    probe = np.array([[5.0, 0.0]])
    assert moving(probe, time=5.0)[0] > moving(probe, time=0.0)[0]


# -------------------------------------------------------------- coverage ---


def test_coverage_controllers_are_registered():
    assert {"lloyd", "limited_range", "adaptive", "gmm", "time_varying"} <= set(
        available_coverage()
    )


@pytest.mark.parametrize("domain", ["planar", "disk", "sphere", "hemisphere", "annulus"])
def test_lloyd_reduces_cost_on_every_domain(domain):
    result = CoverageSimulator("lloyd", domain=domain, n_agents=8, seed=1).run(steps=30)
    assert result.improvement > 0.3
    assert len(result.history) == 31


def test_limited_range_is_harder_than_unlimited():
    unlimited = CoverageSimulator("lloyd", n_agents=6, seed=1).run(steps=30)
    limited = CoverageSimulator(
        "limited_range", sensing_range=3.0, n_agents=6, seed=1
    ).run(steps=30)
    assert limited.improvement < unlimited.improvement


def test_agents_stay_on_their_domain():
    result = CoverageSimulator("lloyd", domain="hemisphere", n_agents=6, seed=2).run(steps=20)
    final = result.final
    assert np.allclose(np.linalg.norm(final, axis=1), 10.0)
    assert np.all(final[:, 2] >= -1e-6)


def test_density_pulls_the_team_toward_the_hot_spots():
    mixture = GaussianMixtureDensity(means=[(3.0, 3.0)], covariances=[2.0])
    weighted = CoverageSimulator("lloyd", density=mixture, n_agents=6, seed=1).run(steps=30)
    flat = CoverageSimulator("lloyd", n_agents=6, seed=1).run(steps=30)
    centre = np.array([3.0, 3.0])
    assert np.mean(np.linalg.norm(weighted.final - centre, axis=1)) < np.mean(
        np.linalg.norm(flat.final - centre, axis=1)
    )


def test_adaptive_coverage_learns_the_density():
    truth = GaussianMixtureDensity(means=[(4.0, 4.0), (16.0, 16.0)], covariances=[3.0, 3.0])
    simulator = CoverageSimulator("adaptive", truth=truth, n_agents=8, seed=1)
    result = simulator.run(steps=40)
    errors = result.extras["estimation_error"]
    assert errors[-1] <= errors[0]  # the estimate did not get worse
    assert result.improvement > 0.0


def test_mixture_coverage_splits_the_team_across_components():
    mixture = GaussianMixtureDensity(
        means=[(4.0, 4.0), (16.0, 16.0)], covariances=[2.0, 2.0], weights=[0.5, 0.5]
    )
    simulator = CoverageSimulator("gmm", mixture=mixture, n_agents=8, seed=1)
    simulator.run(steps=30)
    assignment = simulator.controller._assignment
    assert set(np.unique(assignment)) == {0, 1}  # both components are served


def test_time_varying_coverage_tracks_a_moving_density():
    from pymapf.swarm.density import GaussianDensity

    moving = TimeVaryingDensity(
        GaussianDensity((5.0, 10.0), sigma=2.0), motion=lambda t: (0.25 * t, 0.0)
    )
    result = CoverageSimulator(
        "time_varying", density=moving, n_agents=6, seed=1, dt=1.0
    ).run(steps=30)
    # The team must have followed the peak to the right, not stayed put.
    assert result.final[:, 0].mean() > result.history[0][:, 0].mean()


def test_unknown_coverage_controller_is_rejected():
    with pytest.raises(ValueError):
        get_coverage("telekinesis")


# ---------------------------------------------------------- distribution ---


def _match_error(positions, mixture, seed=0):
    draws = mixture.sample(2000, np.random.default_rng(seed))
    return float(
        np.linalg.norm(draws[:, None, :] - positions[None, :, :], axis=2).min(axis=1).mean()
    )


@pytest.mark.parametrize("behavior", ["density_matching", "mixture_assignment"])
def test_distribution_control_matches_the_target(behavior):
    target = GaussianMixtureDensity(
        means=[(-8.0, 0.0), (8.0, 4.0)], covariances=[3.0, 3.0], weights=[0.5, 0.5]
    )
    kwargs = {"target": target} if behavior == "density_matching" else {"mixture": target}
    result = SwarmSimulator(
        behavior, n_agents=20, params=SwarmParams(seed=2, max_speed=6), **kwargs
    ).run(steps=300)
    before = _match_error(result.history[0].positions, target)
    after = _match_error(result.final.positions, target)
    assert after < before


def test_mixture_assignment_allocates_roughly_by_weight():
    target = GaussianMixtureDensity(
        means=[(-10.0, 0.0), (10.0, 0.0)], covariances=[2.0, 2.0], weights=[0.75, 0.25]
    )
    simulator = SwarmSimulator(
        "mixture_assignment", n_agents=20, params=SwarmParams(seed=1, max_speed=6), mixture=target
    )
    simulator.run(steps=200)
    counts = np.bincount(simulator.behavior._assignment, minlength=2)
    assert counts[0] > counts[1]  # the heavier component gets more agents
    assert counts.min() >= 1  # ...but nobody is abandoned


def test_mixture_assignment_rejects_a_non_mixture_target():
    with pytest.raises(TypeError):
        MixtureAssignment(mixture=UniformDensity())


def test_density_matching_works_in_three_dimensions():
    target = GaussianMixtureDensity(
        means=[(0.0, 0.0, 0.0), (10.0, 0.0, 5.0)], covariances=[2.0, 2.0]
    )
    result = SwarmSimulator(
        "density_matching", n_agents=12, dimension=3,
        params=SwarmParams(seed=1, max_speed=6), target=target
    ).run(steps=200)
    assert np.all(np.isfinite(result.final.positions))
    assert _match_error(result.final.positions, target) < _match_error(
        result.history[0].positions, target
    )


# ------------------------------- the post-2020 flocking models -------------
#
# Both come from the Albani / Ferrante / Manoni / Saska cluster and both make a
# specific, testable claim beyond "it flocks". These pin the claims.


def test_minimalistic_flocks_on_range_and_bearing_alone():
    """Amorim et al. (2024): cohesive aligned flight with nothing else sensed.

    No GPS, no compass, no communication, no velocity sensing -- and the group
    still agrees on a heading nobody transmitted.
    """
    for dimension in (2, 3):
        summary = SwarmSimulator(
            "minimalistic", n_agents=20, dimension=dimension,
            params=SwarmParams(seed=1),
        ).run(steps=400).metrics.summary()
        assert summary["order"] > 0.7
        assert summary["mean_speed"] > 0.5
        assert summary["steady_collisions"] == 0


def test_minimalistic_never_reads_a_velocity():
    state = SwarmState.lattice(10, 3)
    behavior = get_behavior("minimalistic")
    behavior.reset(state)
    force = behavior.elastic_force(state, 0)

    scrambled = state.copy()
    scrambled.velocities = np.random.default_rng(5).normal(size=state.velocities.shape)
    assert np.allclose(force, behavior.elastic_force(scrambled, 0))


def test_minimalistic_attraction_is_bounded_and_repulsion_is_not():
    """The Lennard-Jones trade: no straggler can drag the flock apart, and the
    safety margin is enforced by the potential rather than bolted on."""
    behavior = get_behavior("minimalistic", params=SwarmParams())
    reference = behavior.params.reference_distance

    assert behavior.proximal(reference * 0.5) < 0        # repels when too close
    assert behavior.proximal(reference * 1.5) > 0        # attracts when too far
    assert behavior.proximal(reference) == pytest.approx(0.0, abs=1e-9)

    # Attraction decays with distance; repulsion grows without it.
    assert behavior.proximal(reference * 3) < behavior.proximal(reference * 1.5)
    assert behavior.proximal(reference * 0.3) < behavior.proximal(reference * 0.6)


@pytest.mark.parametrize("name", ["proximal", "minimalistic", "distributed_3d"])
def test_the_proximal_family_rests_at_the_reference_distance(name):
    """A Lennard-Jones force is zero at ``2^(1/m) sigma``, not at ``sigma``.

    Passing the reference distance straight in as ``sigma`` builds a controller
    whose rest spacing is 41% wider than its own configuration -- and in open
    space that gap compounds until the flock sheds its outer agents.
    """
    behavior = get_behavior(name, params=SwarmParams())
    reference = behavior.params.reference_distance
    assert behavior.proximal(reference) == pytest.approx(0.0, abs=1e-9)
    assert behavior.equilibrium_sigma() < reference


def test_distributed_3d_settles_into_a_flatter_lattice_than_an_isotropic_law():
    """Albani et al. (2022): a multirotor swarm should not be a ball."""
    def aspect(name):
        params = SwarmParams(seed=1, migration_point=(60.0, 60.0, 10.0))
        final = SwarmSimulator(
            name, n_agents=20, dimension=3, params=params
        ).run(steps=500).final.positions
        offsets = final - final.mean(axis=0)
        horizontal = float(np.sqrt(np.mean(np.sum(offsets[:, :2] ** 2, axis=1))))
        vertical = float(np.sqrt(np.mean(offsets[:, 2] ** 2)))
        return vertical / max(horizontal, 1e-9)

    assert aspect("distributed_3d") < 0.8 * aspect("proximal")


def test_the_vertical_weighting_is_capped_by_the_safety_margin():
    """Flattening stops where it would hold the swarm at a spacing the same
    controller calls a collision."""
    params = SwarmParams()
    behavior = get_behavior("distributed_3d", params=params, vertical_scale=10.0)
    scale = behavior.effective_vertical_scale()
    assert scale < 10.0
    assert params.reference_distance / scale >= behavior.safety_margin * params.separation_distance


def test_the_vertical_weighting_does_nothing_in_the_plane():
    state = SwarmState.lattice(9, 2)
    flat = get_behavior("distributed_3d", vertical_scale=4.0)
    plain = get_behavior("distributed_3d", vertical_scale=1.0)
    flat.reset(state)
    plain.reset(state)
    for index in range(state.n):
        assert np.allclose(flat.command(state, index), plain.command(state, index))


def test_downwash_pushes_a_drone_off_the_one_below_it():
    state = SwarmState.lattice(2, 3)
    state.positions = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, -1.0]])
    state.velocities = np.zeros((2, 3))
    behavior = get_behavior("distributed_3d", params=SwarmParams())
    behavior.reset(state)

    # Agent 0 is directly above agent 1, so agent 0 climbs out of the stack.
    assert behavior.downwash(state, 0)[2] > 0
    assert behavior.downwash(state, 1)[2] == 0


def test_downwash_ignores_neighbours_that_are_merely_nearby():
    state = SwarmState.lattice(2, 3)
    state.positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, -1.0]])
    state.velocities = np.zeros((2, 3))
    behavior = get_behavior("distributed_3d", params=SwarmParams())
    behavior.reset(state)
    assert np.allclose(behavior.downwash(state, 0), 0.0)


def test_the_proximal_family_is_cohesive_when_it_has_somewhere_to_go():
    """Documented limitation, pinned so it does not drift into a surprise.

    Proximal control has a *bounded*, decaying attraction. In open space with no
    goal, twenty agents in the plane spread past each other's interaction range
    and the group disperses -- the models were designed for confined
    environments and for missions with a direction. Give them either and they
    are tight.
    """
    open_space = SwarmSimulator(
        "proximal", n_agents=20, dimension=2, params=SwarmParams(seed=1)
    ).run(steps=400).metrics.summary()

    with_goal = SwarmSimulator(
        "proximal", n_agents=20, dimension=2,
        params=SwarmParams(seed=1, migration_point=(60.0, 60.0)),
    ).run(steps=400).metrics.summary()

    assert open_space["cohesion"] > 20.0
    assert with_goal["cohesion"] < 8.0
    assert with_goal["steady_collisions"] == 0
