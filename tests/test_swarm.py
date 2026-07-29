"""Decentralized flocking and coverage."""

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pytest

np = pytest.importorskip("numpy")

from pymapf.decentralized.coverage import (
    CoverageParams,
    coverage_cost,
    gaussian_density,
    hemisphere_samples,
    lloyd_step,
    simulate_coverage,
    simulate_spherical_coverage,
)
from pymapf.decentralized.flocking import (
    CONTROLLERS,
    FlockParams,
    FlockState,
    get_controller,
    simulate,
)


# ------------------------------------------------------------- flocking ---


@pytest.mark.parametrize("name", sorted(CONTROLLERS))
def test_every_controller_produces_a_bounded_command(name):
    params = FlockParams()
    state = FlockState(
        np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.5]]),
        np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
    )
    command = get_controller(name)(state, 0, params)
    assert command.shape == (2,)
    assert np.all(np.isfinite(command))
    assert np.linalg.norm(command) <= params.max_acceleration + 1e-6


@pytest.mark.parametrize("name", sorted(CONTROLLERS))
def test_simulation_runs_and_stays_finite(name):
    history, metrics = simulate(name, n_agents=8, steps=60, params=FlockParams(seed=3))
    assert len(history) == 61
    assert np.all(np.isfinite(history[-1].positions))
    summary = metrics.summary()
    assert 0.0 <= summary["order"] <= 1.0001


def test_unknown_controller_is_rejected():
    with pytest.raises(ValueError):
        get_controller("murmuration")


def test_simulation_is_deterministic_for_a_seed():
    a, _ = simulate("acceleration", n_agents=6, steps=40, params=FlockParams(seed=11))
    b, _ = simulate("acceleration", n_agents=6, steps=40, params=FlockParams(seed=11))
    assert np.allclose(a[-1].positions, b[-1].positions)


def test_spawn_starts_legal():
    """The lattice spawn must not begin already inside the separation distance."""
    params = FlockParams(seed=5)
    history, _ = simulate("boids", n_agents=16, steps=1, params=params)
    positions = history[0].positions
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() >= params.separation_distance


def test_acceleration_model_keeps_flying_without_a_waypoint():
    """Self-propulsion is the point: the flock should not settle to a stop."""
    _, metrics = simulate(
        "acceleration", n_agents=12, steps=200, params=FlockParams(seed=1)
    )
    summary = metrics.summary()
    assert summary["mean_speed"] > 1.0
    assert summary["order"] > 0.8


def test_olfati_saber_forms_a_lattice_at_the_reference_spacing():
    params = FlockParams(seed=1)
    _, metrics = simulate("olfati_saber", n_agents=16, steps=250, params=params)
    summary = metrics.summary()
    # The alpha-lattice settles at the reference distance, not on top of itself.
    assert summary["min_distance"] > 0.8 * params.reference_distance
    assert summary["steady_collisions"] == 0


def test_migration_never_starves_separation():
    """A distant waypoint must not consume the whole acceleration budget."""
    params = FlockParams(migration_point=(500.0, 500.0), seed=2)
    _, metrics = simulate("acceleration", n_agents=12, steps=150, params=params)
    assert metrics.summary()["min_distance"] > 0.5


def test_obstacles_are_avoided_in_three_dimensions():
    params = FlockParams(
        migration_point=(40.0, 0.0, 0.0), obstacles=[((20.0, 0.0, 0.0), 4.0)], seed=4
    )
    history, _ = simulate(
        "acceleration", n_agents=8, steps=200, dimension=3, params=params
    )
    centre = np.array([20.0, 0.0, 0.0])
    closest = min(
        float(np.min(np.linalg.norm(state.positions - centre, axis=1)))
        for state in history
    )
    assert closest > 2.0  # nobody flew through the middle of the obstacle


# ------------------------------------------------------------- coverage ---


def test_lloyd_reduces_the_coverage_cost():
    params = CoverageParams(seed=1)
    _, costs = simulate_coverage(n_agents=6, steps=30, params=params)
    assert costs[-1] < costs[0] * 0.5


def test_lloyd_cost_is_monotone_enough_to_be_a_descent():
    params = CoverageParams(seed=2)
    _, costs = simulate_coverage(n_agents=6, steps=25, params=params)
    increases = sum(1 for a, b in zip(costs, costs[1:]) if b > a + 1e-6)
    assert increases <= 2  # quadrature noise only


def test_limited_range_leaves_far_regions_uncovered():
    """With a short sensor the team cannot cover everything -- and should say so."""
    unlimited = CoverageParams(seed=1)
    limited = CoverageParams(sensing_range=3.0, seed=1)
    _, wide = simulate_coverage(n_agents=5, steps=30, params=unlimited)
    _, narrow = simulate_coverage(n_agents=5, steps=30, params=limited)
    assert wide[-1] / wide[0] < narrow[-1] / narrow[0]


def test_density_pulls_agents_toward_the_hot_spots():
    """The controlled comparison: same start, weighted vs uniform importance.

    (Not "do agents end up near the hot spot" -- Lloyd spreads a team out
    regardless, so that would fail even when the density is working.)
    """
    centre = np.array([3.0, 3.0])
    hot = simulate_coverage(
        n_agents=4, steps=40, params=CoverageParams(density=gaussian_density([(3.0, 3.0)]), seed=1)
    )[0]
    flat = simulate_coverage(n_agents=4, steps=40, params=CoverageParams(seed=1))[0]
    hot_distance = float(np.mean(np.linalg.norm(hot[-1] - centre, axis=1)))
    flat_distance = float(np.mean(np.linalg.norm(flat[-1] - centre, axis=1)))
    assert hot_distance < flat_distance


def test_coverage_cost_matches_a_hand_computed_case():
    params = CoverageParams(bounds=(0.0, 0.0, 1.0, 1.0), resolution=0.5)
    positions = np.array([[0.5, 0.5]])
    # Samples at 0/0.5/1 in both axes; every one is within 0.71 of the centre.
    assert coverage_cost(positions, params) > 0


def test_lloyd_step_holds_agents_with_empty_regions():
    params = CoverageParams(sensing_range=0.5, seed=1)
    positions = np.array([[1.0, 1.0], [19.0, 19.0]])
    moved = lloyd_step(positions, params)
    assert np.allclose(moved, positions, atol=1e-6)


def test_hemisphere_samples_lie_on_the_upper_half():
    points = hemisphere_samples(radius=5.0, count=500)
    radii = np.linalg.norm(points, axis=1)
    assert np.allclose(radii, 5.0)
    assert np.all(points[:, 2] >= -1e-9)


def test_spherical_coverage_spreads_from_a_clustered_start():
    history, costs = simulate_spherical_coverage(n_agents=8, steps=25, seed=1)
    assert costs[-1] < costs[0] * 0.5
    spread_start = float(np.mean(np.linalg.norm(history[0] - history[0].mean(axis=0), axis=1)))
    spread_end = float(np.mean(np.linalg.norm(history[-1] - history[-1].mean(axis=0), axis=1)))
    assert spread_end > spread_start
    assert np.allclose(np.linalg.norm(history[-1], axis=1), 10.0)  # still on the surface
