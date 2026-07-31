"""The MAPF environment: API conformance and, above all, transition semantics.

The environment's job is to be a faithful MAPF simulator, because the whole
value of comparing a learned policy against CBS rests on both playing the same
game. So most of these tests are about the conflict rules -- vertex, edge, and
the cascade -- and about the property that follows from getting them right: a
rollout is conflict-free *by construction*, so ``Solution.is_valid()`` is true
no matter how bad the policy is.
"""

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pytest

np = pytest.importorskip("numpy")

import pymapf
from pymapf.rl import (
    DIAGONAL_ACTIONS,
    ORTHOGONAL_ACTIONS,
    GlobalGrid,
    LocalWindow,
    MAPFEnv,
    ShapedReward,
    SparseReward,
    available_observations,
    available_rewards,
    get_observation,
    get_reward,
    register_observation,
)
from pymapf.rl.observation import ENCODERS

STAY, UP, DOWN, LEFT, RIGHT = range(5)


def corridor_env(**kwargs):
    """Two agents facing each other in a 1-cell corridor -- a forced swap."""
    grid = pymapf.GridMap([[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]])
    problem = pymapf.MAPFProblem(
        grid,
        [pymapf.Agent("A", (1, 1), (1, 3)), pymapf.Agent("B", (1, 3), (1, 1))],
    )
    return MAPFEnv(problem, **kwargs)


def open_env(n_agents=2, **kwargs):
    kwargs.setdefault("randomise", False)
    return MAPFEnv("empty_room", height=8, width=8, n_agents=n_agents, **kwargs)


# --------------------------------------------------------------------------
# parallel API conformance
# --------------------------------------------------------------------------


def test_reset_returns_observations_and_infos_for_every_agent():
    env = open_env(3)
    observations, infos = env.reset(seed=0)
    assert set(observations) == set(env.possible_agents)
    assert set(infos) == set(env.possible_agents)
    assert env.agents == env.possible_agents


def test_step_returns_the_five_parallel_api_dicts():
    env = open_env(2)
    env.reset(seed=0)
    result = env.step({agent: STAY for agent in env.agents})
    assert len(result) == 5
    observations, rewards, terminations, truncations, infos = result
    for mapping in (observations, rewards, terminations, truncations, infos):
        assert set(mapping) == set(env.possible_agents)


def test_observations_match_the_declared_space():
    for encoder in available_observations():
        env = open_env(2, observation=encoder)
        observations, _ = env.reset(seed=0)
        for agent, observation in observations.items():
            space = env.observation_space(agent)
            assert observation.shape == space.shape
            assert space.contains(observation), encoder


def test_actions_outside_the_space_are_rejected():
    env = open_env(2)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step({agent: 99 for agent in env.agents})


def test_step_requires_an_action_for_every_agent():
    env = open_env(3)
    env.reset(seed=0)
    with pytest.raises(KeyError):
        env.step({env.agents[0]: STAY})


def test_the_agent_list_empties_when_the_episode_ends():
    """The parallel API's signal that there is nothing left to act on."""
    env = corridor_env(max_steps=4)
    env.reset(seed=0)
    for _ in range(4):
        if not env.agents:
            break
        env.step({agent: STAY for agent in env.agents})
    assert env.agents == []


def test_seeding_makes_episodes_reproducible():
    first = open_env(3, randomise=True)
    second = open_env(3, randomise=True)
    a, _ = first.reset(seed=7)
    b, _ = second.reset(seed=7)
    for agent in a:
        assert np.array_equal(a[agent], b[agent])


# --------------------------------------------------------------------------
# MAPF transition semantics -- the part that has to be exactly right
# --------------------------------------------------------------------------


def test_a_move_into_a_wall_is_refused():
    env = corridor_env()
    env.reset(seed=0)
    before = dict(env.positions)
    _, _, _, _, infos = env.step({"A": UP, "B": STAY})
    assert env.positions["A"] == before["A"]
    assert infos["A"]["blocked"]
    assert not infos["A"]["collided"]


def test_two_agents_claiming_one_cell_is_a_vertex_conflict():
    grid = pymapf.GridMap([[0] * 5 for _ in range(5)])
    problem = pymapf.MAPFProblem(
        grid, [pymapf.Agent("A", (2, 1), (2, 4)), pymapf.Agent("B", (2, 3), (2, 0))]
    )
    env = MAPFEnv(problem)
    env.reset(seed=0)
    # Both aim for (2, 2).
    _, _, _, _, infos = env.step({"A": RIGHT, "B": LEFT})
    assert env.positions["A"] == (2, 1)
    assert env.positions["B"] == (2, 3)
    assert infos["A"]["collided"] and infos["B"]["collided"]


def test_two_agents_exchanging_cells_is_an_edge_conflict():
    """The rule a naive "is the destination empty?" check misses."""
    env = corridor_env()
    env.reset(seed=0)
    env.positions = {"A": (1, 1), "B": (1, 2)}
    _, _, _, _, infos = env.step({"A": RIGHT, "B": LEFT})
    assert env.positions == {"A": (1, 1), "B": (1, 2)}
    assert infos["A"]["collided"] and infos["B"]["collided"]


def test_following_is_allowed():
    """A may enter the cell B is leaving -- that is not a conflict in MAPF."""
    grid = pymapf.GridMap([[1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1]])
    problem = pymapf.MAPFProblem(
        grid, [pymapf.Agent("A", (1, 1), (1, 4)), pymapf.Agent("B", (1, 2), (1, 4))]
    )
    env = MAPFEnv(problem)
    env.reset(seed=0)
    _, _, _, _, infos = env.step({"A": RIGHT, "B": RIGHT})
    assert env.positions["A"] == (1, 2)
    assert env.positions["B"] == (1, 3)
    assert not infos["A"]["collided"] and not infos["B"]["collided"]


def test_moving_into_a_stationary_agent_is_refused():
    grid = pymapf.GridMap([[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]])
    problem = pymapf.MAPFProblem(
        grid, [pymapf.Agent("A", (1, 1), (1, 3)), pymapf.Agent("B", (1, 2), (1, 2))]
    )
    env = MAPFEnv(problem)
    env.reset(seed=0)
    _, _, _, _, infos = env.step({"A": RIGHT, "B": STAY})
    assert env.positions["A"] == (1, 1)
    assert infos["A"]["collided"]


def test_refusals_cascade():
    """A blocked agent can invalidate the agent moving into the cell it was
    about to vacate, so resolution has to iterate rather than resolve once."""
    grid = pymapf.GridMap([[1] * 6, [1, 0, 0, 0, 0, 1], [1] * 6])
    problem = pymapf.MAPFProblem(
        grid,
        [
            pymapf.Agent("A", (1, 1), (1, 1)),
            pymapf.Agent("B", (1, 2), (1, 2)),
            pymapf.Agent("C", (1, 3), (1, 3)),
        ],
    )
    env = MAPFEnv(problem)
    env.reset(seed=0)
    # A walks into the wall; B follows A into (1,1); C follows B into (1,2).
    # A is refused, so B must be too -- and then C must be as well.
    _, _, _, _, infos = env.step({"A": LEFT, "B": LEFT, "C": LEFT})
    assert env.positions == {"A": (1, 1), "B": (1, 2), "C": (1, 3)}
    assert infos["A"]["blocked"]
    assert infos["B"]["collided"] and infos["C"]["collided"]


@pytest.mark.parametrize("n_agents", [2, 4, 6])
def test_any_rollout_is_a_conflict_free_plan(n_agents):
    """The property everything else rests on.

    Whatever actions are supplied -- including uniformly random ones -- the
    trajectory the environment produces is a *valid* MAPF plan, checked by the
    same conflict detector that validates CBS. A learned policy therefore can
    never score well by cheating on the physics.
    """
    env = MAPFEnv(
        "random_obstacles", height=10, width=10, n_agents=n_agents, density=0.15, seed=1
    )
    generator = np.random.default_rng(n_agents)
    for episode in range(6):
        env.reset(seed=episode)
        while env.agents:
            actions = {agent: int(generator.integers(5)) for agent in env.agents}
            env.step(actions)
        assert env.solution().is_valid()


def test_agents_never_stand_on_an_obstacle():
    env = MAPFEnv("maze", n_agents=3, seed=0)
    generator = np.random.default_rng(0)
    env.reset(seed=0)
    while env.agents:
        env.step({agent: int(generator.integers(5)) for agent in env.agents})
        for position in env.positions.values():
            assert env.grid.is_free(position)


def test_diagonal_actions_appear_only_when_the_problem_allows_them():
    assert len(ORTHOGONAL_ACTIONS) == 5
    assert len(DIAGONAL_ACTIONS) == 9
    env = open_env(2)
    assert env.action_space("A").n == 5


# --------------------------------------------------------------------------
# scoring on the planner's terms
# --------------------------------------------------------------------------


def test_a_solved_episode_matches_the_planner_cost_convention():
    """Trailing waits on the goal are trimmed, exactly as the planners' paths are."""
    grid = pymapf.GridMap([[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]])
    problem = pymapf.MAPFProblem(grid, [pymapf.Agent("A", (1, 1), (1, 3))])
    env = MAPFEnv(problem, max_steps=20)
    env.reset(seed=0)
    env.step({"A": RIGHT})
    env.step({"A": RIGHT})  # arrives at t=2
    for _ in range(5):
        if env.agents:
            env.step({"A": STAY})

    solution = env.solution()
    assert solution.sum_of_costs == 2  # not 7
    assert solution.paths["A"] == [(1, 1), (1, 2), (1, 3)]
    optimal = pymapf.solve(problem, "cbs")
    assert solution.sum_of_costs == optimal.sum_of_costs


def test_the_episode_ends_when_every_agent_is_on_its_goal():
    grid = pymapf.GridMap([[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]])
    problem = pymapf.MAPFProblem(
        grid, [pymapf.Agent("A", (1, 1), (1, 2)), pymapf.Agent("B", (1, 3), (1, 3))]
    )
    env = MAPFEnv(problem, max_steps=20)
    env.reset(seed=0)
    _, _, terminations, truncations, _ = env.step({"A": RIGHT, "B": STAY})
    assert all(terminations.values())
    assert not any(truncations.values())
    assert env.episode_summary()["solved"]


def test_an_unfinished_episode_truncates_rather_than_terminating():
    env = corridor_env(max_steps=3)
    env.reset(seed=0)
    for _ in range(3):
        _, _, terminations, truncations, _ = env.step(
            {agent: STAY for agent in env.agents}
        )
    assert not any(terminations.values())
    assert all(truncations.values())
    assert env.episode_summary()["solved"] is False


def test_state_is_the_centralized_view_mappo_needs():
    env = open_env(3)
    env.reset(seed=0)
    state = env.state()
    assert state.shape == (env.state_size,)
    assert np.all(state >= 0) and np.all(state <= 1)
    # It moves when the world does.
    before = state.copy()
    env.step({agent: RIGHT for agent in env.agents})
    assert not np.array_equal(before, env.state())


# --------------------------------------------------------------------------
# observations and rewards
# --------------------------------------------------------------------------


def test_the_local_window_marks_walls_and_the_map_edge():
    env = corridor_env(observation="local", observation_kwargs={"radius": 1})
    env.reset(seed=0)
    encoder = env.encoder
    observation = encoder.encode(env, "A")
    obstacles = observation[: encoder.size**2].reshape(encoder.size, encoder.size)
    # A sits at (1,1) in a 3x5 grid: everything above and below is wall.
    assert obstacles[0, 1] == 1.0
    assert obstacles[2, 1] == 1.0
    assert obstacles[1, 1] == 0.0


def test_the_local_window_is_the_same_size_whatever_the_team_size():
    small = MAPFEnv("empty_room", n_agents=2, height=9, width=9, randomise=False)
    large = MAPFEnv("empty_room", n_agents=6, height=9, width=9, randomise=False)
    assert small.observation_space("A").shape == large.observation_space("A").shape


def test_the_goal_vector_points_at_the_goal():
    env = corridor_env(observation="local", observation_kwargs={"radius": 1})
    env.reset(seed=0)
    observation = env.encoder.encode(env, "A")
    bearing = observation[-3:-1]
    # A is at (1,1) heading for (1,3): due "right", i.e. +column.
    assert bearing[0] == pytest.approx(0.0, abs=1e-6)
    assert bearing[1] == pytest.approx(1.0, abs=1e-6)


def test_unknown_observations_and_rewards_are_rejected():
    with pytest.raises(ValueError):
        get_observation("telepathy")
    with pytest.raises(ValueError):
        get_reward("vibes")


def test_encoders_can_be_registered_from_outside():
    @register_observation("test_tiny")
    class Tiny(LocalWindow):
        name = "test_tiny"

    try:
        assert "test_tiny" in available_observations()
        assert isinstance(get_observation("test_tiny"), Tiny)
    finally:
        ENCODERS.pop("test_tiny", None)


def test_reaching_the_goal_pays_and_collisions_cost():
    reward = SparseReward(step=-0.1, goal=1.0, collision=-0.5, blocked=-0.2)
    env = open_env(2, reward=reward)
    env.reset(seed=0)
    arrived = reward.compute(env, "A", (1, 1), (1, 2), False, False, True, False)
    collided = reward.compute(env, "A", (1, 1), (1, 1), False, True, False, False)
    assert arrived > 0
    assert collided < 0


def test_an_agent_parked_on_its_goal_is_not_charged():
    """Charging it would teach the policy to wander off the goal."""
    reward = SparseReward(step=-0.1, goal=1.0)
    env = open_env(2, reward=reward)
    env.reset(seed=0)
    parked = reward.compute(env, "A", (1, 1), (1, 1), False, False, True, True)
    assert parked == pytest.approx(0.0)


def test_shaping_uses_the_exact_distance_not_a_straight_line():
    """The potential must route around walls, which is the whole point of
    having a real distance oracle rather than a Manhattan guess."""
    grid = pymapf.GridMap(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    problem = pymapf.MAPFProblem(grid, [pymapf.Agent("A", (1, 1), (1, 3))])
    reward = ShapedReward()
    env = MAPFEnv(problem, reward=reward)
    env.reset(seed=0)
    # Manhattan says 2; the only route is down, across and back up, so 6.
    assert reward.potential("A", (1, 1)) == pytest.approx(-6.0)
    assert reward.potential("A", (1, 3)) == pytest.approx(0.0)


def test_shaping_is_potential_based_so_a_round_trip_nets_out():
    """Ng et al. (1999): the shaping of a cycle telescopes to (gamma^k - 1) * Phi,
    which is zero when gamma is 1. That is what makes it policy-invariant."""
    grid = pymapf.GridMap([[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]])
    problem = pymapf.MAPFProblem(grid, [pymapf.Agent("A", (1, 1), (1, 3))])
    reward = ShapedReward(gamma=1.0, step=0.0, goal=0.0, collision=0.0, blocked=0.0)
    env = MAPFEnv(problem, reward=reward)
    env.reset(seed=0)

    out = reward.compute(env, "A", (1, 1), (1, 2), False, False, False, False)
    back = reward.compute(env, "A", (1, 2), (1, 1), False, False, False, False)
    assert out + back == pytest.approx(0.0, abs=1e-9)


def test_unreachable_cells_do_not_produce_infinities():
    grid = pymapf.GridMap([[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]])
    problem = pymapf.MAPFProblem(grid, [pymapf.Agent("A", (1, 1), (1, 1))])
    reward = ShapedReward()
    env = MAPFEnv(problem, reward=reward)
    env.reset(seed=0)
    assert np.isfinite(reward.potential("A", (1, 3)))


def test_randomising_draws_a_fresh_instance_each_reset():
    env = MAPFEnv("random_obstacles", height=10, width=10, n_agents=3, randomise=True)
    env.reset(seed=1)
    first = dict(env.goals)
    for _ in range(10):
        env.reset()
        if dict(env.goals) != first:
            return
    pytest.fail("randomise=True never produced a different instance")


def test_randomise_needs_a_family_to_draw_from():
    problem = pymapf.build_scenario("empty_room", n_agents=2).to_problem()
    with pytest.raises(ValueError):
        MAPFEnv(problem, randomise=True)


def test_render_shows_agents_walls_and_goals():
    env = corridor_env()
    env.reset(seed=0)
    frame = env.render(mode="ansi")
    assert "#" in frame and "A" in frame and "B" in frame
    assert len(frame.splitlines()) == env.grid.height


# --------------------------------------------------------------------------
# respawn, and the shared-state bug it has to avoid
# --------------------------------------------------------------------------


def test_respawn_preserves_the_configuration():
    env = MAPFEnv(
        "random_obstacles",
        height=9,
        width=9,
        n_agents=3,
        density=0.2,
        observation_kwargs={"radius": 2},
        max_steps=37,
    )
    copy = env.respawn()
    assert copy.encoder.radius == 2
    assert copy.max_steps == 37
    assert copy.randomise == env.randomise
    assert copy.observation_space("A").shape == env.observation_space("A").shape


@pytest.mark.parametrize("reward", ["shaped", ShapedReward(scale=2.0)])
def test_respawn_never_shares_a_reward_function(reward):
    """Sharing one would be a correctness bug, not an efficiency one.

    ShapedReward caches a distance field per instance. Vectorised workers reset
    at different times, so a shared reward function means one worker's reset
    silently overwrites the potentials another is midway through using.
    """
    env = MAPFEnv("random_obstacles", height=9, width=9, n_agents=2, reward=reward)
    first, second = env.respawn(), env.respawn()
    assert first.reward_function is not second.reward_function
    assert first.reward_function is not env.reward_function
    assert first.encoder is not second.encoder


def test_every_vector_worker_gets_its_own_reward_function():
    from pymapf.rl import VectorMAPFEnv

    template = MAPFEnv(
        "random_obstacles",
        height=10,
        width=10,
        n_agents=2,
        density=0.15,
        reward="shaped",
    )
    vector = VectorMAPFEnv(template.respawn, n=4, seed=0)
    identities = {id(worker.reward_function) for worker in vector.envs}
    assert len(identities) == 4


def test_the_problem_is_reachable_without_touching_privates():
    env = MAPFEnv("empty_room", height=8, width=8, n_agents=2, randomise=False)
    env.reset(seed=0)
    optimal = pymapf.solve(env.problem, "cbs")
    assert optimal is not None and optimal.is_valid()
