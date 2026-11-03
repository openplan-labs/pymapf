"""Lifelong MAPF: agents are re-tasked on arrival and the objective is throughput."""

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pytest

np = pytest.importorskip("numpy")

import pymapf  # noqa: E402
from pymapf.rl import MAPFEnv, make_trainer  # noqa: E402
from pymapf.rl.baselines import PIBTPolicy, ReplanPolicy, RandomPolicy  # noqa: E402
from pymapf.rl.evaluate import compare_lifelong, rollout  # noqa: E402
from pymapf.rl.wrappers import VectorMAPFEnv  # noqa: E402


def lifelong_env(**kwargs):
    defaults = dict(height=7, width=7, n_agents=3, seed=0, randomise=False)
    defaults.update(kwargs)
    return MAPFEnv("empty_room", lifelong=True, **defaults)


# --------------------------------------------------------------------------
# the environment
# --------------------------------------------------------------------------


def test_lifelong_is_off_by_default_and_declared_when_on():
    assert MAPFEnv("empty_room", n_agents=2, randomise=False).lifelong is False
    env = lifelong_env()
    assert env.lifelong is True
    assert "lifelong" in repr(env)


def test_reaching_a_goal_hands_out_a_new_one_and_counts():
    env = lifelong_env()
    env.reset(seed=1)
    agent = env.possible_agents[0]
    first_goal = env.goals[agent]
    policy = PIBTPolicy(env)
    observations, _ = env.reset(seed=1)
    reached = False
    for _ in range(200):
        observations, rewards, terminations, truncations, infos = env.step(
            policy.act(observations)
        )
        if infos[agent]["at_goal"]:
            reached = True
            # The goal has already moved on by the time the info is returned.
            assert env.goals[agent] != first_goal
            assert env.goals_completed >= 1
            assert rewards[agent] > 0  # the arrival bonus was paid for the goal reached
            break
        assert not any(terminations.values())
    assert reached


def test_new_goals_are_free_reachable_and_not_someone_elses():
    """At the moment a goal is handed out it is free, reachable, nobody's
    current position and nobody's goal. Other agents may pass through it
    later -- that is MAPF's rule, only the owner parks -- so occupancy is
    checked at assignment, uniqueness always."""
    env = lifelong_env(n_agents=4)
    observations, _ = env.reset(seed=3)
    policy = PIBTPolicy(env)
    assignments = 0
    for _ in range(300):
        observations, _, _, truncations, infos = env.step(policy.act(observations))
        goals = list(env.goals.values())
        assert len(set(goals)) == len(goals), "two agents were handed the same goal"
        for agent, goal in env.goals.items():
            assert env.grid.is_free(goal)
            assert goal in env._component(env.positions[agent])
            if infos[agent]["at_goal"]:  # a fresh goal was just assigned
                assignments += 1
                assert all(
                    env.positions[other] != goal
                    for other in env.possible_agents
                    if other != agent
                )
        if any(truncations.values()):
            break
    assert assignments >= 4


def test_a_lifelong_episode_ends_only_at_the_horizon():
    env = lifelong_env(max_steps=40)
    observations, _ = env.reset(seed=0)
    policy = PIBTPolicy(env)
    for step in range(1, 41):
        observations, _, terminations, truncations, infos = env.step(
            policy.act(observations)
        )
        assert not any(terminations.values())
        if step < 40:
            assert not any(truncations.values())
    assert all(truncations.values())
    summary = infos[env.possible_agents[0]]["episode"]
    assert summary["solved"] is False
    assert summary["steps"] == 40
    assert summary["goals_completed"] == env.goals_completed
    assert summary["throughput"] == pytest.approx(env.goals_completed / 40)
    assert env.agents == []


def test_shaped_reward_follows_the_new_goal():
    """The potential must be the distance to the *current* goal, not the
    one the episode started with; otherwise every reassignment teaches the
    agent to walk back."""
    env = lifelong_env(reward="shaped", max_steps=300)
    observations, _ = env.reset(seed=2)
    policy = PIBTPolicy(env)
    agent = env.possible_agents[0]
    switched = False
    for _ in range(300):
        goal_before = env.goals[agent]
        observations, rewards, _, truncations, infos = env.step(
            policy.act(observations)
        )
        if infos[agent]["at_goal"]:
            switched = True
            # After the switch, a PIBT step toward the new goal is rewarded
            # above the plain step cost: shaping saw the new distance field.
            observations, rewards, _, _, infos = env.step(policy.act(observations))
            if (
                not infos[agent]["at_goal"]
                and not infos[agent]["blocked"]
                and not infos[agent]["collided"]
            ):
                assert rewards[agent] > env.reward_function.step
            break
        if any(truncations.values()):
            break
    assert switched


def test_the_trajectory_is_still_a_valid_plan():
    env = lifelong_env(max_steps=120)
    observations, _ = env.reset(seed=5)
    policy = PIBTPolicy(env)
    while env.agents:
        observations, _, _, _, _ = env.step(policy.act(observations))
    solution = env.solution()
    assert solution.is_valid()
    assert all(len(path) >= 1 for path in solution.paths.values())


def test_vectorised_lifelong_environments_reset_at_the_horizon():
    # respawn() must carry the lifelong flag across, or the workers are one-shot.
    template = lifelong_env(max_steps=25)
    vector = VectorMAPFEnv(template.respawn, n=3, seed=0)
    assert all(env.lifelong for env in vector.envs)
    observations = vector.reset()
    saw_final = 0
    for _ in range(60):
        actions = [{agent: 0 for agent in obs} for obs in observations]
        observations, _, _, _, infos = vector.step(actions)
        for info in infos:
            for entry in info.values():
                final = entry.get("final_info")
                if final and "episode" in final:
                    assert "throughput" in final["episode"]
                    saw_final += 1
                break
    assert saw_final >= 3


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------


def test_pibt_baseline_completes_goals_and_never_collides():
    env = lifelong_env(n_agents=4, max_steps=200)
    _, summary = rollout(env, PIBTPolicy(env), seed=7)
    assert summary["goals_completed"] >= 8
    assert summary["collisions"] == 0
    assert env.solution().is_valid()


def test_replanning_baseline_completes_goals():
    env = lifelong_env(n_agents=3, max_steps=150)
    _, summary = rollout(env, ReplanPolicy(env, "lacam", time_limit=2.0), seed=7)
    assert summary["goals_completed"] >= 5
    assert env.solution().is_valid()


def test_random_policy_is_the_floor():
    env = lifelong_env(n_agents=3, max_steps=200)
    _, random_summary = rollout(env, RandomPolicy(env), seed=7)
    _, pibt_summary = rollout(env, PIBTPolicy(env), seed=7)
    assert pibt_summary["throughput"] > 3 * max(random_summary["throughput"], 1e-9)


# --------------------------------------------------------------------------
# comparison and training
# --------------------------------------------------------------------------


def test_compare_lifelong_reports_throughput_on_shared_instances():
    env = lifelong_env(n_agents=3, max_steps=80)
    rows = compare_lifelong(
        env,
        {"random": RandomPolicy(env)},
        episodes=3,
        baselines=("pibt",),
        seed=11,
        modes=("greedy",),
    )
    names = [row["method"] for row in rows]
    assert "random" in names and "pibt" in names
    by_name = {row["method"]: row for row in rows}
    for row in rows:
        assert row["episodes"] == 3
        assert row["throughput"] >= 0
        assert "goals_per_agent_per_100_steps" in row
        assert row["validity_rate"] == 1.0
    assert by_name["pibt"]["throughput"] > by_name["random"]["throughput"]


def test_ippo_trains_on_a_lifelong_environment():
    env = lifelong_env(n_agents=2, max_steps=40)
    trainer = make_trainer("ippo", env, n_envs=2, rollout_steps=16, seed=0)
    trainer.learn(total_steps=400)
    assert trainer.history
    assert "throughput" in trainer.history[-1]
    assert trainer.history[-1]["throughput"] >= 0.0


def test_one_shot_summaries_are_unchanged():
    env = MAPFEnv("empty_room", height=6, width=6, n_agents=2, randomise=False)
    _, summary = rollout(env, PIBTPolicy(env), seed=1)
    assert "throughput" not in summary
    assert set(summary) >= {"solved", "steps", "collisions", "sum_of_costs", "makespan"}


def test_an_agent_with_nowhere_new_to_go_is_not_counted_again_every_step():
    """With every free cell taken, `_new_goal` keeps the current goal. Staying
    on it is not another completed task: this used to report 102 completions
    for four real arrivals."""
    STAY, UP, DOWN, LEFT, RIGHT = range(5)
    grid = pymapf.GridMap([[0, 0, 0], [0, 0, 0]])
    problem = pymapf.MAPFProblem(
        grid,
        [
            pymapf.Agent("A", (0, 0), (0, 1)),
            pymapf.Agent("B", (0, 1), (0, 2)),
            pymapf.Agent("C", (1, 2), (1, 1)),
            pymapf.Agent("D", (1, 1), (1, 0)),
        ],
    )
    env = MAPFEnv(problem, lifelong=True, max_steps=50)
    env.reset(seed=0)
    env.step({"A": RIGHT, "B": RIGHT, "C": LEFT, "D": LEFT})
    arrivals = env.goals_completed
    assert arrivals == 4
    while env.agents:
        env.step({agent: STAY for agent in env.agents})
    # Some agents may have been re-tasked onto a cell another one vacated, but
    # nobody moved, so nobody can have arrived anywhere new.
    assert env.goals_completed == arrivals
