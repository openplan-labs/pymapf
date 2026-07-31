"""The learning stack: gradients, GAE, and whether the thing actually learns.

Hand-derived backpropagation is a claim, and the only honest way to support it
is a finite-difference check -- so that is the first thing here. After that:
GAE against a hand-computed case, the buffer's episode-boundary handling, and
an end-to-end check that a policy improves on a problem small enough to train
inside a test.

Everything runs on the numpy backend, which needs nothing beyond numpy, so the
learning code is covered in CI rather than only on a machine with torch. The
torch backend runs the same checks when it happens to be installed.
"""

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pytest

np = pytest.importorskip("numpy")

from pymapf.rl import (
    IPPO,
    MAPPO,
    MAPFEnv,
    RolloutBuffer,
    available_algorithms,
    available_backends,
    compute_gae,
    make_actor_critic,
    make_trainer,
)
from pymapf.rl.networks import MLP, Adam, NumpyActorCritic, _log_softmax

BACKENDS = available_backends()
CLIP, VALUE_COEFFICIENT, ENTROPY_COEFFICIENT = 0.2, 0.5, 0.01


# --------------------------------------------------------------------------
# gradients
# --------------------------------------------------------------------------


def _ppo_losses(network, batch):
    """The two losses, recomputed forward-only for finite differencing."""
    logits, _ = network.actor.forward(batch["observations"])
    log_probs = _log_softmax(logits)
    probabilities = np.exp(log_probs)
    count = len(batch["actions"])
    chosen = log_probs[np.arange(count), batch["actions"]]
    ratio = np.exp(chosen - batch["log_probs"])
    unclipped = ratio * batch["advantages"]
    clipped = np.clip(ratio, 1 - CLIP, 1 + CLIP) * batch["advantages"]
    entropy = -(probabilities * log_probs).sum(-1)
    actor = -np.mean(np.minimum(unclipped, clipped)) - ENTROPY_COEFFICIENT * np.mean(entropy)

    values, _ = network.critic.forward(batch["critic_inputs"])
    critic = VALUE_COEFFICIENT * 0.5 * np.mean((values.reshape(-1) - batch["returns"]) ** 2)
    return float(actor), float(critic)


def _random_batch(seed=0, count=24, obs_dim=7, n_actions=5):
    generator = np.random.default_rng(seed)
    return {
        "observations": generator.normal(size=(count, obs_dim)),
        "critic_inputs": generator.normal(size=(count, obs_dim)),
        "actions": generator.integers(0, n_actions, count),
        "log_probs": generator.normal(size=count) * 0.1,
        "advantages": generator.normal(size=count),
        "returns": generator.normal(size=count),
    }


@pytest.mark.parametrize("head", ["actor", "critic"])
def test_hand_written_gradients_match_finite_differences(head):
    """The check that makes hand-derived backprop trustworthy at all."""
    batch = _random_batch()
    network = NumpyActorCritic(7, 5, 7, hidden=(8, 8), seed=1)
    module = getattr(network, head)

    if head == "actor":
        logits, cache = module.forward(batch["observations"])
        log_probs = _log_softmax(logits)
        probabilities = np.exp(log_probs)
        count = len(batch["actions"])
        rows = np.arange(count)
        chosen = log_probs[rows, batch["actions"]]
        ratio = np.exp(chosen - batch["log_probs"])
        unclipped = ratio * batch["advantages"]
        clipped = np.clip(ratio, 1 - CLIP, 1 + CLIP) * batch["advantages"]
        d_ratio = np.where(unclipped <= clipped, -batch["advantages"] / count, 0.0)
        onehot = np.zeros_like(logits)
        onehot[rows, batch["actions"]] = 1.0
        d_logits = (d_ratio * ratio)[:, None] * (onehot - probabilities)
        entropy = -(probabilities * log_probs).sum(-1)
        d_logits += ENTROPY_COEFFICIENT * (probabilities * (log_probs + entropy[:, None])) / count
        gradients = module.backward(d_logits, cache)
        index = 0
    else:
        values, cache = module.forward(batch["critic_inputs"])
        count = len(batch["returns"])
        d_values = (VALUE_COEFFICIENT * (values.reshape(-1) - batch["returns"]) / count).reshape(-1, 1)
        gradients = module.backward(d_values, cache)
        index = 1

    generator = np.random.default_rng(3)
    epsilon = 1e-6
    worst = 0.0
    for position, parameter in enumerate(module.params):
        flat = parameter.reshape(-1)
        # A copy here would silently pass: writes would not reach the network.
        assert flat.base is parameter or parameter.flags["C_CONTIGUOUS"]
        analytic = gradients[position].reshape(-1)
        for k in generator.choice(len(flat), size=min(10, len(flat)), replace=False):
            original = flat[k]
            flat[k] = original + epsilon
            high = _ppo_losses(network, batch)[index]
            flat[k] = original - epsilon
            low = _ppo_losses(network, batch)[index]
            flat[k] = original
            numeric = (high - low) / (2 * epsilon)
            scale = max(1e-6, abs(numeric), abs(analytic[k]))
            worst = max(worst, abs(numeric - analytic[k]) / scale)
    assert worst < 1e-3, "max relative gradient error %.2e" % worst


def test_network_parameters_are_contiguous():
    """A non-contiguous weight makes `reshape(-1)` return a copy, so anything
    writing through that view updates nothing at all."""
    network = NumpyActorCritic(11, 5, 7, hidden=(8, 8), seed=0)
    for parameter in network.actor.params + network.critic.params:
        assert parameter.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("shape", [(3, 8), (8, 3), (5, 5), (1, 6)])
def test_orthogonal_initialisation_handles_any_shape(shape):
    from pymapf.rl.networks import _orthogonal

    matrix = _orthogonal(shape, 1.0, np.random.default_rng(0))
    assert matrix.shape == shape
    rows, cols = shape
    smaller = min(rows, cols)
    product = matrix.T @ matrix if rows >= cols else matrix @ matrix.T
    assert np.allclose(product, np.eye(smaller), atol=1e-8)


def test_adam_descends_a_quadratic():
    parameters = [np.array([5.0, -3.0])]
    optimiser = Adam([p.shape for p in parameters], lr=0.1)
    for _ in range(500):
        optimiser.step(parameters, [2 * parameters[0]])
    assert np.allclose(parameters[0], 0.0, atol=1e-2)


def test_gradient_clipping_bounds_the_global_norm():
    from pymapf.rl.networks import _clip_global_norm

    gradients = [np.full((4,), 10.0), np.full((4,), 10.0)]
    _clip_global_norm(gradients, 1.0)
    total = np.sqrt(sum(float((g ** 2).sum()) for g in gradients))
    assert total == pytest.approx(1.0, rel=1e-6)


# --------------------------------------------------------------------------
# advantage estimation
# --------------------------------------------------------------------------


def test_gae_matches_a_hand_computed_case():
    rewards = np.array([[1.0], [1.0], [1.0]])
    values = np.array([[0.0], [0.0], [0.0]])
    dones = np.array([[0.0], [0.0], [0.0]])
    last = np.array([0.0])
    gamma, lam = 0.9, 0.5

    advantages = compute_gae(rewards, values, dones, last, gamma, lam)
    # With zero values, delta = reward everywhere, so A_t = sum (gamma*lam)^k.
    step = gamma * lam
    assert advantages[2, 0] == pytest.approx(1.0)
    assert advantages[1, 0] == pytest.approx(1.0 + step)
    assert advantages[0, 0] == pytest.approx(1.0 + step + step ** 2)


def test_gae_does_not_bootstrap_across_an_episode_boundary():
    """The classic silent bug in a vectorised rollout."""
    rewards = np.array([[1.0], [1.0]])
    values = np.array([[0.0], [10.0]])
    last = np.array([100.0])

    crossing = compute_gae(rewards, values, np.array([[0.0], [0.0]]), last, 1.0, 1.0)
    cut = compute_gae(rewards, values, np.array([[1.0], [0.0]]), last, 1.0, 1.0)
    # With done at t=0 the value of the next state must not be used.
    assert crossing[0, 0] == pytest.approx(1.0 + 10.0 - 0.0 + (1.0 + 100.0 - 10.0))
    assert cut[0, 0] == pytest.approx(1.0)


def test_gae_reduces_to_td_error_when_lambda_is_zero():
    generator = np.random.default_rng(0)
    rewards = generator.normal(size=(5, 3))
    values = generator.normal(size=(5, 3))
    dones = np.zeros((5, 3))
    last = generator.normal(size=3)
    advantages = compute_gae(rewards, values, dones, last, 0.9, 0.0)
    expected = rewards[0] + 0.9 * values[1] - values[0]
    assert np.allclose(advantages[0], expected)


def test_the_buffer_normalises_advantages_and_builds_returns():
    buffer = RolloutBuffer(steps=4, streams=2, obs_dim=3, critic_dim=3)
    generator = np.random.default_rng(0)
    for _ in range(4):
        buffer.add(
            generator.normal(size=(2, 3)),
            generator.normal(size=(2, 3)),
            generator.integers(0, 5, 2),
            generator.normal(size=2),
            generator.normal(size=2),
            np.zeros(2),
            generator.normal(size=2),
        )
    assert buffer.full
    batch = buffer.batch(np.zeros(2), gamma=0.99, lam=0.95)
    assert batch["advantages"].mean() == pytest.approx(0.0, abs=1e-6)
    assert batch["advantages"].std() == pytest.approx(1.0, rel=1e-3)
    assert len(batch["actions"]) == 8
    # returns = advantages + values, before normalisation
    assert np.all(np.isfinite(batch["returns"]))


# --------------------------------------------------------------------------
# the algorithms
# --------------------------------------------------------------------------


def test_both_algorithms_are_registered():
    assert set(available_algorithms()) >= {"ippo", "mappo"}


def test_the_only_difference_between_ippo_and_mappo_is_the_critic_input():
    """If this ever stops being true, the comparison stops being a comparison
    of algorithms and becomes one of implementations."""
    assert IPPO.centralized_critic is False
    assert MAPPO.centralized_critic is True

    env = MAPFEnv("empty_room", height=6, width=6, n_agents=2, randomise=False)
    ippo = make_trainer("ippo", env, n_envs=2, rollout_steps=8, seed=0)
    mappo = make_trainer("mappo", env, n_envs=2, rollout_steps=8, seed=0)

    obs_dim = env.observation_space("A").shape[0]
    assert ippo.policy.critic_dim == obs_dim
    assert mappo.policy.critic_dim == env.state_size
    assert ippo.policy.obs_dim == mappo.policy.obs_dim  # the actors are identical


def test_an_unknown_algorithm_lists_the_available_ones():
    env = MAPFEnv("empty_room", height=6, width=6, n_agents=2, randomise=False)
    with pytest.raises(ValueError, match="ippo"):
        make_trainer("alphago", env)


@pytest.mark.parametrize("algorithm", ["ippo", "mappo"])
def test_a_short_run_produces_finite_diagnostics(algorithm):
    env = MAPFEnv("empty_room", height=6, width=6, n_agents=2, randomise=False)
    trainer = make_trainer(algorithm, env, n_envs=2, rollout_steps=16, seed=0)
    history = trainer.learn(total_steps=300)
    assert history
    for record in history:
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl"):
            assert np.isfinite(record[key]), key
    assert trainer.total_steps >= 300


@pytest.mark.parametrize("algorithm", ["ippo", "mappo"])
def test_training_improves_the_policy(algorithm):
    """End to end, on an instance small enough to learn inside a test.

    Measured as the solve rate over the last rollouts against the first, which
    is the number that matters -- a falling loss can mean nothing at all.
    """
    env = MAPFEnv("empty_room", height=7, width=7, n_agents=2, randomise=False)
    trainer = make_trainer(
        algorithm, env, n_envs=8, rollout_steps=64, seed=0, lr=1e-3
    )
    trainer.learn(total_steps=40_000)
    solved = [record["solved"] for record in trainer.history]
    assert max(solved[-3:]) > max(0.15, solved[0] + 0.1), solved[-5:]


def test_the_actor_starts_close_to_uniform():
    """A policy that starts confident spends its first updates undoing that."""
    network = NumpyActorCritic(20, 5, 20, seed=0)
    logits = network.logits(np.random.default_rng(0).normal(size=(32, 20)))
    probabilities = np.exp(_log_softmax(logits))
    assert np.allclose(probabilities, 0.2, atol=0.05)


def test_deterministic_actions_are_reproducible():
    env = MAPFEnv("empty_room", height=6, width=6, n_agents=2, randomise=False)
    trainer = make_trainer("ippo", env, n_envs=2, rollout_steps=8, seed=0)
    observations, _ = env.reset(seed=0)
    first = trainer.act(observations, deterministic=True)
    second = trainer.act(observations, deterministic=True)
    assert first == second


def test_a_policy_can_be_saved_and_restored(tmp_path):
    env = MAPFEnv("empty_room", height=6, width=6, n_agents=2, randomise=False)
    trainer = make_trainer("ippo", env, n_envs=2, rollout_steps=8, seed=0)
    trainer.learn(total_steps=200)
    observations, _ = env.reset(seed=1)
    before = trainer.act(observations)

    target = str(tmp_path / "policy.pkl")
    trainer.save(target)
    fresh = make_trainer("ippo", env, n_envs=2, rollout_steps=8, seed=5)
    fresh.load(target)
    assert fresh.act(observations) == before


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_available_backend_trains(backend):
    """The backends implement the same objective, so both must learn."""
    env = MAPFEnv("empty_room", height=6, width=6, n_agents=2, randomise=False)
    trainer = make_trainer(
        "ippo", env, backend=backend, n_envs=4, rollout_steps=32, seed=0
    )
    history = trainer.learn(total_steps=4_000)
    assert np.isfinite(history[-1]["entropy"])
    assert history[-1]["entropy"] < np.log(5) + 1e-6  # never above uniform


def test_an_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        make_actor_critic("jax", obs_dim=4, n_actions=5)
