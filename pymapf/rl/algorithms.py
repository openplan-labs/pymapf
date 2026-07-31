"""IPPO and MAPPO, which differ in exactly one thing.

Both are PPO with parameters shared across agents. Both collect the same
rollouts, compute the same advantages, and take the same clipped update. The
only difference is **what the critic is allowed to look at**:

``ippo``
    Independent PPO (de Witt et al. 2020). The critic sees the same local
    observation the actor does. Every agent is, formally, solving a
    single-agent problem in a world that happens to move on its own -- which is
    a lie, because the other agents' policies change under it. IPPO is the
    demonstration that the lie often does not matter.

``mappo``
    Multi-Agent PPO (Yu et al. 2022). The critic sees the *global state*:
    every agent's position and goal. Centralized training, decentralized
    execution -- at run time the policy still only sees its own window, so the
    learned controller is just as deployable, but the value estimates it was
    trained against were not fighting the same non-stationarity.

Because that is the entire difference, it is the entire difference in the code
too: :class:`MAPPO` sets ``centralized_critic = True`` and nothing else. Anyone
claiming a larger gap between the two is describing an implementation
difference, not an algorithmic one -- which is the finding of the MAPPO paper
and the reason the two share a trainer here.

References
----------
* Schulman, J.; Wolski, F.; Dhariwal, P.; Radford, A.; and Klimov, O. 2017.
  *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
* Schulman, J.; Moritz, P.; Levine, S.; Jordan, M.; and Abbeel, P. 2016.
  *High-Dimensional Continuous Control Using Generalized Advantage
  Estimation.* ICLR 2016.
* de Witt, C. S.; Gupta, T.; Makoviichuk, D.; Makoviychuk, V.; Torr, P. H. S.;
  Sun, M.; and Whiteson, S. 2020. *Is Independent Learning All You Need in the
  StarCraft Multi-Agent Challenge?* arXiv:2011.09533.
* Yu, C.; Velu, A.; Vinitsky, E.; Gao, J.; Wang, Y.; Bayen, A.; and Wu, Y.
  2022. *The Surprising Effectiveness of PPO in Cooperative Multi-Agent
  Games.* NeurIPS 2022 Datasets and Benchmarks.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence, Type

import numpy as np

from .env import MAPFEnv
from .networks import ActorCritic, make_actor_critic
from .wrappers import VectorMAPFEnv

__all__ = [
    "compute_gae",
    "RolloutBuffer",
    "PPOTrainer",
    "IPPO",
    "MAPPO",
    "make_trainer",
    "available_algorithms",
    "register_algorithm",
]


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> np.ndarray:
    """Generalized Advantage Estimation over ``[T, N]`` streams.

    ``dones[t]`` marks that the episode ended *at* step ``t``, which cuts the
    backward recursion there: bootstrapping across an episode boundary is the
    classic silent bug in a vectorised rollout, and it shows up as a value
    function that looks fine and a policy that never improves.
    """
    steps = len(rewards)
    advantages = np.zeros_like(rewards)
    running = np.zeros_like(last_values)
    for t in reversed(range(steps)):
        if t == steps - 1:
            next_values = last_values
        else:
            next_values = values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * not_done - values[t]
        running = delta + gamma * lam * not_done * running
        advantages[t] = running
    return advantages


class RolloutBuffer:
    """Flat storage for one PPO rollout, shaped ``[T, envs * agents]``.

    Agents are stacked into the batch dimension rather than kept in their own
    axis, which is what "parameter sharing" means concretely: one network sees
    every agent's experience, so a four-agent instance yields four times the
    data per environment step and the learned policy is symmetric in the agents
    by construction.
    """

    def __init__(self, steps: int, streams: int, obs_dim: int, critic_dim: int):
        self.steps, self.streams = steps, streams
        self.observations = np.zeros((steps, streams, obs_dim), dtype=np.float32)
        self.critic_inputs = np.zeros((steps, streams, critic_dim), dtype=np.float32)
        self.actions = np.zeros((steps, streams), dtype=np.int64)
        self.log_probs = np.zeros((steps, streams), dtype=np.float32)
        self.rewards = np.zeros((steps, streams), dtype=np.float32)
        self.dones = np.zeros((steps, streams), dtype=np.float32)
        self.values = np.zeros((steps, streams), dtype=np.float32)
        self._cursor = 0

    def add(
        self, observations, critic_inputs, actions, log_probs, rewards, dones, values
    ):
        t = self._cursor
        self.observations[t] = observations
        self.critic_inputs[t] = critic_inputs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.values[t] = values
        self._cursor += 1

    @property
    def full(self) -> bool:
        return self._cursor >= self.steps

    def reset(self) -> None:
        self._cursor = 0

    def batch(self, last_values, gamma: float, lam: float, normalise: bool = True):
        advantages = compute_gae(
            self.rewards, self.values, self.dones, last_values, gamma, lam
        )
        returns = advantages + self.values
        flat = {
            "observations": self.observations.reshape(-1, self.observations.shape[-1]),
            "critic_inputs": self.critic_inputs.reshape(
                -1, self.critic_inputs.shape[-1]
            ),
            "actions": self.actions.reshape(-1),
            "log_probs": self.log_probs.reshape(-1),
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
        }
        if normalise:
            values = flat["advantages"]
            flat["advantages"] = (values - values.mean()) / (values.std() + 1e-8)
        return flat


class PPOTrainer:
    """Shared-parameter PPO over a vectorised set of MAPF environments.

    Args:
        env: a template environment. It is copied ``n_envs`` times; the
            template itself is left alone so it stays usable for evaluation.
        backend: ``"numpy"`` (default, no dependencies) or ``"torch"``.
        n_envs, rollout_steps: the rollout is
            ``rollout_steps * n_envs * n_agents`` transitions.
        epochs, minibatches: PPO's inner loop over that rollout.
    """

    #: Whether the critic sees the global state. The one bit that separates
    #: MAPPO from IPPO.
    centralized_critic = False
    name = "ppo"

    def __init__(
        self,
        env: MAPFEnv,
        backend: str = "numpy",
        n_envs: int = 8,
        rollout_steps: int = 128,
        epochs: int = 4,
        minibatches: int = 4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip: float = 0.2,
        lr: float = 3e-4,
        entropy_coefficient: float = 0.01,
        value_coefficient: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        keep_best: bool = True,
        hidden: Sequence[int] = (64, 64),
        seed: Optional[int] = None,
        env_factory: Optional[Callable[[], MAPFEnv]] = None,
    ):
        self.template = env
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatches = epochs, minibatches
        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self._random = np.random.default_rng(seed)

        factory = env_factory or env.respawn
        self.vector = VectorMAPFEnv(factory, n=n_envs, seed=seed)
        self.agents = list(self.vector.possible_agents)
        self.rollout_steps = rollout_steps
        self.streams = n_envs * len(self.agents)

        obs_dim = int(env.observation_space(self.agents[0]).shape[0])
        critic_dim = env.state_size if self.centralized_critic else obs_dim
        self.policy: ActorCritic = make_actor_critic(
            backend,
            obs_dim=obs_dim,
            n_actions=env.action_space(self.agents[0]).n,
            critic_dim=critic_dim,
            hidden=hidden,
            lr=lr,
            seed=seed,
        )
        self.buffer = RolloutBuffer(rollout_steps, self.streams, obs_dim, critic_dim)
        self.history: List[dict] = []
        self.total_steps = 0
        self._episode_returns: List[float] = []
        self._solved: List[float] = []
        #: Best (solve rate, parameters) seen during training. Training here
        #: does not converge monotonically -- it peaks and then degrades -- so
        #: the final weights are routinely much worse than the best ones, and
        #: anything that reports "the trained policy" without this is reporting
        #: a number that was thrown away.
        self.best: Optional[dict] = None
        self.best_score: float = -1.0
        self.keep_best = keep_best

    # ------------------------------------------------------------------
    def _stack(self, observations: Sequence[Dict[str, np.ndarray]]) -> np.ndarray:
        """``[envs * agents, obs_dim]`` in a fixed, reproducible order."""
        return np.stack(
            [
                observation[agent]
                for observation in observations
                for agent in self.agents
            ]
        )

    def _critic_inputs(
        self, observations: Sequence[Dict[str, np.ndarray]]
    ) -> np.ndarray:
        if not self.centralized_critic:
            return self._stack(observations)
        # One global state per environment, repeated for each of its agents:
        # they share a state, they do not share an observation.
        states = self.vector.state()
        return np.repeat(states, len(self.agents), axis=0)

    def collect(self, observations):
        """Fill the buffer with one rollout, returning the trailing observations."""
        self.buffer.reset()
        for _ in range(self.rollout_steps):
            flat_observations = self._stack(observations)
            critic_inputs = self._critic_inputs(observations)
            actions, log_probs = self.policy.act(flat_observations)
            values = self.policy.value(critic_inputs)

            per_env: List[Dict[str, int]] = []
            index = 0
            for _ in range(self.vector.num_envs):
                per_env.append(
                    {
                        agent: int(actions[index + offset])
                        for offset, agent in enumerate(self.agents)
                    }
                )
                index += len(self.agents)

            next_observations, rewards, terminations, truncations, infos = (
                self.vector.step(per_env)
            )
            flat_rewards = np.array(
                [
                    reward.get(agent, 0.0)
                    for reward, info in zip(rewards, infos)
                    for agent in self.agents
                ],
                dtype=np.float32,
            )
            flat_dones = np.array(
                [
                    float(
                        terminations[e].get(agent, False)
                        or truncations[e].get(agent, False)
                    )
                    for e in range(self.vector.num_envs)
                    for agent in self.agents
                ],
                dtype=np.float32,
            )
            self.buffer.add(
                flat_observations,
                critic_inputs,
                actions,
                log_probs,
                flat_rewards,
                flat_dones,
                values,
            )
            self._record(infos)
            observations = next_observations
            self.total_steps += self.streams
        return observations

    def _record(self, infos) -> None:
        for info in infos:
            for entry in info.values():
                final = entry.get("final_info")
                if final and "episode" in final:
                    summary = final["episode"]
                    self._solved.append(1.0 if summary["solved"] else 0.0)
                    break

    def learn(
        self, total_steps: int = 100_000, log_every: int = 10, verbose: bool = False
    ):
        """Train for approximately ``total_steps`` environment-agent steps."""
        observations = self.vector.reset()
        started = time.perf_counter()
        iteration = 0
        while self.total_steps < total_steps:
            observations = self.collect(observations)
            last_values = self.policy.value(self._critic_inputs(observations))
            batch = self.buffer.batch(last_values, self.gamma, self.lam)
            stats = self._optimise(batch)
            iteration += 1

            recent = self._solved[-200:]
            record = dict(
                stats,
                iteration=iteration,
                steps=self.total_steps,
                solved=float(np.mean(recent)) if recent else 0.0,
                elapsed=time.perf_counter() - started,
            )
            self.history.append(record)
            if (
                self.keep_best
                and len(recent) >= 50
                and record["solved"] > self.best_score
            ):
                self.best_score = record["solved"]
                self.best = self.policy.state_dict()
            if verbose and iteration % log_every == 0:
                print(
                    "  %-6s iter %4d  steps %8d  solved %5.1f%% (best %5.1f%%)  "
                    "entropy %.3f  kl %.4f"
                    % (
                        self.name,
                        iteration,
                        self.total_steps,
                        100 * record["solved"],
                        100 * max(self.best_score, 0.0),
                        record["entropy"],
                        record["approx_kl"],
                    )
                )
        if self.keep_best and self.best is not None:
            # Hand back the best policy, not the last one.
            self.policy.load_state_dict(self.best)
        return self.history

    def _optimise(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """PPO's inner loop, optionally stopped early on KL divergence.

        ``target_kl`` defaults to off, and that is a measured decision rather
        than an oversight. This trainer does show the classic PPO failure --
        the solve rate peaks around 100k steps and then halves -- but KL-based
        early stopping is not the cure for it here: the measured KL per update
        stays between 0.0004 and 0.014 throughout, so the usual 0.02 threshold
        never once fires, and runs with and without it are bit-identical. The
        degradation tracks the entropy collapsing (1.61 to 0.32), not the
        policy taking large steps. :attr:`PPOTrainer.best` is the mitigation
        that does work; see :meth:`learn`.
        """
        size = len(batch["actions"])
        indices = np.arange(size)
        chunk = max(1, size // self.minibatches)
        last: Dict[str, float] = {}
        for _ in range(self.epochs):
            self._random.shuffle(indices)
            for start in range(0, size, chunk):
                subset = indices[start : start + chunk]
                if len(subset) < 2:
                    continue
                last = self.policy.update(
                    {key: value[subset] for key, value in batch.items()},
                    clip=self.clip,
                    value_coefficient=self.value_coefficient,
                    entropy_coefficient=self.entropy_coefficient,
                    max_grad_norm=self.max_grad_norm,
                )
            if (
                self.target_kl is not None
                and last.get("approx_kl", 0.0) > self.target_kl
            ):
                break
        return last

    # ------------------------------------------------------------------
    def act(
        self, observations: Dict[str, np.ndarray], deterministic: bool = True
    ) -> Dict[str, int]:
        """Greedy (or sampled) joint action -- the deployment interface."""
        agents = list(observations)
        stacked = np.stack([observations[agent] for agent in agents])
        actions, _ = self.policy.act(stacked, deterministic=deterministic)
        return {agent: int(action) for agent, action in zip(agents, actions)}

    def save(self, path: str) -> None:
        import pickle

        with open(path, "wb") as handle:
            pickle.dump(
                {"algorithm": self.name, "state": self.policy.state_dict()}, handle
            )

    def load(self, path: str) -> None:
        import pickle

        with open(path, "rb") as handle:
            self.policy.load_state_dict(pickle.load(handle)["state"])


class IPPO(PPOTrainer):
    """Independent PPO: every agent's critic sees only what its actor sees."""

    name = "ippo"
    centralized_critic = False


class MAPPO(PPOTrainer):
    """Multi-Agent PPO: one centralized critic over the global state."""

    name = "mappo"
    centralized_critic = True


ALGORITHMS: Dict[str, Type[PPOTrainer]] = {"ippo": IPPO, "mappo": MAPPO}


def register_algorithm(name: str):
    """Class decorator registering a trainer under ``name``."""

    def decorator(cls: Type[PPOTrainer]) -> Type[PPOTrainer]:
        ALGORITHMS[name.lower()] = cls
        return cls

    return decorator


def available_algorithms() -> List[str]:
    return sorted(ALGORITHMS)


def make_trainer(name, env: MAPFEnv, **kwargs) -> PPOTrainer:
    """Resolve ``name`` to a trainer bound to ``env``."""
    if isinstance(name, PPOTrainer):
        return name
    try:
        factory = ALGORITHMS[str(name).lower()]
    except KeyError as error:
        raise ValueError(
            "Unknown algorithm %r. Available: %s"
            % (name, ", ".join(available_algorithms()))
        ) from error
    return factory(env, **kwargs)
