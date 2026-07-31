"""Adapters onto the wider ecosystem, and a vectorised runner.

:class:`~pymapf.rl.env.MAPFEnv` already speaks the PettingZoo Parallel API, so
these wrappers add interoperability rather than translation: a real
``ParallelEnv`` subclass for libraries that type-check against it, a
single-agent Gymnasium view for the many algorithms that only handle one agent,
and a vectorised runner so rollout collection is not one environment at a time.

Every import of an optional dependency happens inside the function that needs
it, so importing this module costs nothing on a machine with neither installed.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .env import MAPFEnv
from .spaces import to_gymnasium

__all__ = ["PettingZooParallel", "SingleAgentGym", "VectorMAPFEnv"]


def PettingZooParallel(env: MAPFEnv):
    """Wrap ``env`` in a genuine ``pettingzoo.ParallelEnv``.

    Built as a factory rather than a module-level class because the base class
    only exists once PettingZoo is imported, and this package must import
    cleanly without it.
    """
    try:
        from pettingzoo.utils.env import ParallelEnv
    except ImportError as error:  # pragma: no cover - needs pettingzoo
        raise ImportError(
            "pettingzoo is not installed; `pip install pymapf[rl]`. MAPFEnv "
            "already implements the parallel API, so this wrapper is only "
            "needed for code that isinstance-checks against ParallelEnv"
        ) from error

    class _PettingZooMAPF(ParallelEnv):
        metadata = dict(MAPFEnv.metadata, render_modes=["human", "ansi"])

        def __init__(self, inner: MAPFEnv):
            super().__init__()
            self.inner = inner
            self.possible_agents = list(inner.possible_agents)
            self.agents = list(inner.agents)

        def observation_space(self, agent):
            return to_gymnasium(self.inner.observation_space(agent))

        def action_space(self, agent):
            return to_gymnasium(self.inner.action_space(agent))

        def reset(self, seed=None, options=None):
            observations, infos = self.inner.reset(seed=seed, options=options)
            self.agents = list(self.inner.agents)
            return observations, infos

        def step(self, actions):
            result = self.inner.step(actions)
            self.agents = list(self.inner.agents)
            return result

        def render(self):
            return self.inner.render(mode="ansi")

        def state(self):
            return self.inner.state()

        def close(self):
            self.inner.close()

    return _PettingZooMAPF(env)


class SingleAgentGym:
    """A Gymnasium-style view in which *one* agent learns and the rest are fixed.

    The standard way to get a single-agent algorithm onto a multi-agent problem,
    and worth being explicit about what it costs: the other agents become part
    of the environment, so from the learner's side the world is non-stationary
    the moment their policy changes. That is the whole reason IPPO and MAPPO
    exist. This is here as a baseline and a debugging aid -- if a policy cannot
    learn with the others frozen, the multi-agent machinery is not what is
    wrong.

    Args:
        env: the underlying :class:`MAPFEnv`.
        agent: which agent learns. Defaults to the first.
        others: policy for everyone else, ``f(observation) -> action``.
            Defaults to standing still, which is the least confusing baseline.
    """

    def __init__(
        self,
        env: MAPFEnv,
        agent: Optional[str] = None,
        others: Optional[Callable[[np.ndarray], int]] = None,
    ):
        self.env = env
        self.agent = agent or env.possible_agents[0]
        if self.agent not in env.possible_agents:
            raise ValueError("%r is not an agent in this environment" % self.agent)
        self.others = others or (lambda observation: 0)
        self._last: Dict[str, np.ndarray] = {}

    @property
    def observation_space(self):
        return self.env.observation_space(self.agent)

    @property
    def action_space(self):
        return self.env.action_space(self.agent)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self._last = observations
        return observations[self.agent], infos[self.agent]

    def step(self, action: int):
        actions = {
            name: (int(action) if name == self.agent else int(self.others(self._last[name])))
            for name in self.env.agents
        }
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self._last = observations
        return (
            observations[self.agent],
            rewards[self.agent],
            terminations[self.agent],
            truncations[self.agent],
            infos[self.agent],
        )

    def render(self, mode: str = "human"):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()


class VectorMAPFEnv:
    """Runs several independent environments in lockstep, in one process.

    PPO wants a wide batch of decorrelated transitions per update, and one
    episode at a time gives it a narrow, highly correlated one. This is the
    cheap fix: ``n`` copies stepped together, each auto-resetting when it
    finishes so the rollout never stalls waiting for the slowest episode.

    Deliberately single-process. These environments are pure Python over small
    grids, so the cost is dominated by the policy forward pass; subprocess
    workers would add serialisation overhead to buy parallelism that numpy is
    already providing at the batch level.
    """

    def __init__(self, factory: Callable[[], MAPFEnv], n: int = 8, seed: Optional[int] = None):
        if n < 1:
            raise ValueError("need at least one environment, got %d" % n)
        self.envs: List[MAPFEnv] = [factory() for _ in range(n)]
        self.n = n
        self._random = np.random.default_rng(seed)
        self.possible_agents = list(self.envs[0].possible_agents)

    @property
    def num_envs(self) -> int:
        return self.n

    def reset(self, seed: Optional[int] = None) -> List[Dict[str, np.ndarray]]:
        if seed is not None:
            self._random = np.random.default_rng(seed)
        return [
            env.reset(seed=int(self._random.integers(1 << 30)))[0] for env in self.envs
        ]

    def step(self, actions: Sequence[Dict[str, int]]):
        """Step every environment; finished ones reset and report it.

        The observation returned for an environment that just finished is the
        *first observation of the next episode*, and its ``info`` carries
        ``final_info``. That is the auto-reset convention the vector APIs use,
        and getting it wrong silently bootstraps a value estimate across an
        episode boundary.
        """
        observations, rewards, terminations, truncations, infos = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            observation, reward, termination, truncation, info = env.step(action)
            done = (not env.agents) or any(termination.values()) or any(truncation.values())
            if done:
                final_info = {agent: dict(entry) for agent, entry in info.items()}
                final_observation = observation
                truncated = any(truncation.values())
                observation, reset_info = env.reset(
                    seed=int(self._random.integers(1 << 30))
                )
                info = {
                    agent: dict(
                        reset_info.get(agent, {}),
                        final_info=final_info.get(agent, {}),
                        # Kept because a *truncated* episode still has a future:
                        # bootstrapping its value from zero, as one would for a
                        # real terminal state, biases every estimate near the
                        # horizon downward.
                        final_observation=final_observation.get(agent),
                        truncated=truncated,
                    )
                    for agent in env.agents
                }
            observations.append(observation)
            rewards.append(reward)
            terminations.append(termination)
            truncations.append(truncation)
            infos.append(info)
        return observations, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        return np.stack([env.state() for env in self.envs])

    def close(self) -> None:
        for env in self.envs:
            env.close()
