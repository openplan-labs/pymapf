"""MAPF as a multi-agent reinforcement learning environment.

This follows the **PettingZoo Parallel API** -- ``reset`` returning
``(observations, infos)``, ``step`` taking a dict of actions and returning five
dicts -- because that is the interface every current MARL library speaks. It
implements that API without importing PettingZoo, so the environment still runs
in a bare CI job; :meth:`MAPFEnv.to_pettingzoo` wraps it in the real
``ParallelEnv`` when the package is installed.

What makes this worth building rather than reaching for an existing gridworld
is the last method on the class: :meth:`MAPFEnv.solution`. A rollout comes back
as a :class:`pymapf.Solution` -- the *same* object CBS returns -- so a learned
policy and an optimal planner are scored by identical code, on identical
instances, under identical conflict rules. Not a reimplementation of
sum-of-costs that agrees with the planner's if you read both carefully: the
planner's own.

Transition semantics
--------------------

Agents move simultaneously, and the conflict rules are MAPF's, not a
gridworld's approximation of them:

* a move into a wall or off the map is refused (``blocked``);
* two agents claiming the same cell is a **vertex conflict** -- both are
  refused (``collided``);
* two agents exchanging cells is an **edge conflict**, refused likewise, which
  is the rule a naive "check the destination is empty" implementation misses;
* refusals **cascade**. An agent pushed back onto its current cell can invalidate
  a third agent that was moving into the cell it was about to vacate, so
  resolution iterates to a fixed point rather than resolving once.

Agents are never deleted mid-episode, and an agent sitting on its goal may move
off it again -- sometimes it must, to let someone through. The episode
terminates the moment every agent is simultaneously on its goal, which is
exactly the condition a MAPF plan has to satisfy.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from pymapf.core.grid import Cell
from pymapf.core.solver import Solution

from .observation import ObservationEncoder, get_observation
from .reward import RewardFunction, get_reward
from .spaces import Box, Discrete

__all__ = ["MAPFEnv", "ORTHOGONAL_ACTIONS", "DIAGONAL_ACTIONS"]

#: Action index -> (row, column) delta. Index 0 is always "stay", which matters:
#: waiting is a first-class move in MAPF, not a failure to act.
ORTHOGONAL_ACTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)
DIAGONAL_ACTIONS: Tuple[Tuple[int, int], ...] = ORTHOGONAL_ACTIONS + (
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


class MAPFEnv:
    """A MAPF instance as a parallel multi-agent environment.

    Args:
        scenario: a :class:`pymapf.Scenario`, a :class:`pymapf.MAPFProblem`, or
            the name of a registered scenario family.
        observation: encoder name or instance (see
            :mod:`pymapf.rl.observation`).
        reward: reward function name or instance (see :mod:`pymapf.rl.reward`).
        max_steps: truncation horizon. Defaults to a multiple of the grid
            diagonal, which scales with the instance instead of being a number
            that silently becomes too small on bigger maps.
        randomise: draw a fresh instance from the scenario family on every
            ``reset``. This is what makes a policy generalise rather than
            memorise one map, so it is on by default when a family name is
            given.
        scenario_kwargs: forwarded to :func:`pymapf.build_scenario` when
            ``randomise`` is drawing new instances.
    """

    metadata = {"name": "pymapf_mapf_v0", "is_parallelizable": True}

    def __init__(
        self,
        scenario="random_obstacles",
        observation="local",
        reward="shaped",
        max_steps: Optional[int] = None,
        randomise: Optional[bool] = None,
        seed: Optional[int] = None,
        observation_kwargs: Optional[dict] = None,
        reward_kwargs: Optional[dict] = None,
        **scenario_kwargs,
    ):
        from pymapf import MAPFProblem, build_scenario
        from pymapf.scenarios import Scenario

        self._scenario_kwargs = dict(scenario_kwargs)
        self._family: Optional[str] = None
        self._problem = None

        if isinstance(scenario, str):
            self._family = scenario
            self._problem = build_scenario(scenario, **scenario_kwargs).to_problem()
            self.randomise = True if randomise is None else randomise
        elif isinstance(scenario, Scenario):
            self._problem = scenario.to_problem()
            self.randomise = bool(randomise)
        elif isinstance(scenario, MAPFProblem):
            self._problem = scenario
            self.randomise = bool(randomise)
        else:
            raise TypeError(
                "scenario must be a name, a Scenario or a MAPFProblem, got %r"
                % type(scenario).__name__
            )
        if self.randomise and self._family is None:
            raise ValueError("randomise=True needs a scenario family name to draw from")

        # The *specification* is kept, not just the built objects, so respawn()
        # can construct fresh ones. Sharing a reward function across vectorised
        # workers would be a correctness bug rather than an efficiency one:
        # ShapedReward caches a distance field per instance, and workers reset
        # at different times, so one worker's reset would silently overwrite the
        # potentials another worker is midway through using.
        self._observation_spec = (observation, dict(observation_kwargs or {}))
        self._reward_spec = (reward, dict(reward_kwargs or {}))
        self.encoder: ObservationEncoder = get_observation(
            observation, **(observation_kwargs or {})
        )
        self.reward_function: RewardFunction = get_reward(reward, **(reward_kwargs or {}))
        self._max_steps = max_steps
        self._random = np.random.default_rng(seed)
        self._episode = 0

        self._bind(self._problem)
        self._reset_state()

    # ------------------------------------------------------------------
    # instance binding
    # ------------------------------------------------------------------
    def _bind(self, problem) -> None:
        """Adopt a concrete instance: grid, agent names, starts and goals."""
        self._problem = problem
        self.grid = problem.grid
        self.allow_diagonals = problem.allow_diagonals
        self.possible_agents: List[str] = [agent.name for agent in problem.agents]
        self.starts: Dict[str, Cell] = {a.name: a.start for a in problem.agents}
        self.goals: Dict[str, Cell] = {a.name: a.goal for a in problem.agents}
        self.actions = DIAGONAL_ACTIONS if self.allow_diagonals else ORTHOGONAL_ACTIONS

        if self._max_steps is not None:
            self.max_steps = int(self._max_steps)
        else:
            # Long enough that a competent policy is never truncated, short
            # enough that a hopeless one does not burn the rollout budget.
            self.max_steps = int(4 * (self.grid.height + self.grid.width))

    @property
    def problem(self) -> "object":
        """The :class:`pymapf.MAPFProblem` currently loaded.

        Public because the evaluation harness needs to hand exactly this
        instance to a planner; reaching through ``env._problem`` to do it was
        the tell that the attribute belonged in the API.
        """
        return self._problem

    def respawn(self) -> "MAPFEnv":
        """A fresh environment configured like this one.

        Rebuilt from the configuration rather than deep-copied, because the
        scenario family and its kwargs define the instance *distribution* --
        copying a bound instance would hand every vectorised worker the same
        map, and the policy would learn that one map.
        """
        import copy

        observation, observation_kwargs = self._observation_spec
        reward, reward_kwargs = self._reward_spec
        # A spec given as an *instance* would otherwise be shared by every
        # worker, which is the same cache-clobbering bug as above arriving by a
        # different route. Copied rather than reused; a name already builds
        # something fresh.
        if isinstance(observation, ObservationEncoder):
            observation = copy.deepcopy(observation)
        if isinstance(reward, RewardFunction):
            reward = copy.deepcopy(reward)
        return MAPFEnv(
            self._family if self._family is not None else self._problem,
            observation=observation,
            reward=reward,
            observation_kwargs=observation_kwargs,
            reward_kwargs=reward_kwargs,
            max_steps=self._max_steps,
            randomise=self.randomise,
            **self._scenario_kwargs,
        )

    def instance_signature(self) -> tuple:
        """Identifies the current instance, for caches keyed on it."""
        return (
            id(self.grid),
            tuple(self.possible_agents),
            tuple(self.goals[a] for a in self.possible_agents),
        )

    def _reset_state(self) -> None:
        self.agents: List[str] = list(self.possible_agents)
        self.positions: Dict[str, Cell] = dict(self.starts)
        self.step_count = 0
        self._trajectory: Dict[str, List[Cell]] = {
            agent: [self.starts[agent]] for agent in self.possible_agents
        }
        self._collisions = 0

    # ------------------------------------------------------------------
    # spaces (PettingZoo calls these as methods)
    # ------------------------------------------------------------------
    def observation_space(self, agent: str) -> Box:
        return self.encoder.space(self)

    def action_space(self, agent: str) -> Discrete:
        return Discrete(len(self.actions))

    @property
    def observation_spaces(self) -> Dict[str, Box]:
        return {agent: self.observation_space(agent) for agent in self.possible_agents}

    @property
    def action_spaces(self) -> Dict[str, Discrete]:
        return {agent: self.action_space(agent) for agent in self.possible_agents}

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        return len(self.possible_agents)

    # ------------------------------------------------------------------
    # the parallel API
    # ------------------------------------------------------------------
    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        if seed is not None:
            self._random = np.random.default_rng(seed)
        if self.randomise:
            from pymapf import build_scenario

            kwargs = dict(self._scenario_kwargs)
            kwargs["seed"] = int(self._random.integers(1 << 30))
            self._bind(build_scenario(self._family, **kwargs).to_problem())

        self._reset_state()
        self._episode += 1
        self.reward_function.reset(self)
        observations = self.encoder.encode_all(self)
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: Dict[str, int]):
        """Advance one timestep. ``actions`` maps every live agent to an index."""
        missing = set(self.agents) - set(actions)
        if missing:
            raise KeyError(
                "step() needs an action for every agent; missing %s"
                % ", ".join(sorted(missing))
            )

        previous = dict(self.positions)
        resolved, blocked, collided = self._resolve(actions)
        self.positions = resolved
        self.step_count += 1
        for agent in self.possible_agents:
            self._trajectory[agent].append(self.positions[agent])
        self._collisions += sum(collided.values())

        at_goal = {
            agent: self.positions[agent] == self.goals[agent]
            for agent in self.possible_agents
        }
        was_at_goal = {
            agent: previous[agent] == self.goals[agent]
            for agent in self.possible_agents
        }
        solved = all(at_goal.values())
        truncated = self.step_count >= self.max_steps and not solved

        rewards = {
            agent: self.reward_function.compute(
                self,
                agent,
                previous[agent],
                self.positions[agent],
                blocked[agent],
                collided[agent],
                at_goal[agent],
                was_at_goal[agent],
            )
            for agent in self.agents
        }
        terminations = {agent: solved for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos = {
            agent: {
                "blocked": blocked[agent],
                "collided": collided[agent],
                "at_goal": at_goal[agent],
                "distance_to_goal": self._manhattan(agent),
            }
            for agent in self.agents
        }
        if solved or truncated:
            summary = self.episode_summary(solved)
            for agent in self.agents:
                infos[agent]["episode"] = summary

        observations = self.encoder.encode_all(self)
        if solved or truncated:
            # The parallel API empties `agents` once the episode is over.
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def _resolve(self, actions: Dict[str, int]):
        """Apply the joint action under MAPF's conflict rules.

        Refusals cascade, so this iterates: an agent pushed back onto its own
        cell can invalidate another agent that was moving into the cell it was
        about to leave. Each round can only ever *add* refusals, and there are
        finitely many agents, so the loop terminates.
        """
        intended: Dict[str, Cell] = {}
        blocked: Dict[str, bool] = {agent: False for agent in self.possible_agents}
        collided: Dict[str, bool] = {agent: False for agent in self.possible_agents}

        for agent in self.possible_agents:
            current = self.positions[agent]
            action = int(actions.get(agent, 0))
            if not 0 <= action < len(self.actions):
                raise ValueError(
                    "action %d out of range for %s (0..%d)"
                    % (action, agent, len(self.actions) - 1)
                )
            delta = self.actions[action]
            target = (current[0] + delta[0], current[1] + delta[1])
            if delta == (0, 0):
                intended[agent] = current
            elif not self.grid.is_free(target):
                intended[agent] = current          # walls refuse the move
                blocked[agent] = True
            elif self.allow_diagonals and delta[0] and delta[1]:
                # Corner-cutting uses the same rule as the planners' neighbours.
                if target in self.grid.neighbors(current, allow_diagonals=True):
                    intended[agent] = target
                else:
                    intended[agent] = current
                    blocked[agent] = True
            else:
                intended[agent] = target

        while True:
            changed = False

            # Vertex conflicts: more than one agent claiming a cell.
            claims: Dict[Cell, List[str]] = {}
            for agent, cell in intended.items():
                claims.setdefault(cell, []).append(agent)
            for cell, claimants in claims.items():
                if len(claimants) < 2:
                    continue
                for agent in claimants:
                    if intended[agent] != self.positions[agent]:
                        intended[agent] = self.positions[agent]
                        collided[agent] = True
                        changed = True

            # Edge conflicts: two agents exchanging cells.
            for first in self.possible_agents:
                for second in self.possible_agents:
                    if first >= second:
                        continue
                    if (
                        intended[first] == self.positions[second]
                        and intended[second] == self.positions[first]
                        and self.positions[first] != self.positions[second]
                    ):
                        for agent in (first, second):
                            if intended[agent] != self.positions[agent]:
                                intended[agent] = self.positions[agent]
                                collided[agent] = True
                                changed = True

            if not changed:
                return intended, blocked, collided

    def _manhattan(self, agent: str) -> int:
        row, col = self.positions[agent]
        goal_row, goal_col = self.goals[agent]
        return abs(row - goal_row) + abs(col - goal_col)

    # ------------------------------------------------------------------
    # the centralized view, for MAPPO's critic
    # ------------------------------------------------------------------
    def state(self) -> np.ndarray:
        """Global state: every agent's position and goal, normalised.

        MAPPO's critic is centralized -- it is allowed to see this at training
        time even though no policy ever does, which is the entire content of
        "centralized training, decentralized execution".
        """
        height = max(1, self.grid.height - 1)
        width = max(1, self.grid.width - 1)
        values: List[float] = []
        for agent in self.possible_agents:
            row, col = self.positions[agent]
            goal_row, goal_col = self.goals[agent]
            values.extend(
                [
                    row / height,
                    col / width,
                    goal_row / height,
                    goal_col / width,
                    1.0 if self.positions[agent] == self.goals[agent] else 0.0,
                ]
            )
        values.append(self.step_count / max(1, self.max_steps))
        return np.asarray(values, dtype=np.float32)

    @property
    def state_size(self) -> int:
        return 5 * len(self.possible_agents) + 1

    # ------------------------------------------------------------------
    # scoring, on the planner's own terms
    # ------------------------------------------------------------------
    def solution(self, algorithm: str = "policy") -> Solution:
        """The episode so far, as a :class:`pymapf.Solution`.

        Trailing timesteps in which an agent simply sits on its goal are
        trimmed, because that is what the planners' paths do: ``sum_of_costs``
        is defined as the sum of arrival times, so leaving the padding in would
        charge a learned policy for waiting that CBS is not charged for. With
        the padding gone, the two are directly comparable -- and
        :meth:`Solution.is_valid` will check the learned paths for conflicts
        using the very same code that validates the planner's.
        """
        paths: Dict[str, List[Cell]] = {}
        for agent, path in self._trajectory.items():
            goal = self.goals[agent]
            end = len(path)
            while end > 1 and path[end - 1] == goal and path[end - 2] == goal:
                end -= 1
            paths[agent] = list(path[:end])
        return Solution(paths=paths, algorithm=algorithm, runtime=0.0, expansions=0)

    def episode_summary(self, solved: Optional[bool] = None) -> dict:
        if solved is None:
            solved = all(
                self.positions[agent] == self.goals[agent]
                for agent in self.possible_agents
            )
        solution = self.solution()
        return {
            "solved": bool(solved),
            "steps": self.step_count,
            "collisions": self._collisions,
            "sum_of_costs": solution.sum_of_costs if solved else None,
            "makespan": solution.makespan if solved else None,
            "agents_at_goal": sum(
                self.positions[agent] == self.goals[agent]
                for agent in self.possible_agents
            ),
        }

    # ------------------------------------------------------------------
    # interop and presentation
    # ------------------------------------------------------------------
    def to_pettingzoo(self):
        """Wrap in a real ``pettingzoo.ParallelEnv``, if it is installed."""
        from .wrappers import PettingZooParallel

        return PettingZooParallel(self)

    def render(self, mode: str = "human") -> str:
        """An ASCII frame. Lowercase marks an agent standing on its goal."""
        canvas = [
            ["#" if not self.grid.is_free((r, c)) else "." for c in range(self.grid.width)]
            for r in range(self.grid.height)
        ]
        for index, agent in enumerate(self.possible_agents):
            row, col = self.goals[agent]
            if canvas[row][col] == ".":
                canvas[row][col] = "_"
        for index, agent in enumerate(self.possible_agents):
            row, col = self.positions[agent]
            letter = chr(ord("A") + index % 26)
            canvas[row][col] = (
                letter.lower() if self.positions[agent] == self.goals[agent] else letter
            )
        frame = "\n".join("".join(row) for row in canvas)
        if mode == "human":
            print(frame)
        return frame

    def close(self) -> None:
        """Nothing to release; here because the API has it."""

    def __repr__(self) -> str:
        return "MAPFEnv(%d agents, %dx%d, observation=%r, reward=%r)" % (
            len(self.possible_agents),
            self.grid.height,
            self.grid.width,
            self.encoder.name,
            self.reward_function.name,
        )
