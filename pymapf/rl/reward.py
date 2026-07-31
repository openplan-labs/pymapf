"""Reward functions, as swappable objects.

The interesting one here is :class:`ShapedReward`, and it is interesting
because of something this library happens to own: an *exact* distance-to-goal
oracle. :func:`pymapf.core.heuristics.true_distance` runs a backward Dijkstra
over the real grid, so it knows the true remaining cost from every cell,
obstacles and all -- not a Manhattan guess.

That makes **potential-based shaping** available in its strong form. Ng, Harada
and Russell (1999) proved that a shaping term of the form

    ``F(s, s') = gamma * Phi(s') - Phi(s)``

leaves the optimal policy unchanged, for *any* potential ``Phi``. It is the one
kind of reward engineering that cannot quietly teach the agent the wrong task.
Taking ``Phi(s) = -true_distance(s)`` turns the sparse "you arrived" signal into
a dense one without changing what "arrived" is worth, and on grid MAPF the
difference between the two is the difference between learning something and
learning nothing: with sparse rewards a random policy essentially never reaches
a goal, so there is no gradient to follow.

The shaping is applied per agent on its own progress, which keeps it
decentralized -- an agent needs only its own distance-to-go, which it could
compute onboard from a map it already has.

Registry, like everything else::

    MAPFEnv(scenario, reward="shaped")
    MAPFEnv(scenario, reward=ShapedReward(collision=-2.0))

References
----------
* Ng, A. Y.; Harada, D.; and Russell, S. 1999. *Policy invariance under reward
  transformations: theory and application to reward shaping.* ICML 1999:
  278-287.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Type

__all__ = [
    "RewardFunction",
    "SparseReward",
    "ShapedReward",
    "register_reward",
    "get_reward",
    "available_rewards",
]


class RewardFunction(ABC):
    """Scores one transition, per agent.

    ``compute`` is handed everything about the step that just happened, which
    keeps the environment free of reward policy entirely.
    """

    name = "abstract"

    @abstractmethod
    def compute(
        self,
        env,
        agent: str,
        previous,
        current,
        blocked: bool,
        collided: bool,
        at_goal: bool,
        was_at_goal: bool,
    ) -> float:
        """Reward for ``agent`` moving from ``previous`` to ``current``."""

    def reset(self, env) -> None:
        """Hook for per-episode state. Most functions need none."""

    def __repr__(self) -> str:
        return "%s()" % type(self).__name__


class SparseReward(RewardFunction):
    """Step cost, goal bonus, collision penalty. Nothing else.

    The honest baseline. It encodes exactly the MAPF objective -- every
    timestep an agent is not parked on its goal costs it one -- and nothing
    about *how* to get there. On anything but a tiny open map a randomly
    initialised policy will not stumble onto a goal often enough to learn from,
    which is precisely why :class:`ShapedReward` exists and why the comparison
    between them is worth running.

    Args:
        step: charged every timestep the agent is not settled on its goal.
        goal: one-off bonus for arriving.
        collision: charged when a move is refused because of another agent.
        blocked: charged when a move is refused by a wall or the map edge.
    """

    name = "sparse"

    def __init__(
        self,
        step: float = -0.05,
        goal: float = 1.0,
        collision: float = -0.5,
        blocked: float = -0.1,
    ):
        self.step = step
        self.goal = goal
        self.collision = collision
        self.blocked = blocked

    def compute(
        self,
        env,
        agent: str,
        previous,
        current,
        blocked: bool,
        collided: bool,
        at_goal: bool,
        was_at_goal: bool,
    ) -> float:
        reward = 0.0
        # An agent parked on its goal is done; charging it would teach it to
        # leave, and it is not accruing cost in the MAPF objective either.
        if not at_goal:
            reward += self.step
        if collided:
            reward += self.collision
        elif blocked:
            reward += self.blocked
        if at_goal and not was_at_goal:
            reward += self.goal
        return reward


class ShapedReward(SparseReward):
    """Sparse reward plus exact potential-based shaping toward the goal.

    ``Phi(s) = -distance_to_goal(s)`` using the library's own backward-Dijkstra
    oracle, so the potential is the true remaining cost rather than a straight
    line through walls. The shaping term ``gamma * Phi(s') - Phi(s)`` is
    policy-invariant (Ng et al. 1999): it changes how fast the policy learns,
    not what it converges to, which is the property that makes it safe to use
    when the whole point is to compare against an optimal planner.

    One implementation note. Cells with no path to the goal have infinite
    potential; those are clamped to the largest finite distance rather than
    propagating an infinity into the return. An agent that walks into a sealed
    pocket should get a bad reward, not a NaN.

    Args:
        gamma: must match the discount the learner uses, or the invariance
            argument does not hold.
        scale: multiplies the shaping term. 1.0 keeps the theoretical guarantee;
            larger values are a knob, and a lie about the guarantee.
    """

    name = "shaped"

    def __init__(self, gamma: float = 0.99, scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.scale = scale
        self._potentials: Dict[str, object] = {}
        self._finite_max: Dict[str, float] = {}

    def reset(self, env) -> None:
        """Build one distance field per goal, once per episode.

        Cached on the environment's instance identity: the goals only change
        when the instance does, and a backward Dijkstra per agent per episode
        would otherwise dominate the rollout cost.
        """
        from pymapf.algorithms.search import distance_table

        signature = env.instance_signature()
        if getattr(self, "_signature", None) == signature and self._potentials:
            return
        self._signature = signature
        self._potentials = {}
        self._finite_max = {}
        for agent in env.possible_agents:
            # The backward Dijkstra behind `true_distance`, used directly: the
            # table is what we want, and it is goal-specific either way.
            table = distance_table(env.grid, env.goals[agent], env.allow_diagonals)
            self._potentials[agent] = table
            finite = [value for value in table.values() if value != float("inf")]
            self._finite_max[agent] = float(max(finite)) if finite else 0.0

    def potential(self, agent: str, cell) -> float:
        """``-distance to goal``, with unreachable cells clamped."""
        table = self._potentials.get(agent)
        if table is None:
            return 0.0
        distance = table.get(cell, float("inf"))
        if distance == float("inf"):
            # Worse than anywhere reachable, but finite.
            distance = self._finite_max.get(agent, 0.0) + 1.0
        return -float(distance)

    def compute(
        self,
        env,
        agent: str,
        previous,
        current,
        blocked: bool,
        collided: bool,
        at_goal: bool,
        was_at_goal: bool,
    ) -> float:
        base = super().compute(
            env, agent, previous, current, blocked, collided, at_goal, was_at_goal
        )
        shaping = self.gamma * self.potential(agent, current) - self.potential(
            agent, previous
        )
        return base + self.scale * shaping


REWARDS: Dict[str, Type[RewardFunction]] = {
    "sparse": SparseReward,
    "shaped": ShapedReward,
}


def register_reward(name: str):
    """Class decorator registering a reward function under ``name``."""

    def decorator(cls: Type[RewardFunction]) -> Type[RewardFunction]:
        REWARDS[name.lower()] = cls
        return cls

    return decorator


def available_rewards() -> List[str]:
    return sorted(REWARDS)


def get_reward(name, **kwargs) -> RewardFunction:
    """Resolve ``name`` (or an instance) to a :class:`RewardFunction`."""
    if isinstance(name, RewardFunction):
        return name
    try:
        factory = REWARDS[str(name).lower()]
    except KeyError as error:
        raise ValueError(
            "Unknown reward %r. Available: %s" % (name, ", ".join(available_rewards()))
        ) from error
    return factory(**kwargs)
