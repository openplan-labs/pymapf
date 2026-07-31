"""Reinforcement learning on the same MAPF instances the planners solve.

The point of putting RL *inside* this library rather than beside it is that the
hard part of a learned-MAPF paper is usually the part this repository already
owns: correct instances, correct conflict semantics, and an optimal baseline to
be measured against. A rollout here comes back as a
:class:`pymapf.Solution` -- the same type CBS returns -- so "did the policy
solve it" and "how far off optimal was it" are answered by the planner's own
code::

    from pymapf.rl import MAPFEnv, make_trainer, evaluate

    env = MAPFEnv("random_obstacles", n_agents=4, width=10, height=10)
    trainer = make_trainer("mappo", env)
    trainer.learn(total_steps=200_000)
    print(evaluate(env, trainer.policy, episodes=100, baseline="cbs"))

Everything is a registry, matching the rest of the library: observations,
rewards and algorithms are all chosen by name and extended from outside with a
decorator, so adding a new encoding or a new learner does not mean editing this
package.
"""

from .algorithms import (
    IPPO,
    MAPPO,
    PPOTrainer,
    RolloutBuffer,
    available_algorithms,
    compute_gae,
    make_trainer,
    register_algorithm,
)
from .env import DIAGONAL_ACTIONS, ORTHOGONAL_ACTIONS, MAPFEnv
from .evaluate import EvaluationResult, compare, evaluate, rollout
from .networks import (
    ActorCritic,
    NumpyActorCritic,
    TorchActorCritic,
    available_backends,
    make_actor_critic,
)
from .observation import (
    GlobalGrid,
    LocalWindow,
    ObservationEncoder,
    available_observations,
    get_observation,
    register_observation,
)
from .reward import (
    RewardFunction,
    ShapedReward,
    SparseReward,
    available_rewards,
    get_reward,
    register_reward,
)
from .spaces import Box, Discrete, Space, to_gymnasium
from .wrappers import PettingZooParallel, SingleAgentGym, VectorMAPFEnv

__all__ = [
    # environment
    "MAPFEnv",
    "ORTHOGONAL_ACTIONS",
    "DIAGONAL_ACTIONS",
    # spaces
    "Space",
    "Discrete",
    "Box",
    "to_gymnasium",
    # observations
    "ObservationEncoder",
    "LocalWindow",
    "GlobalGrid",
    "register_observation",
    "get_observation",
    "available_observations",
    # rewards
    "RewardFunction",
    "SparseReward",
    "ShapedReward",
    "register_reward",
    "get_reward",
    "available_rewards",
    # wrappers
    "PettingZooParallel",
    "SingleAgentGym",
    "VectorMAPFEnv",
    # networks
    "ActorCritic",
    "NumpyActorCritic",
    "TorchActorCritic",
    "make_actor_critic",
    "available_backends",
    # algorithms
    "PPOTrainer",
    "IPPO",
    "MAPPO",
    "RolloutBuffer",
    "compute_gae",
    "make_trainer",
    "register_algorithm",
    "available_algorithms",
    # evaluation
    "rollout",
    "evaluate",
    "compare",
    "EvaluationResult",
]
