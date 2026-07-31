#!/usr/bin/env python3
"""Train IPPO and MAPPO on MAPF, then score them against the planners.

Everything the film and the survey claim about the RL layer is produced here,
on instances drawn by seed so every method sees the same maps::

    python scripts/train_rl.py                       # the standard sweep
    python scripts/train_rl.py --steps 1000000       # longer
    python scripts/train_rl.py --backend torch       # if torch is installed
    python scripts/train_rl.py --output docs/assets/rl-benchmark.json

The comparison is against CBS, which is *optimal*, so the reported ratio is
true suboptimality rather than a gap against another heuristic. Instances CBS
cannot close inside its time limit are excluded from the ratio rather than
counted as wins for the learner.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import pymapf  # noqa: E402
from pymapf.rl import MAPFEnv, compare, make_trainer  # noqa: E402

# Each setting is a scenario family plus the knobs that make it harder. The
# progression is deliberate: an open room is a coordination problem only where
# paths cross, obstacles add real routing, and the bottleneck forces agents to
# take turns through one gap -- which is where independent learners classically
# struggle and a centralized critic should earn its keep.
SETTINGS = [
    ("empty_room", dict(height=8, width=8, n_agents=2)),
    ("empty_room", dict(height=10, width=10, n_agents=4)),
    ("random_obstacles", dict(height=10, width=10, n_agents=4, density=0.15)),
    ("bottleneck", dict(n_agents=4)),
]


class RandomPolicy:
    """Uniform random actions -- the floor any learner has to clear."""

    def __init__(self, n_actions: int, seed: int = 0):
        self.n_actions = n_actions
        self._random = np.random.default_rng(seed)

    def act(self, observations, deterministic: bool = True):
        return {
            agent: int(self._random.integers(self.n_actions)) for agent in observations
        }


def run_setting(family, kwargs, args):
    print("\n=== %s %s ===" % (family, kwargs))
    env = MAPFEnv(
        family,
        observation="local",
        reward="shaped",
        observation_kwargs={"radius": args.radius},
        seed=args.seed,
        **kwargs,
    )
    print("   %r" % env)

    policies = {"random": RandomPolicy(env.action_space(env.possible_agents[0]).n)}
    curves = {}
    for algorithm in ("ippo", "mappo"):
        started = time.perf_counter()
        trainer = make_trainer(
            algorithm,
            env,
            backend=args.backend,
            n_envs=args.n_envs,
            rollout_steps=args.rollout_steps,
            lr=args.lr,
            seed=args.seed,
        )
        trainer.learn(total_steps=args.steps, verbose=True, log_every=args.log_every)
        elapsed = time.perf_counter() - started
        print(
            "   %s trained: %d steps in %.0fs (%.0f steps/s)"
            % (algorithm, trainer.total_steps, elapsed, trainer.total_steps / elapsed)
        )
        policies[algorithm] = trainer
        curves[algorithm] = [
            {"steps": record["steps"], "solved": record["solved"]}
            for record in trainer.history
        ]

    rows = compare(
        env,
        policies,
        episodes=args.episodes,
        baselines=("cbs", "pibt"),
        baseline_time_limit=args.time_limit,
        seed=args.eval_seed,
    )
    return {
        "family": family,
        "kwargs": kwargs,
        "rows": rows,
        "curves": curves,
    }


def print_table(result) -> None:
    print(
        "\n   %-16s %9s %9s %9s %11s %10s"
        % ("method", "solved", "valid", "cost", "vs optimal", "runtime")
    )
    for row in result["rows"]:
        ratio = row["suboptimality"]
        cost = row["mean_cost"]
        print(
            "   %-16s %8.0f%% %8.0f%% %9s %11s %9.1fms"
            % (
                row["method"],
                100 * row["success_rate"],
                100 * row["validity_rate"],
                "-" if np.isnan(cost) else "%.1f" % cost,
                "-" if np.isnan(ratio) else "%.2fx" % ratio,
                1000 * row["mean_runtime"],
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--backend", default="numpy")
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--time-limit", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--output", default=os.path.join(ROOT, "docs", "assets", "rl-benchmark.json")
    )
    parser.add_argument("--only", default=None, help="run one family by name")
    args = parser.parse_args()

    print(
        "PyMAPF RL benchmark  (backend=%s, %d steps/algorithm)"
        % (args.backend, args.steps)
    )
    results = []
    for family, kwargs in SETTINGS:
        if args.only and family != args.only:
            continue
        result = run_setting(family, kwargs, args)
        print_table(result)
        results.append(result)

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as handle:
            json.dump(
                {
                    "settings": results,
                    "config": vars(args),
                    "version": pymapf.__version__,
                },
                handle,
                indent=2,
            )
        print("\nwrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
