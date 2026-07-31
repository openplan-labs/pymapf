"""Scoring learned policies against the planners, on the planners' terms.

This is the part that is hard to do honestly without owning the domain, and
easy once you do. Every learned rollout comes back as a
:class:`pymapf.Solution`, so:

* **validity** is checked by ``Solution.is_valid()`` -- the same conflict
  detector that validates CBS output. A learned policy cannot quietly score
  well by producing paths that overlap;
* **cost** is ``sum_of_costs`` from the same property, on paths trimmed the
  same way;
* and because CBS is *optimal*, the ratio of the two is a **suboptimality
  ratio against ground truth**, not against another heuristic. Most learned-MAPF
  numbers are success rates, because the optimum is expensive to obtain. Here it
  is one call.

The instances are shared: every method is run on the identical problem, so the
comparison never turns on one method getting easier maps. Instances that the
optimal planner cannot solve within its time limit are reported separately
rather than counted as learned-policy wins.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .env import MAPFEnv

__all__ = ["EvaluationResult", "rollout", "evaluate", "compare", "plan"]


class EvaluationResult:
    """Aggregate outcome of running one method over a set of instances."""

    def __init__(self, name: str):
        self.name = name
        self.solved: List[bool] = []
        self.valid: List[bool] = []
        self.costs: List[Optional[int]] = []
        self.makespans: List[Optional[int]] = []
        self.collisions: List[int] = []
        self.runtimes: List[float] = []
        self.ratios: List[float] = []

    def add(self, solved, valid, cost, makespan, collisions, runtime, ratio=None):
        self.solved.append(bool(solved))
        self.valid.append(bool(valid))
        self.costs.append(cost)
        self.makespans.append(makespan)
        self.collisions.append(int(collisions))
        self.runtimes.append(float(runtime))
        if ratio is not None:
            self.ratios.append(float(ratio))

    def summary(self) -> Dict[str, float]:
        n = max(1, len(self.solved))
        solved_costs = [cost for cost in self.costs if cost is not None]
        return {
            "method": self.name,
            "instances": len(self.solved),
            "success_rate": sum(self.solved) / n,
            "validity_rate": sum(self.valid) / n,
            # Averaged over solved instances only: averaging a cost over runs
            # that never finished would be meaningless, and averaging it as
            # zero would reward failure.
            "mean_cost": float(np.mean(solved_costs)) if solved_costs else float("nan"),
            "mean_makespan": (
                float(np.mean([m for m in self.makespans if m is not None]))
                if any(m is not None for m in self.makespans)
                else float("nan")
            ),
            "mean_collisions": float(np.mean(self.collisions)) if self.collisions else 0.0,
            "mean_runtime": float(np.mean(self.runtimes)) if self.runtimes else 0.0,
            "suboptimality": float(np.mean(self.ratios)) if self.ratios else float("nan"),
        }

    def __repr__(self) -> str:
        summary = self.summary()
        return "%s: %.0f%% solved, cost %.1f, %.2fx optimal" % (
            summary["method"],
            100 * summary["success_rate"],
            summary["mean_cost"],
            summary["suboptimality"],
        )


def rollout(env: MAPFEnv, policy, deterministic: bool = True, seed: Optional[int] = None):
    """Run one episode under ``policy``; return its :class:`Solution` and summary.

    ``policy`` is anything with ``act(observations) -> {agent: action}``, which
    is what the trainers expose, so an untrained baseline is a two-line class.
    """
    observations, _ = env.reset(seed=seed)
    while env.agents:
        actions = _ask(policy, observations, deterministic)
        observations, _, terminations, truncations, _ = env.step(actions)
        if any(terminations.values()) or any(truncations.values()):
            break
    return env.solution(), env.episode_summary()


#: What a planner can hit on an instance it cannot handle, as opposed to what a
#: caller can get wrong. Deliberately narrow: a bare ``except Exception`` also
#: swallows the ``ValueError`` from asking for a solver that does not exist, and
#: reporting a typo as "the planner found this unsolvable" is worse than
#: crashing.
PLANNER_FAILURES = (RuntimeError, MemoryError, RecursionError)


def plan(problem, algorithm: str, time_limit: float = 5.0):
    """Run a planner, returning ``None`` if it cannot solve the instance.

    The solver is *constructed first*, outside the guard, so an unknown name
    raises rather than being absorbed. Only the search itself is guarded, and
    only against the ways a search can genuinely give out. Also absorbs the one
    interface wrinkle in the registry: not every solver takes ``time_limit``,
    and the ones that do not signal it with a ``TypeError`` from construction
    rather than advertising it.
    """
    import pymapf

    try:
        solver = pymapf.get_solver(algorithm, time_limit=time_limit)
    except TypeError:
        solver = pymapf.get_solver(algorithm)

    try:
        solution = solver.solve(problem)
    except PLANNER_FAILURES:
        return None
    return solution if solution is not None and solution.is_valid() else None


def _ask(policy, observations, deterministic):
    if hasattr(policy, "act"):
        try:
            return policy.act(observations, deterministic=deterministic)
        except TypeError:
            return policy.act(observations)
    return policy(observations)


def evaluate(
    env: MAPFEnv,
    policy,
    episodes: int = 50,
    baseline: Optional[str] = "cbs",
    baseline_time_limit: float = 5.0,
    deterministic: bool = True,
    seed: int = 0,
) -> Dict[str, EvaluationResult]:
    """Run ``policy`` and (optionally) a planner on the same instances.

    Returns one :class:`EvaluationResult` per method, keyed by name. The
    learned policy's ``suboptimality`` is populated only for instances where
    *both* it and the optimal baseline succeeded -- the only instances on which
    the ratio means anything.
    """
    results = {"policy": EvaluationResult("policy")}
    if baseline:
        results[baseline] = EvaluationResult(baseline)

    for episode in range(episodes):
        instance_seed = seed + episode
        started = time.perf_counter()
        solution, summary = rollout(env, policy, deterministic, seed=instance_seed)
        learned_runtime = time.perf_counter() - started
        learned_valid = solution.is_valid()
        learned_solved = bool(summary["solved"]) and learned_valid
        learned_cost = solution.sum_of_costs if learned_solved else None

        optimal_cost = None
        if baseline:
            started = time.perf_counter()
            planned = plan(env.problem, baseline, baseline_time_limit)
            planner_runtime = time.perf_counter() - started
            found = planned is not None
            optimal_cost = planned.sum_of_costs if found else None
            results[baseline].add(
                solved=found,
                valid=bool(found),
                cost=optimal_cost,
                makespan=planned.makespan if found else None,
                collisions=0,
                runtime=planner_runtime,
            )

        ratio = None
        if learned_cost is not None and optimal_cost:
            ratio = learned_cost / optimal_cost
        results["policy"].add(
            solved=learned_solved,
            valid=learned_valid,
            cost=learned_cost,
            makespan=solution.makespan if learned_solved else None,
            collisions=summary["collisions"],
            runtime=learned_runtime,
            ratio=ratio,
        )
    return results


def compare(
    env: MAPFEnv,
    policies: Dict[str, object],
    episodes: int = 50,
    baselines: Sequence[str] = ("cbs", "pibt"),
    baseline_time_limit: float = 5.0,
    seed: int = 0,
    modes: Sequence[str] = ("greedy", "sampled"),
) -> List[Dict[str, float]]:
    """Score several policies and several planners on one shared instance set.

    The instances are drawn from ``env``'s scenario family by seed, and each
    method sees exactly the same seeds, so nothing in the table turns on one
    method having been handed easier maps.

    Each learned policy is scored in **both** action modes, because on this
    domain the choice is not a detail -- it is the single largest effect in the
    table. Taking the argmax makes the policy deterministic, and two
    deterministic agents that both want the same cell are refused, revert, and
    then choose exactly the same thing again: a livelock, and the direct
    analogue of the one PIBT suffers. Sampling breaks the symmetry, so the same
    weights go from solving 47% of instances to solving 100% of them -- at
    three times the cost, because a sampled policy also wanders. Reporting only
    one of those two numbers would be reporting half the result.
    """
    rows: List[Dict[str, float]] = []
    labelled = {
        ("%s (%s)" % (name, mode) if len(modes) > 1 else name): (policy, mode == "greedy")
        for name, policy in policies.items()
        for mode in modes
    }
    tallies = {name: EvaluationResult(name) for name in labelled}
    tallies.update({name: EvaluationResult(name) for name in baselines})
    optimal: Dict[int, Optional[int]] = {}

    # The optimal costs first, so every learned row can be scored against them.
    for episode in range(episodes):
        env.reset(seed=seed + episode)
        started = time.perf_counter()
        planned = plan(env.problem, "cbs", baseline_time_limit)
        optimal[episode] = planned.sum_of_costs if planned is not None else None
        if "cbs" in tallies:
            found = planned is not None
            tallies["cbs"].add(
                found,
                bool(found),
                optimal[episode],
                planned.makespan if found else None,
                0,
                time.perf_counter() - started,
                1.0 if found else None,
            )

    for name in baselines:
        if name == "cbs":
            continue
        for episode in range(episodes):
            env.reset(seed=seed + episode)
            started = time.perf_counter()
            planned = plan(env.problem, name, baseline_time_limit)
            found = planned is not None
            cost = planned.sum_of_costs if found else None
            reference = optimal.get(episode)
            tallies[name].add(
                found,
                bool(found),
                cost,
                planned.makespan if found else None,
                0,
                time.perf_counter() - started,
                (cost / reference) if (cost and reference) else None,
            )

    for name, (policy, deterministic) in labelled.items():
        for episode in range(episodes):
            started = time.perf_counter()
            solution, summary = rollout(env, policy, deterministic, seed=seed + episode)
            elapsed = time.perf_counter() - started
            valid = solution.is_valid()
            solved = bool(summary["solved"]) and valid
            cost = solution.sum_of_costs if solved else None
            reference = optimal.get(episode)
            tallies[name].add(
                solved,
                valid,
                cost,
                solution.makespan if solved else None,
                summary["collisions"],
                elapsed,
                (cost / reference) if (cost and reference) else None,
            )

    for name in list(labelled) + list(baselines):
        rows.append(tallies[name].summary())
    return rows
