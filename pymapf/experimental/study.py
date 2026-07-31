"""Run the experiments and report what actually happened.

Four questions, each answered by measurement over many instances rather than by
a single flattering example:

A. Does a static congestion penalty improve PIBT's vertex preference?
B. Does targeting LNS neighbourhoods by delay beat the stock operators?
C. What is the best use of LaCAM's leftover time budget: continuing the search,
   or restarting it?
D. How much does the true-distance heuristic buy the CBS low level?

Every experiment reports the *paired* comparison -- same instance, same seed,
same budget -- because MAPF instance difficulty varies by orders of magnitude
and unpaired means are dominated by which instances happened to be sampled.

    python -m pymapf.experimental.study --output .docs/assets --scale quick
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence

import pymapf
from pymapf import experimental  # noqa: F401  # pylint: disable=unused-import

# ^ imported for its side effect: importing the package registers the x-* solvers.
from pymapf.core.heuristics import true_distance

SCENARIOS = ("warehouse", "random_obstacles", "maze", "bottleneck", "corner_swap")

SCALES = {
    "quick": {"seeds": 4, "agents": (8, 12), "budget": 0.5},
    "full": {"seeds": 10, "agents": (8, 12, 16, 20), "budget": 2.0},
}


def _instances(seeds: int, agents: Sequence[int]):
    for scenario in SCENARIOS:
        for n in agents:
            for seed in range(seeds):
                try:
                    yield pymapf.build_scenario(scenario, n_agents=n, seed=seed)
                except ValueError:
                    continue  # map too small for that many agents


def _ratio_summary(pairs: List[tuple]) -> Dict[str, float]:
    """Paired comparison: per-instance ratio of variant cost to baseline cost."""
    ratios = [variant / base for base, variant in pairs if base and variant]
    if not ratios:
        return {"n": 0}
    wins = sum(1 for r in ratios if r < 0.999)
    losses = sum(1 for r in ratios if r > 1.001)
    return {
        "n": len(ratios),
        "mean_ratio": statistics.fmean(ratios),
        "median_ratio": statistics.median(ratios),
        "best_ratio": min(ratios),
        "worst_ratio": max(ratios),
        "wins": wins,
        "losses": losses,
        "ties": len(ratios) - wins - losses,
    }


# --------------------------------------------------------------------------
# A. congestion-aware PIBT
# --------------------------------------------------------------------------


def experiment_congestion_pibt(
    seeds: int, agents: Sequence[int], alphas=(0.0, 0.1, 0.3, 0.6, 1.0, 2.0)
):
    rows = []
    per_alpha: Dict[float, List[tuple]] = {alpha: [] for alpha in alphas}
    solved: Dict[float, int] = {alpha: 0 for alpha in alphas}
    total = 0

    for scenario in _instances(seeds, agents):
        problem = scenario.to_problem()
        baseline = pymapf.solve(problem, "pibt")
        total += 1
        base_cost = baseline.sum_of_costs if baseline else None
        if baseline:
            solved[0.0] += 0  # counted below via the alpha=0 run
        for alpha in alphas:
            started = time.perf_counter()
            solution = pymapf.solve(problem, "x-pibt-congestion", alpha=alpha)
            runtime = time.perf_counter() - started
            if solution:
                solved[alpha] += 1
            rows.append(
                {
                    "experiment": "congestion_pibt",
                    "scenario": scenario.name,
                    "agents": scenario.n_agents,
                    "alpha": alpha,
                    "solved": solution is not None,
                    "cost": solution.sum_of_costs if solution else None,
                    "runtime": runtime,
                }
            )
            if base_cost and solution:
                per_alpha[alpha].append((base_cost, solution.sum_of_costs))

    summary = {
        "instances": total,
        "by_alpha": {
            str(alpha): {
                **_ratio_summary(pairs),
                "success_rate": solved[alpha] / total if total else 0.0,
            }
            for alpha, pairs in per_alpha.items()
        },
    }
    return rows, summary


# --------------------------------------------------------------------------
# B. delay-targeted LNS
# --------------------------------------------------------------------------


def experiment_delay_lns(seeds: int, agents: Sequence[int], budget: float):
    rows = []
    pairs = []
    for scenario in _instances(seeds, agents):
        problem = scenario.to_problem()
        results = {}
        for algorithm in ("lns", "x-lns-delay"):
            started = time.perf_counter()
            solution = pymapf.solve(problem, algorithm, time_limit=budget, seed=0)
            runtime = time.perf_counter() - started
            results[algorithm] = solution
            rows.append(
                {
                    "experiment": "delay_lns",
                    "scenario": scenario.name,
                    "agents": scenario.n_agents,
                    "algorithm": algorithm,
                    "solved": solution is not None,
                    "cost": solution.sum_of_costs if solution else None,
                    "iterations": solution.expansions if solution else 0,
                    "runtime": runtime,
                }
            )
        if results["lns"] and results["x-lns-delay"]:
            pairs.append(
                (results["lns"].sum_of_costs, results["x-lns-delay"].sum_of_costs)
            )
    return rows, {"budget": budget, **_ratio_summary(pairs)}


# --------------------------------------------------------------------------
# C. what to do with LaCAM's leftover budget
# --------------------------------------------------------------------------


def experiment_lacam_anytime(seeds: int, agents: Sequence[int], budget: float):
    rows = []
    variants = {
        "first solution": dict(algorithm="lacam", kwargs={"time_limit": budget}),
        "anytime (in-search)": dict(
            algorithm="lacam", kwargs={"time_limit": budget, "anytime": True}
        ),
        "restarts": dict(
            algorithm="x-lacam-restart",
            kwargs={"time_limit": budget, "per_run_limit": max(0.05, budget / 8)},
        ),
        "LNS refinement": dict(
            algorithm="lns",
            kwargs={"time_limit": budget, "initial": "lacam"},
        ),
    }
    pairs: Dict[str, List[tuple]] = {label: [] for label in variants}

    for scenario in _instances(seeds, agents):
        problem = scenario.to_problem()
        costs = {}
        for label, spec in variants.items():
            started = time.perf_counter()
            solution = pymapf.solve(problem, spec["algorithm"], **spec["kwargs"])
            runtime = time.perf_counter() - started
            costs[label] = solution.sum_of_costs if solution else None
            rows.append(
                {
                    "experiment": "lacam_anytime",
                    "scenario": scenario.name,
                    "agents": scenario.n_agents,
                    "variant": label,
                    "solved": solution is not None,
                    "cost": costs[label],
                    "runtime": runtime,
                }
            )
        base = costs["first solution"]
        for label in variants:
            if base and costs[label]:
                pairs[label].append((base, costs[label]))

    return rows, {
        "budget": budget,
        "by_variant": {label: _ratio_summary(p) for label, p in pairs.items()},
    }


# --------------------------------------------------------------------------
# D. true-distance heuristic in the CBS low level
# --------------------------------------------------------------------------


def experiment_true_distance(seeds: int, budget: float = 5.0):
    """Manhattan vs exact goal distance, on the maps where walls actually bite."""
    rows = []
    pairs_runtime = []
    pairs_expansions = []

    for scenario_name in ("maze", "random_obstacles", "warehouse"):
        for seed in range(seeds):
            for n in (4, 6):
                try:
                    scenario = pymapf.build_scenario(
                        scenario_name, n_agents=n, seed=seed
                    )
                except ValueError:
                    continue
                problem = scenario.to_problem()

                measured = {}
                for label in ("manhattan", "true_distance"):
                    if label == "manhattan":
                        heuristic = "manhattan"
                    else:
                        # Keyed by goal so dispatch is O(1): looping over the
                        # tables per call would measure our wrapper, not the
                        # heuristic.
                        tables = {
                            agent.goal: true_distance(problem.grid, agent.goal)
                            for agent in problem.agents
                        }

                        def heuristic(cell, goal, _tables=tables):
                            table = _tables.get(goal)
                            if table is None:
                                return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])
                            return table(cell, goal)

                    started = time.perf_counter()
                    solution = pymapf.solve(
                        problem, "cbs", heuristic=heuristic, time_limit=budget
                    )
                    runtime = time.perf_counter() - started
                    measured[label] = (solution, runtime)
                    rows.append(
                        {
                            "experiment": "true_distance",
                            "scenario": scenario_name,
                            "agents": n,
                            "heuristic": label,
                            "solved": solution is not None,
                            "cost": solution.sum_of_costs if solution else None,
                            "expansions": solution.expansions if solution else None,
                            "runtime": runtime,
                        }
                    )

                a, b = measured["manhattan"], measured["true_distance"]
                if a[0] and b[0]:
                    pairs_runtime.append((a[1], b[1]))
                    pairs_expansions.append((a[0].expansions, b[0].expansions))

    return rows, {
        "runtime": _ratio_summary(pairs_runtime),
        "expansions": _ratio_summary(pairs_expansions),
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def run(scale: str = "quick", output: Optional[str] = None) -> Dict:
    config = SCALES[scale]
    seeds, agents, budget = config["seeds"], config["agents"], config["budget"]
    print(
        "Experimental study (%s): %d seeds, agents %s, budget %.2gs"
        % (scale, seeds, agents, budget)
    )

    results: Dict[str, Dict] = {}
    all_rows: List[Dict] = []

    for label, runner in (
        ("A. congestion-aware PIBT", lambda: experiment_congestion_pibt(seeds, agents)),
        ("B. delay-targeted LNS", lambda: experiment_delay_lns(seeds, agents, budget)),
        (
            "C. LaCAM leftover budget",
            lambda: experiment_lacam_anytime(seeds, agents, budget),
        ),
        ("D. true-distance heuristic", lambda: experiment_true_distance(seeds, budget)),
    ):
        print("  running %s..." % label, end="", flush=True)
        started = time.perf_counter()
        rows, summary = runner()
        all_rows.extend(rows)
        results[label] = summary
        print(" %.0fs" % (time.perf_counter() - started))

    if output:
        os.makedirs(output, exist_ok=True)
        with open(os.path.join(output, "experiments.json"), "w") as handle:
            json.dump({"scale": scale, "results": results}, handle, indent=2)
        import csv

        fields = sorted({key for row in all_rows for key in row})
        with open(os.path.join(output, "experiments.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)
        print("  wrote %s/experiments.{json,csv}" % output)

    return results


def _format(results: Dict) -> str:
    lines = []
    for label, summary in results.items():
        lines.append("\n%s" % label)
        lines.append("-" * len(label))
        if "by_alpha" in summary:
            lines.append(
                "  alpha   n   mean ratio  median  best   worst  win/tie/loss  solved"
            )
            for alpha, stats in summary["by_alpha"].items():
                if not stats.get("n"):
                    continue
                lines.append(
                    "  %-6s %3d   %8.3f  %6.3f  %5.3f  %5.3f  %3d/%3d/%3d   %4.0f%%"
                    % (
                        alpha,
                        stats["n"],
                        stats["mean_ratio"],
                        stats["median_ratio"],
                        stats["best_ratio"],
                        stats["worst_ratio"],
                        stats["wins"],
                        stats["ties"],
                        stats["losses"],
                        100 * stats["success_rate"],
                    )
                )
        elif "by_variant" in summary:
            lines.append(
                "  variant                n   mean ratio  median  win/tie/loss"
            )
            for variant, stats in summary["by_variant"].items():
                if not stats.get("n"):
                    continue
                lines.append(
                    "  %-20s %3d   %8.3f  %6.3f  %3d/%3d/%3d"
                    % (
                        variant,
                        stats["n"],
                        stats["mean_ratio"],
                        stats["median_ratio"],
                        stats["wins"],
                        stats["ties"],
                        stats["losses"],
                    )
                )
        elif "runtime" in summary and isinstance(summary["runtime"], dict):
            for metric in ("runtime", "expansions"):
                stats = summary[metric]
                if not stats.get("n"):
                    continue
                lines.append(
                    "  %-11s n=%d  mean ratio %.3f  median %.3f  win/tie/loss %d/%d/%d"
                    % (
                        metric,
                        stats["n"],
                        stats["mean_ratio"],
                        stats["median_ratio"],
                        stats["wins"],
                        stats["ties"],
                        stats["losses"],
                    )
                )
        else:
            if summary.get("n"):
                lines.append(
                    "  n=%d  mean ratio %.3f  median %.3f  win/tie/loss %d/%d/%d"
                    % (
                        summary["n"],
                        summary["mean_ratio"],
                        summary["median_ratio"],
                        summary["wins"],
                        summary["ties"],
                        summary["losses"],
                    )
                )
    lines.append("\n(ratio < 1 means the variant is cheaper than the baseline)")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=sorted(SCALES), default="quick")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    results = run(args.scale, args.output)
    print(_format(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
