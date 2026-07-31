#!/usr/bin/env python3
"""Render every figure the docs site shows, from the library itself.

Nothing here is bespoke plotting code: the gallery is the public
:mod:`pymapf.viz` API, called the way a user would call it. If a figure in the
gallery looks good, the function that produced it is available to everyone.

    python scripts/generate_gallery.py [--output .docs/assets] [--fast]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pymapf  # noqa: E402
from pymapf import viz  # noqa: E402
from pymapf.benchmark import compare_algorithms, scaling_study  # noqa: E402

THEME = "dark"


def _step(label: str):
    print("  %-22s" % label, end="", flush=True)
    return time.perf_counter()


def _done(started: float, path: str) -> None:
    print("%6.1fs  ->  %s" % (time.perf_counter() - started, os.path.basename(path)))


def scenario_sheet(output: str) -> str:
    """One panel per scenario family: the library's instance vocabulary."""
    names = pymapf.available_scenarios()
    resolved = viz.apply(THEME)
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, name in zip(axes.flat, names):
        scenario = pymapf.build_scenario(name, seed=1)
        viz.plot_scenario(scenario, ax=ax, theme=THEME)
        ax.set_xlabel(
            scenario.description, color=resolved.muted, fontsize=8.5, labelpad=6
        )
    for ax in axes.flat[len(names) :]:
        ax.axis("off")
    figure.suptitle(
        "Six reproducible scenario families",
        color=resolved.ink,
        fontsize=16,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(output, "scenarios.png")
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=os.path.join(ROOT, ".docs", "assets"))
    parser.add_argument(
        "--fast",
        action="store_true",
        help="skip the animated GIFs (the slow part)",
    )
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    out = lambda name: os.path.join(args.output, name)  # noqa: E731

    print("Rendering gallery into %s" % args.output)

    scenario = pymapf.build_scenario("warehouse", n_agents=8, seed=3)
    problem = scenario.to_problem()

    trace = pymapf.SearchTrace()
    optimal = pymapf.solve(problem, "cbs", observer=trace, time_limit=20)
    bounded = pymapf.solve(problem, "wcbs", weight=1.5)
    greedy = pymapf.solve(problem, "prioritized")
    reference = optimal or bounded

    started = _step("scenarios")
    _done(started, scenario_sheet(args.output))

    started = _step("solution")
    _done(
        started,
        viz.save(
            viz.plot_solution(reference, scenario, theme=THEME), out("solution.png")
        ),
    )

    started = _step("congestion")
    _done(
        started,
        viz.save(
            viz.plot_congestion(reference, scenario, theme=THEME), out("congestion.png")
        ),
    )

    started = _step("space-time")
    _done(
        started,
        viz.save(
            viz.plot_spacetime(reference, scenario, theme=THEME), out("spacetime.png")
        ),
    )

    started = _step("timeline")
    _done(
        started,
        viz.save(viz.plot_timeline(reference, theme=THEME), out("timeline.png")),
    )

    started = _step("comparison")
    figure = viz.compare_solutions(
        {
            "CBS · optimal": optimal,
            "Weighted CBS · w=1.5": bounded,
            "Prioritized · greedy": greedy,
        },
        scenario,
        theme=THEME,
        ncols=3,
        suptitle="Same instance, three trade-offs",
    )
    figure.savefig(out("comparison.png"), dpi=120, bbox_inches="tight")
    plt.close(figure)
    _done(started, out("comparison.png"))

    started = _step("search progress")
    traces = {"cbs": trace}
    _done(
        started,
        viz.save(viz.plot_cost_curve(traces, theme=THEME), out("cost-curve.png")),
    )

    started = _step("benchmark dashboard")
    scaling = scaling_study(
        "random_obstacles",
        agent_counts=(2, 4, 6, 8, 10, 12),
        algorithms=("cbs", "wcbs", "prioritized"),
        seeds=(0, 1, 2),
        solver_kwargs={"time_limit": 2.0},
    )
    comparison = compare_algorithms(
        ["corner_swap", "empty_room", "warehouse", "random_obstacles"],
        ["cbs", "wcbs", "prioritized"],
        time_limit=2.0,
    )
    figure = viz.dashboard(scaling, comparison, theme=THEME)
    figure.savefig(out("dashboard.png"), dpi=110, bbox_inches="tight")
    plt.close(figure)
    scaling.to_csv(out("benchmark.csv"))
    _done(started, out("dashboard.png"))

    if not args.fast:
        started = _step("animated plan")
        animation = viz.animate_solution(reference, scenario, theme=THEME, trail=10)
        _done(
            started,
            viz.save_animation(animation, out("animated-plan.gif"), fps=16, dpi=90),
        )

        started = _step("animated search")
        animation = viz.animate_search(trace, scenario, theme=THEME, max_nodes=24)
        _done(
            started,
            viz.save_animation(animation, out("animated-search.gif"), fps=6, dpi=90),
        )

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
