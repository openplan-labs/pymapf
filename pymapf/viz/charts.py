"""Benchmark charts: how algorithms compare on cost, effort and time.

These consume :class:`~pymapf.benchmark.BenchmarkReport` objects. Series are
algorithms, so each keeps a fixed categorical color across every chart in a
report -- the same algorithm is the same color in the scaling curve, the cost
bars and the dashboard.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from ..benchmark import BenchmarkReport, aggregate
from . import theme as theme_module

__all__ = [
    "plot_scaling",
    "plot_cost_comparison",
    "plot_success_rate",
    "plot_cost_curve",
    "dashboard",
]

_Y_LABELS = {
    "runtime": "runtime (s)",
    "sum_of_costs": "sum of costs",
    "makespan": "makespan",
    "expansions": "high-level expansions",
}


def _style_axes(ax, resolved, xlabel="", ylabel="", title=""):
    ax.set_facecolor(resolved.surface)
    ax.grid(axis="y", color=resolved.grid, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel, color=resolved.ink_secondary, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=resolved.ink_secondary, fontsize=9)
    if title:
        ax.set_title(title, color=resolved.ink, pad=10)
    return ax


def _algorithm_colors(algorithms: Sequence[str], resolved) -> Dict[str, str]:
    return {name: resolved.agent_color(i) for i, name in enumerate(sorted(algorithms))}


def plot_scaling(
    report: BenchmarkReport,
    y: str = "runtime",
    x: str = "n_agents",
    ax=None,
    theme="dark",
    title: Optional[str] = None,
    log_y: bool = True,
):
    """Mean ``y`` against ``x`` per algorithm -- the scaling picture.

    Runtime spans orders of magnitude between a greedy and an optimal solver, so
    the y axis is logarithmic by default; a linear axis would flatten the fast
    solver into the baseline.
    """
    import matplotlib.pyplot as plt

    resolved = theme_module.apply(theme)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.2))

    data = aggregate(report, x=x, y=y)
    colors = _algorithm_colors(list(data), resolved)
    for name in sorted(data):
        xs, ys = data[name]
        ax.plot(
            xs,
            ys,
            color=colors[name],
            linewidth=2.0,
            marker="o",
            markersize=7,
            markeredgecolor=resolved.surface,
            markeredgewidth=1.4,
            label=name,
        )
        if xs:  # direct label at the end of the line, in ink not series color
            ax.annotate(
                name,
                (xs[-1], ys[-1]),
                textcoords="offset points",
                xytext=(8, 0),
                fontsize=9,
                color=resolved.ink_secondary,
                va="center",
            )

    if log_y and any(v > 0 for _, values in data.values() for v in values):
        ax.set_yscale("log")
    _style_axes(
        ax,
        resolved,
        xlabel="agents" if x == "n_agents" else x,
        ylabel=_Y_LABELS.get(y, y),
        title=title or "Scaling: %s" % _Y_LABELS.get(y, y),
    )
    if len(data) > 1:
        ax.legend(loc="upper left", fontsize=9, labelcolor=resolved.ink_secondary)
    ax.margins(x=0.12)
    return ax


def plot_cost_comparison(
    report: BenchmarkReport,
    metric: str = "sum_of_costs",
    ax=None,
    theme="dark",
    title: Optional[str] = None,
):
    """Grouped bars: one group per scenario, one bar per algorithm.

    Unsolved runs get a hatched placeholder rather than a missing bar, because
    "the greedy solver failed here" is the most interesting cell in the chart.
    """
    import matplotlib.pyplot as plt

    resolved = theme_module.apply(theme)
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 4.2))

    scenarios = report.scenarios
    algorithms = report.algorithms
    colors = _algorithm_colors(algorithms, resolved)
    slot = 0.8 / max(1, len(algorithms))

    lookup = {(row.scenario, row.algorithm): row for row in report.rows}
    for index, algorithm in enumerate(algorithms):
        for position, scenario in enumerate(scenarios):
            row = lookup.get((scenario, algorithm))
            left = position - 0.4 + index * slot
            if row is None:
                continue
            value = getattr(row, metric) if row.solved else 0
            if row.solved and value is not None:
                ax.bar(
                    left + slot / 2,
                    value,
                    width=slot * 0.86,  # the 14% gap is the 2px surface spacer
                    color=colors[algorithm],
                    edgecolor="none",
                    label=algorithm if position == 0 else None,
                )
                ax.annotate(
                    str(value),
                    (left + slot / 2, value),
                    textcoords="offset points",
                    xytext=(0, 4),
                    ha="center",
                    fontsize=8,
                    color=resolved.ink_secondary,
                )
            else:
                ax.bar(
                    left + slot / 2,
                    1,
                    width=slot * 0.86,
                    facecolor="none",
                    edgecolor=theme_module.STATUS["critical"],
                    hatch="///",
                    linewidth=1.2,
                    label=algorithm if position == 0 else None,
                )
                ax.annotate(
                    "failed",
                    (left + slot / 2, 1),
                    textcoords="offset points",
                    xytext=(0, 4),
                    ha="center",
                    fontsize=8,
                    color=theme_module.STATUS["critical"],
                )

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(
        [s.replace("_", " ") for s in scenarios],
        color=resolved.ink_secondary,
        fontsize=9,
        rotation=0 if len(scenarios) <= 4 else 20,
        ha="center" if len(scenarios) <= 4 else "right",
    )
    _style_axes(
        ax,
        resolved,
        ylabel=_Y_LABELS.get(metric, metric),
        title=title or "Solution quality by scenario",
    )
    if len(algorithms) > 1:
        ax.legend(loc="upper left", fontsize=9, labelcolor=resolved.ink_secondary)
    return ax


def plot_success_rate(report: BenchmarkReport, ax=None, theme="dark", title=None):
    """Share of instances each algorithm solved -- completeness, measured."""
    import matplotlib.pyplot as plt

    resolved = theme_module.apply(theme)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.4))

    algorithms = report.algorithms
    colors = _algorithm_colors(algorithms, resolved)
    rates = [100 * report.success_rate(name) for name in algorithms]
    ax.barh(
        range(len(algorithms)),
        rates,
        height=0.55,
        color=[colors[name] for name in algorithms],
    )
    for index, rate in enumerate(rates):
        ax.annotate(
            "%.0f%%" % rate,
            (rate, index),
            textcoords="offset points",
            xytext=(6, 0),
            va="center",
            fontsize=9,
            color=resolved.ink_secondary,
        )
    ax.set_yticks(range(len(algorithms)))
    ax.set_yticklabels(algorithms, color=resolved.ink_secondary, fontsize=9)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    _style_axes(ax, resolved, xlabel="instances solved (%)", title=title or "Success rate")
    ax.grid(axis="x", color=resolved.grid, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    return ax


def plot_cost_curve(traces: Dict[str, object], ax=None, theme="dark", title=None):
    """Cost of each expanded node, per recorded :class:`SearchTrace`.

    A flat line means the optimal cost was reached at the root and CBS spent its
    effort proving feasibility; a rising line means it had to pay for conflicts.
    """
    import matplotlib.pyplot as plt

    resolved = theme_module.apply(theme)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.8))

    colors = _algorithm_colors(list(traces), resolved)
    for label in sorted(traces):
        curve = traces[label].cost_curve()
        if not curve:
            continue
        ax.plot(
            range(1, len(curve) + 1),
            curve,
            color=colors[label],
            linewidth=2.0,
            label=label,
        )
        ax.annotate(
            label,
            (len(curve), curve[-1]),
            textcoords="offset points",
            xytext=(8, 0),
            fontsize=9,
            color=resolved.ink_secondary,
            va="center",
        )
    _style_axes(
        ax,
        resolved,
        xlabel="expanded node",
        ylabel="sum of costs",
        title=title or "Search progress",
    )
    if len(traces) > 1:
        ax.legend(loc="lower right", fontsize=9, labelcolor=resolved.ink_secondary)
    ax.margins(x=0.12)
    return ax


def dashboard(
    scaling: BenchmarkReport,
    comparison: Optional[BenchmarkReport] = None,
    theme="dark",
    suptitle: str = "PyMAPF benchmark",
):
    """A four-panel summary: runtime, expansions, quality and success rate."""
    import matplotlib.pyplot as plt

    resolved = theme_module.apply(theme)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.6))
    figure.patch.set_facecolor(resolved.plane)

    plot_scaling(scaling, y="runtime", ax=axes[0][0], theme=resolved)
    plot_scaling(
        scaling,
        y="expansions",
        ax=axes[0][1],
        theme=resolved,
        title="Search effort",
        log_y=True,
    )
    plot_cost_comparison(comparison or scaling, ax=axes[1][0], theme=resolved)
    plot_success_rate(scaling, ax=axes[1][1], theme=resolved)

    figure.suptitle(suptitle, color=resolved.ink, fontsize=15, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return figure
