"""Static figures: maps, solutions, congestion, space-time and timelines.

Every function takes an optional ``ax`` so figures compose, and returns the
matplotlib object it drew on. Grids are drawn with row 0 at the top, matching
the ``(row, col)`` convention used everywhere in the library.
"""

from __future__ import annotations

from typing import Dict, Optional

from ..core.grid import GridMap
from ..core.solver import MAPFProblem, Solution
from ..scenarios import Scenario
from . import theme as theme_module

__all__ = [
    "plot_grid",
    "plot_scenario",
    "plot_solution",
    "plot_congestion",
    "plot_spacetime",
    "plot_timeline",
    "compare_solutions",
    "save",
]

# Agents share cells over time, so paths are drawn on slightly offset rails;
# without this a four-agent corridor is a single line and the plot says nothing.
_RAIL = 0.13


def _as_grid(source) -> GridMap:
    if isinstance(source, GridMap):
        return source
    if isinstance(source, (Scenario, MAPFProblem)):
        return source.grid
    raise TypeError(
        "expected a GridMap, Scenario or MAPFProblem, got %r" % type(source)
    )


def _agents_of(source):
    if isinstance(source, (Scenario, MAPFProblem)):
        return list(source.agents)
    return []


def _new_axes(grid: GridMap, ax, theme, figsize=None):
    import matplotlib.pyplot as plt

    if ax is None:
        if figsize is None:
            scale = 0.34
            figsize = (
                max(3.2, grid.width * scale),
                max(3.2, grid.height * scale),
            )
        _, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(theme.surface)
    return ax


def _draw_map(ax, grid: GridMap, theme) -> None:
    """Obstacles as blocks, a hairline lattice, row 0 at the top."""
    from matplotlib.patches import Rectangle

    for r in range(grid.height):
        for c in range(grid.width):
            if not grid.is_free((r, c)):
                ax.add_patch(
                    Rectangle(
                        (c - 0.5, r - 0.5),
                        1,
                        1,
                        facecolor=theme.obstacle,
                        edgecolor="none",
                        zorder=1,
                    )
                )
    for c in range(grid.width + 1):
        ax.axvline(c - 0.5, color=theme.grid, linewidth=0.6, zorder=0)
    for r in range(grid.height + 1):
        ax.axhline(r - 0.5, color=theme.grid, linewidth=0.6, zorder=0)

    ax.set_xlim(-0.5, grid.width - 0.5)
    ax.set_ylim(grid.height - 0.5, -0.5)  # invert: row 0 on top
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_grid(source, ax=None, theme="dark", title: str = "", figsize=None):
    """Draw just the map (obstacles + lattice)."""
    resolved = theme_module.apply(theme)
    grid = _as_grid(source)
    ax = _new_axes(grid, ax, resolved, figsize)
    _draw_map(ax, grid, resolved)
    if title:
        ax.set_title(title, color=resolved.ink, pad=10)
    return ax


def plot_scenario(
    scenario: Scenario, ax=None, theme="dark", title: Optional[str] = None
):
    """Draw a map with each agent's start (filled) and goal (ring)."""
    resolved = theme_module.apply(theme)
    ax = plot_grid(scenario.grid, ax=ax, theme=resolved)
    names = [a.name for a in scenario.agents]
    colors = resolved.color_map(names)
    markers = resolved.marker_map(names)

    for agent in scenario.agents:
        color = colors[agent.name]
        ax.plot(
            agent.start[1],
            agent.start[0],
            marker=markers[agent.name],
            markersize=9,
            color=color,
            markeredgecolor=resolved.surface,
            markeredgewidth=1.5,
            zorder=4,
            linestyle="none",
        )
        ax.plot(
            agent.goal[1],
            agent.goal[0],
            marker=markers[agent.name],
            markersize=9,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=2.0,
            zorder=4,
            linestyle="none",
        )
        ax.annotate(
            agent.name,
            (agent.start[1], agent.start[0]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
            color=resolved.ink_secondary,
            zorder=5,
        )

    ax.set_title(
        title if title is not None else scenario.name.replace("_", " "),
        color=resolved.ink,
        pad=10,
    )
    return ax


def plot_solution(
    solution: Solution,
    source,
    ax=None,
    theme="dark",
    title: Optional[str] = None,
    show_metrics: bool = True,
    label_agents: bool = True,
):
    """Draw the map plus every agent's route, start and goal.

    ``source`` is the :class:`~pymapf.scenarios.Scenario` or
    :class:`~pymapf.core.solver.MAPFProblem` the solution was produced from.
    """
    resolved = theme_module.apply(theme)
    grid = _as_grid(source)
    ax = plot_grid(grid, ax=ax, theme=resolved)

    names = list(solution.paths)
    colors = resolved.color_map(names)
    markers = resolved.marker_map(names)

    for index, name in enumerate(names):
        path = solution.paths[name]
        color = colors[name]
        offset = _RAIL * (index - (len(names) - 1) / 2) / max(1, len(names) / 2)
        xs = [cell[1] + offset for cell in path]
        ys = [cell[0] + offset for cell in path]

        # A wide surface-colored underlay separates crossing rails cleanly.
        ax.plot(
            xs,
            ys,
            color=resolved.surface,
            linewidth=4.5,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2.0,
            solid_capstyle="round",
            solid_joinstyle="round",
            alpha=0.95,
            zorder=3,
        )
        ax.plot(
            xs[0],
            ys[0],
            marker=markers[name],
            markersize=8,
            color=color,
            markeredgecolor=resolved.surface,
            markeredgewidth=1.4,
            linestyle="none",
            zorder=4,
        )
        ax.plot(
            xs[-1],
            ys[-1],
            marker=markers[name],
            markersize=9,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=2.0,
            linestyle="none",
            zorder=4,
        )
        if label_agents:
            ax.annotate(
                name,
                (xs[0], ys[0]),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=8,
                color=resolved.ink_secondary,
                zorder=5,
            )

    heading = title if title is not None else (solution.algorithm or "solution")
    ax.set_title(heading, color=resolved.ink, pad=10)
    if show_metrics:
        ax.set_xlabel(
            "cost %d   ·   makespan %d   ·   %d expansions   ·   %.0f ms"
            % (
                solution.sum_of_costs,
                solution.makespan,
                solution.expansions,
                1000 * solution.runtime,
            ),
            color=resolved.muted,
            fontsize=8.5,
            labelpad=8,
        )
    return ax


def plot_congestion(
    solution: Solution,
    source,
    ax=None,
    theme="dark",
    title: str = "Congestion",
):
    """Heatmap of agent-timesteps per cell -- where the plan queues up."""
    resolved = theme_module.apply(theme)
    grid = _as_grid(source)
    ax = _new_axes(grid, ax, resolved)

    counts = solution.congestion()
    values = [
        [
            float("nan") if not grid.is_free((r, c)) else counts.get((r, c), 0)
            for c in range(grid.width)
        ]
        for r in range(grid.height)
    ]
    cmap = theme_module.sequential_colormap(resolved)
    cmap = cmap.copy()
    cmap.set_bad(resolved.obstacle)
    peak = max([v for row in values for v in row if v == v] + [1])
    image = ax.imshow(values, cmap=cmap, vmin=0, vmax=peak, interpolation="nearest")

    _draw_map(ax, grid, resolved)
    # The lattice is drawn over the heatmap, so re-assert the image on top of
    # nothing but the obstacles by keeping its z-order above the axhlines.
    image.set_zorder(1.5)

    bar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=resolved.muted, labelsize=8)
    bar.set_label("agent-timesteps", color=resolved.ink_secondary, fontsize=8.5)
    ax.set_title(title, color=resolved.ink, pad=10)
    return ax


def plot_spacetime(
    solution: Solution,
    source,
    ax=None,
    theme="dark",
    title: str = "Space-time paths",
):
    """The 3D space-time view: the plane is the map, the vertical axis is time.

    Conflicts are exactly the points where two lines touch in this cube, which
    makes it the most honest picture of what a MAPF solver actually searches.
    """
    import matplotlib.pyplot as plt

    resolved = theme_module.apply(theme)
    grid = _as_grid(source)
    if ax is None:
        figure = plt.figure(figsize=(7, 6))
        ax = figure.add_subplot(111, projection="3d")

    ax.set_facecolor(resolved.plane)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0, 0, 0, 0))
        pane._axinfo["grid"]["color"] = resolved.grid
        pane._axinfo["grid"]["linewidth"] = 0.5

    # Obstacles as a footprint on the floor of the cube.
    obstacle_x = [
        c
        for r in range(grid.height)
        for c in range(grid.width)
        if not grid.is_free((r, c))
    ]
    obstacle_y = [
        r
        for r in range(grid.height)
        for c in range(grid.width)
        if not grid.is_free((r, c))
    ]
    if obstacle_x:
        ax.scatter(
            obstacle_x,
            obstacle_y,
            0,
            marker="s",
            s=28,
            color=resolved.obstacle,
            depthshade=False,
        )

    names = list(solution.paths)
    colors = resolved.color_map(names)
    for name in names:
        path = solution.paths[name]
        ax.plot(
            [cell[1] for cell in path],
            [cell[0] for cell in path],
            list(range(len(path))),
            color=colors[name],
            linewidth=2.0,
            label=name,
        )
        ax.scatter(
            path[-1][1],
            path[-1][0],
            len(path) - 1,
            color=colors[name],
            s=26,
            depthshade=False,
        )
        # With eight agents on one pair of axes, hue alone is not enough to tell
        # two paths apart; the arrival label is the identity channel that is.
        ax.text(
            path[-1][1],
            path[-1][0],
            len(path) - 1 + 0.6,
            name,
            color=resolved.ink_secondary,
            fontsize=8,
            ha="center",
        )

    ax.set_xlabel("col", color=resolved.muted, fontsize=8)
    ax.set_ylabel("row", color=resolved.muted, fontsize=8)
    ax.set_zlabel("time", color=resolved.muted, fontsize=8)
    ax.tick_params(colors=resolved.muted, labelsize=7)
    ax.set_title(title, color=resolved.ink, pad=12)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(-0.06, 1.02),
        fontsize=8,
        ncol=min(8, len(names)),
        columnspacing=1.1,
        handlelength=1.4,
        labelcolor=resolved.ink_secondary,
    )
    ax.view_init(elev=22, azim=-58)
    return ax


def plot_timeline(
    solution: Solution,
    ax=None,
    theme="dark",
    title: str = "Who moves, who waits",
):
    """One row per agent; segments show moving vs waiting vs parked on goal.

    Waiting is where the coordination cost actually shows up, and it is
    invisible in a map view.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    resolved = theme_module.apply(theme)
    names = list(solution.paths)
    horizon = solution.makespan

    if ax is None:
        _, ax = plt.subplots(
            figsize=(max(5.5, 0.32 * horizon + 2), 0.5 * len(names) + 1.6)
        )
    ax.set_facecolor(resolved.surface)
    colors = resolved.color_map(names)

    for row, name in enumerate(names):
        path = solution.paths[name]
        arrival = len(path) - 1
        # Parked tail: drawn faintly so a short path still spans the axis.
        if arrival < horizon:
            ax.add_patch(
                Rectangle(
                    (arrival, row - 0.28),
                    horizon - arrival,
                    0.56,
                    facecolor=colors[name],
                    alpha=0.15,
                    edgecolor="none",
                )
            )
        for t in range(arrival):
            waiting = path[t] == path[t + 1]
            ax.add_patch(
                Rectangle(
                    (t + 0.06, row - 0.28),
                    0.88,
                    0.56,
                    facecolor=resolved.surface if waiting else colors[name],
                    edgecolor=colors[name],
                    linewidth=1.1,
                    alpha=1.0 if not waiting else 0.9,
                    hatch="///" if waiting else None,
                )
            )

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, color=resolved.ink_secondary, fontsize=9)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_xlim(0, max(1, horizon))
    ax.set_xlabel("timestep", color=resolved.ink_secondary, fontsize=9)
    ax.set_title(title, color=resolved.ink, pad=10)
    ax.grid(axis="x", color=resolved.grid, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    waits = sum(
        1
        for path in solution.paths.values()
        for t in range(len(path) - 1)
        if path[t] == path[t + 1]
    )
    ax.annotate(
        "hatched = waiting  (%d wait-steps in this plan)" % waits,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 6),
        textcoords="offset points",
        fontsize=8,
        color=resolved.muted,
    )
    return ax


def compare_solutions(
    solutions: Dict[str, Optional[Solution]],
    source,
    theme="dark",
    suptitle: str = "",
    ncols: int = 2,
):
    """Panel per algorithm on the same instance, on one shared figure."""
    import matplotlib.pyplot as plt

    resolved = theme_module.apply(theme)
    grid = _as_grid(source)
    items = list(solutions.items())
    ncols = min(ncols, len(items)) or 1
    nrows = (len(items) + ncols - 1) // ncols
    scale = 0.30
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            ncols * max(3.4, grid.width * scale),
            nrows * max(3.6, grid.height * scale + 0.7),
        ),
        squeeze=False,
    )

    for index, (label, solution) in enumerate(items):
        ax = axes[index // ncols][index % ncols]
        if solution is None:
            plot_grid(grid, ax=ax, theme=resolved, title=label)
            ax.text(
                0.5,
                0.5,
                "no solution",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=theme_module.STATUS["critical"],
                fontsize=11,
                fontweight="bold",
            )
        else:
            plot_solution(solution, source, ax=ax, theme=resolved, title=label)

    for index in range(len(items), nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")

    figure.tight_layout()
    if suptitle:
        # Sits *above* the laid-out axes rather than inside their margin: with a
        # single wide row, panel titles otherwise collide with it.
        figure.suptitle(
            suptitle, color=resolved.ink, fontsize=14, fontweight="bold", y=1.06
        )
    return figure


def save(figure_or_ax, path: str, dpi: int = 150, transparent: bool = False) -> str:
    """Save a figure (or the figure owning an axes) and return the path."""
    figure = getattr(figure_or_ax, "figure", figure_or_ax)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", transparent=transparent)
    return path
