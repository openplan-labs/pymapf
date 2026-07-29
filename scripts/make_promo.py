#!/usr/bin/env python3
"""Render the PyMAPF promo film.

A scripted sequence of scenes -- title, the project's origin, the framework it
became, a live conflict-based search, the plan executing, the benchmark, and a
sign-off -- rendered with matplotlib and muxed by ffmpeg. Everything on screen
is produced by the library: the maps come from :mod:`pymapf.scenarios`, the
search is a real recorded :class:`~pymapf.core.trace.SearchTrace`, and the
benchmark numbers are measured while the script runs.

    python scripts/make_promo.py --output docs/assets/pymapf-promo.mp4
    python scripts/make_promo.py --gif        # also write a looping GIF
    python scripts/make_promo.py --preview 6  # render 6 seconds, for iteration
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pymapf  # noqa: E402
from pymapf.benchmark import scaling_study, aggregate  # noqa: E402
from pymapf.viz import theme as theme_module  # noqa: E402

FPS = 24
WIDTH, HEIGHT = 16, 9  # inches at dpi=80 -> 1280x720

THEME = theme_module.DARK
MONO = ["DejaVu Sans Mono", "monospace"]

# Everything is drawn in axes coordinates on a 16:9 canvas, so a "circle" of
# radius r is 16r wide and 9r tall unless the height is scaled up to match.
ASPECT = WIDTH / HEIGHT


# --------------------------------------------------------------------------
# easing and small drawing helpers
# --------------------------------------------------------------------------


def ease_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def fade(ax, t: float, start: float, duration: float = 0.45) -> float:
    """Opacity for an element appearing at ``start`` seconds into a scene."""
    return ease_out((t - start) / duration)


def text(ax, x, y, body, size=22, color=None, weight="normal", alpha=1.0, family=None, ha="left", va="center"):
    if alpha <= 0.01:
        return None
    return ax.text(
        x,
        y,
        body,
        fontsize=size,
        color=color or THEME.ink,
        fontweight=weight,
        alpha=min(1.0, alpha),
        ha=ha,
        va=va,
        family=family,
        transform=ax.transAxes,
        zorder=20,
    )


def draw_map(ax, grid, origin, cell, alpha=1.0, lattice=True):
    """Draw a grid map in axes coordinates anchored at ``origin``.

    ``cell`` is a *width* in axes units; the height is scaled by ASPECT so cells
    come out square on the 16:9 canvas.
    """
    ox, oy = origin
    tall = cell * ASPECT
    if lattice:
        for c in range(grid.width + 1):
            ax.plot(
                [ox + c * cell, ox + c * cell],
                [oy, oy + grid.height * tall],
                color=THEME.grid,
                linewidth=0.6,
                alpha=alpha,
                transform=ax.transAxes,
                solid_capstyle="butt",
            )
        for r in range(grid.height + 1):
            ax.plot(
                [ox, ox + grid.width * cell],
                [oy + r * tall, oy + r * tall],
                color=THEME.grid,
                linewidth=0.6,
                alpha=alpha,
                transform=ax.transAxes,
                solid_capstyle="butt",
            )
    for r in range(grid.height):
        for c in range(grid.width):
            if not grid.is_free((r, c)):
                ax.add_patch(
                    Rectangle(
                        (ox + c * cell, oy + (grid.height - 1 - r) * tall),
                        cell,
                        tall,
                        facecolor=THEME.obstacle,
                        edgecolor="none",
                        alpha=alpha,
                        transform=ax.transAxes,
                    )
                )


def cell_xy(grid, origin, cell, position):
    """Map a (row, col) -- possibly fractional -- to axes coordinates."""
    ox, oy = origin
    row, col = position
    return (
        ox + (col + 0.5) * cell,
        oy + (grid.height - 1 - row + 0.5) * cell * ASPECT,
    )


def draw_path(ax, grid, origin, cell, path, color, alpha=1.0, width=2.2, upto=None):
    points = path if upto is None else path[: max(2, upto)]
    if len(points) < 2:
        return
    xs, ys = zip(*[cell_xy(grid, origin, cell, p) for p in points])
    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=width,
        alpha=alpha,
        solid_capstyle="round",
        solid_joinstyle="round",
        transform=ax.transAxes,
        zorder=3,
    )


def disc(ax, x, y, radius, **kwargs):
    """An actually-round mark in axes coordinates."""
    return Ellipse((x, y), 2 * radius, 2 * radius * ASPECT, transform=ax.transAxes, **kwargs)


def draw_agent(ax, grid, origin, cell, position, color, radius=0.4, alpha=1.0, label=None):
    x, y = cell_xy(grid, origin, cell, position)
    ax.add_patch(
        disc(
            ax,
            x,
            y,
            cell * radius,
            facecolor=color,
            edgecolor=THEME.surface,
            linewidth=1.4,
            alpha=alpha,
            zorder=5,
        )
    )
    if label and cell > 0.018:
        ax.text(
            x,
            y,
            label,
            fontsize=9,
            color=THEME.surface,
            ha="center",
            va="center",
            fontweight="bold",
            alpha=alpha,
            transform=ax.transAxes,
            zorder=6,
        )


def interpolate(path, t):
    whole = int(t)
    frac = t - whole
    a = path[min(whole, len(path) - 1)]
    b = path[min(whole + 1, len(path) - 1)]
    k = ease_in_out(frac)
    return (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k)


def typewriter(body: str, progress: float) -> str:
    """Reveal ``body`` character by character, keeping whole lines stable."""
    total = len(body)
    return body[: int(total * max(0.0, min(1.0, progress)))]


# --------------------------------------------------------------------------
# content prepared once, then replayed by the scenes
# --------------------------------------------------------------------------


class Content:
    def __init__(self):
        print("  preparing content...")
        self.hero = pymapf.build_scenario("warehouse", n_agents=8, seed=3)
        self.hero_solution = pymapf.solve(self.hero.to_problem(), "wcbs", weight=1.5)

        self.search_scenario = pymapf.build_scenario("corner_swap", n_agents=6)
        self.search_trace = pymapf.SearchTrace()
        self.search_solution = pymapf.solve(
            self.search_scenario.to_problem(),
            "cbs",
            observer=self.search_trace,
            time_limit=20,
        )
        self.search_events = [
            event
            for event in self.search_trace
            if event.kind in ("expand", "conflict", "solved")
        ]

        self.old = pymapf.build_scenario("empty_room", height=10, width=10, n_agents=3, seed=2)
        self.old_solution = pymapf.solve(self.old.to_problem(), "prioritized")

        # The on-screen snippet quotes what this very run measured, so the film
        # can never drift away from the library's actual behaviour.
        snippet_trace = pymapf.SearchTrace()
        snippet_solution = pymapf.solve(
            self.hero.to_problem(), "wcbs", observer=snippet_trace, weight=1.5
        )
        summary = snippet_trace.summary()
        self.snippet = (
            "import pymapf\n\n"
            'scenario = pymapf.build_scenario("warehouse", n_agents=8, seed=3)\n'
            "trace    = pymapf.SearchTrace()\n\n"
            'solution = pymapf.solve(scenario.to_problem(), "wcbs",\n'
            "                        observer=trace)\n\n"
            "solution.sum_of_costs   # %d\n"
            "solution.is_valid()     # %s\n"
            "trace.summary()         # {'expansions': %d, 'conflicts': %d, ...}"
            % (
                snippet_solution.sum_of_costs,
                snippet_solution.is_valid(),
                summary["expansions"],
                summary["conflicts"],
            )
        )

        print("  running swarm controllers...")
        from pymapf.swarm import SwarmSimulator, SwarmParams

        # The 2024 minimalistic model: relative range and bearing only.
        self.flock = SwarmSimulator(
            "minimalistic", n_agents=22, dimension=2, params=SwarmParams(seed=2)
        ).run(steps=420)
        self.flock_order = self.flock.metrics.summary()["order"]

        # A formation forming up, with its error curve recorded as it goes.
        formation = SwarmSimulator(
            "displacement_formation", n_agents=9, dimension=2,
            shape="v", spacing=3.0, params=SwarmParams(seed=1),
        )
        self.formation_error = []
        self.formation = formation.run(
            steps=260,
            observer=lambda step, state: self.formation_error.append(
                formation.behavior.error(state)
            ),
        )
        self.formation_shape = formation.behavior.shape
        # Which agent holds which slot. Struts drawn without this connect
        # agents by *slot* index, which is only right if nobody was reassigned.
        self.formation_slots = formation.behavior.assignment(
            self.formation.final,
            formation.behavior.shape.centred(self.formation.final.n, 2),
        ).copy()

        print("  measuring benchmark...")
        self.report = scaling_study(
            "random_obstacles",
            agent_counts=(2, 4, 6, 8, 10, 12),
            algorithms=("cbs", "wcbs", "prioritized"),
            seeds=(0, 1, 2),
            solver_kwargs={"time_limit": 2.0},
        )
        self.runtime = aggregate(self.report, "n_agents", "runtime")
        self.success = {
            name: 100 * self.report.success_rate(name) for name in self.report.algorithms
        }


# --------------------------------------------------------------------------
# scenes
# --------------------------------------------------------------------------


def scene_title(ax, t, content):
    """Agents drift across a dark plane while the title resolves."""
    grid = content.hero.grid
    cell = 0.035
    # Centred so the title sits on a symmetric field rather than beside a box.
    origin = (0.5 - grid.width * cell / 2, 0.5 - grid.height * cell * ASPECT / 2)
    draw_map(ax, grid, origin, cell, alpha=0.13 * ease_out(t / 1.2), lattice=True)

    paths = content.hero_solution.paths
    names = list(paths)
    for index, name in enumerate(names):
        color = THEME.agent_color(index)
        progress = min(len(paths[name]) - 1, t * 3.0)
        draw_path(ax, grid, origin, cell, paths[name], color,
                  alpha=0.28 * ease_out(t / 1.0), upto=int(progress) + 2)
        draw_agent(ax, grid, origin, cell, interpolate(paths[name], progress), color,
                   radius=0.3, alpha=0.5 * ease_out(t / 0.8))

    text(ax, 0.5, 0.60, "PyMAPF", size=88, weight="bold", ha="center",
         alpha=fade(ax, t, 0.35, 0.7))
    text(ax, 0.5, 0.48, "multi-agent path finding you can watch",
         size=25, color=THEME.ink_secondary, ha="center", alpha=fade(ax, t, 0.9, 0.7))
    text(ax, 0.5, 0.38, "apla-toolbox/pymapf", size=15, color=THEME.muted,
         ha="center", family=MONO, alpha=fade(ax, t, 1.4, 0.7))


def scene_origin(ax, t, content):
    """Where the project came from -- honest about what it used to be."""
    text(ax, 0.08, 0.86, "2020", size=20, color=THEME.muted, family=MONO,
         alpha=fade(ax, t, 0.0))
    text(ax, 0.08, 0.77, "It started as a university project.", size=38, weight="bold",
         alpha=fade(ax, t, 0.15))

    lines = [
        ("one algorithm, hard-coded", THEME.muted),
        ("random maps, no seed, no way to reproduce a run", THEME.muted),
        ("a global HEURISTIC flag", THEME.muted),
        ("search that could loop forever", THEME.muted),
    ]
    for index, (line, color) in enumerate(lines):
        alpha = fade(ax, t, 0.7 + 0.28 * index)
        text(ax, 0.10, 0.62 - 0.075 * index, "—", size=17, color=THEME.axis, alpha=alpha)
        text(ax, 0.14, 0.62 - 0.075 * index, line, size=19, color=color, alpha=alpha)

    grid = content.old.grid
    cell = 0.028
    origin = (0.66, 0.27)
    alpha = fade(ax, t, 0.4, 0.8)
    draw_map(ax, grid, origin, cell, alpha=alpha)
    for index, (name, path) in enumerate(content.old_solution.paths.items()):
        color = THEME.agent_color(index)
        draw_path(ax, grid, origin, cell, path, color, alpha=0.75 * alpha, width=2.0)
        draw_agent(ax, grid, origin, cell, path[min(len(path) - 1, int(t * 2.4))],
                   color, alpha=alpha, label=name)


def scene_rebuild(ax, t, content):
    """The rewrite: a framework with a vocabulary, typed out on screen."""
    text(ax, 0.08, 0.86, "2026", size=20, color=THEME.muted, family=MONO,
         alpha=fade(ax, t, 0.0))
    text(ax, 0.08, 0.77, "Rebuilt as a framework.", size=38, weight="bold",
         alpha=fade(ax, t, 0.1))

    body = typewriter(content.snippet, (t - 0.6) / 3.4)
    if body:
        ax.text(
            0.08,
            0.63,
            body,
            fontsize=14.5,
            color=THEME.ink_secondary,
            family=MONO,
            va="top",
            transform=ax.transAxes,
            linespacing=1.5,
        )

    badges = [
        ("GridMap", THEME.agent_color(0)),
        ("MAPFProblem", THEME.agent_color(2)),
        ("Solution", THEME.agent_color(3)),
        ("SearchTrace", THEME.agent_color(6)),
        ("solver registry", THEME.agent_color(1)),
    ]
    for index, (label, color) in enumerate(badges):
        alpha = fade(ax, t, 3.9 + 0.16 * index)
        if alpha <= 0.02:
            continue
        x = 0.08 + 0.172 * index
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.10),
                0.155,
                0.062,
                boxstyle="round,pad=0.004,rounding_size=0.014",
                facecolor=THEME.surface,
                edgecolor=color,
                linewidth=1.4,
                alpha=alpha,
                transform=ax.transAxes,
            )
        )
        text(ax, x + 0.0775, 0.131, label, size=13.5, color=color, ha="center", alpha=alpha)


def scene_search(ax, t, content):
    """The centrepiece: CBS finding conflicts and re-planning around them."""
    scenario = content.search_scenario
    grid = scenario.grid
    cell = 0.047
    origin = (0.075, 0.055)

    text(ax, 0.07, 0.90, "Watch the search", size=34, weight="bold", alpha=fade(ax, t, 0.0))
    text(ax, 0.07, 0.835, "Every solver streams its own events. Nothing here is a mock-up.",
         size=16, color=THEME.ink_secondary, alpha=fade(ax, t, 0.25))

    draw_map(ax, grid, origin, cell, alpha=0.9)

    # Step through the recorded trace in real time.
    events = content.search_events
    # Replay fast enough that the last event lands ~1.6 s before the cut.
    duration = DURATIONS.get("scene_search", 9.5)
    rate = max(1.0, len(events) / max(1.0, duration - 2.4))
    index = min(len(events) - 1, int((t - 0.8) * rate))
    if index < 0:
        index = 0
    paths = None
    conflicts = []
    solved = False
    for event in events[: index + 1]:
        if event.kind == "expand" and event.get("paths"):
            paths = event["paths"]
        elif event.kind == "conflict":
            conflicts.append(event["cell"])
        elif event.kind == "solved":
            paths = event.get("paths", paths)
            solved = True
    conflicts = conflicts[-3:]

    if paths:
        for order, (name, path) in enumerate(paths.items()):
            color = THEME.agent_color(order)
            draw_path(ax, grid, origin, cell, path, color, alpha=0.95, width=2.4)
            draw_agent(ax, grid, origin, cell, path[0], color, radius=0.3, alpha=0.9, label=name)

    if not solved:
        for cell_position in conflicts:
            x, y = cell_xy(grid, origin, cell, cell_position)
            for dx, dy in ((-1, -1), (-1, 1)):
                ax.plot(
                    [x + dx * cell * 0.24, x - dx * cell * 0.24],
                    [y + dy * cell * 0.24, y - dy * cell * 0.24],
                    color=theme_module.STATUS["critical"],
                    linewidth=3.0,
                    solid_capstyle="round",
                    transform=ax.transAxes,
                    zorder=7,
                )

    # Live readout on the right.
    expansions = sum(1 for event in events[: index + 1] if event.kind == "expand")
    found = sum(1 for event in events[: index + 1] if event.kind == "conflict")
    panel_x = 0.60
    text(ax, panel_x, 0.68, "high-level nodes", size=14, color=THEME.muted)
    text(ax, panel_x, 0.61, str(expansions), size=44, weight="bold", family=MONO)
    text(ax, panel_x, 0.50, "conflicts found", size=14, color=THEME.muted)
    text(ax, panel_x, 0.43, str(found), size=44, weight="bold", family=MONO,
         color=theme_module.STATUS["critical"] if not solved else THEME.ink)

    if solved:
        solution = content.search_solution
        text(ax, panel_x, 0.31, "conflict-free · optimal", size=19,
             color=theme_module.STATUS["good"], weight="bold", alpha=fade(ax, t, 0.0, 0.3))
        text(ax, panel_x, 0.25,
             "sum of costs %d   ·   makespan %d" % (solution.sum_of_costs, solution.makespan),
             size=15, color=THEME.ink_secondary)


def scene_execute(ax, t, content):
    """The plan runs: agents crossing a warehouse without ever touching."""
    scenario = content.hero
    solution = content.hero_solution
    grid = scenario.grid
    cell = 0.032
    origin = (0.08, 0.07)

    text(ax, 0.09, 0.90, "Then watch the plan run", size=34, weight="bold", alpha=fade(ax, t, 0.0))
    text(ax, 0.09, 0.835,
         "%d agents · %d timesteps · zero collisions"
         % (len(solution.paths), solution.makespan),
         size=16, color=THEME.ink_secondary, alpha=fade(ax, t, 0.2))

    draw_map(ax, grid, origin, cell)

    duration = DURATIONS.get("scene_execute", 8.5)
    rate = solution.makespan / max(1.0, duration - 2.0)
    clock = max(0.0, (t - 0.5)) * rate
    for index, (name, path) in enumerate(solution.paths.items()):
        color = THEME.agent_color(index)
        goal = path[-1]
        gx, gy = cell_xy(grid, origin, cell, goal)
        ax.add_patch(
            disc(ax, gx, gy, cell * 0.36, facecolor="none", edgecolor=color,
                 linewidth=1.6, alpha=0.5)
        )
        head = int(min(clock, len(path) - 1))
        tail = max(0, head - 8)
        draw_path(ax, grid, origin, cell, path[tail : head + 1], color, alpha=0.55, width=2.6)
        draw_agent(ax, grid, origin, cell, interpolate(path, min(clock, len(path) - 1)),
                   color, radius=0.36, label=name)

    text(ax, 0.78, 0.48, "t = %d / %d" % (int(min(clock, solution.makespan)), solution.makespan),
         size=22, family=MONO, color=THEME.ink_secondary)
    text(ax, 0.78, 0.40, "cost %d" % solution.sum_of_costs, size=22, family=MONO,
         color=THEME.ink_secondary)


def _swarm_frame(result, t, duration, tail=26):
    """Positions and a short velocity tail at time ``t`` through a swarm run."""
    history = result.history
    index = min(len(history) - 1, int((t / max(duration, 1e-6)) * (len(history) - 1)))
    window = [history[max(0, index - k)].positions for k in range(tail, -1, -1)]
    return history[index].positions, window


def _fit_view(points, margin=1.15):
    """A square window around ``points``, in world units."""
    import numpy as np

    centre = points.mean(axis=0)
    span = max(float(np.abs(points - centre).max()) * margin, 1e-6)
    return centre, span


def _projector(box, centre, span):
    """Map world coordinates into ``box`` without distorting the geometry.

    Axes coordinates are not isotropic on a 16:9 canvas -- one unit across is
    16/9 times longer than one unit up -- so mapping x and y through the same
    fraction turns a V formation into a zigzag and a circle into an ellipse.
    The box is therefore shrunk to whichever axis binds, and the same
    world-units-per-inch is used for both.
    """
    x0, y0, width, height = box
    if width * WIDTH > height * HEIGHT:      # too wide: height binds
        used_w, used_h = height * HEIGHT / WIDTH, height
    else:                                    # too tall: width binds
        used_w, used_h = width, width * WIDTH / HEIGHT
    cx, cy = x0 + width / 2, y0 + height / 2

    def place(point):
        return (
            cx + used_w * (point[0] - centre[0]) / (2 * span),
            cy + used_h * (point[1] - centre[1]) / (2 * span),
        )

    return place


def scene_swarm(ax, t, content):
    """The decentralized half: a flock that needs almost nothing to sense."""
    import numpy as np

    text(ax, 0.07, 0.90, "The other half: no planner at all.", size=34, weight="bold",
         alpha=fade(ax, t, 0.0))
    text(ax, 0.07, 0.835,
         "Ten flocking laws, one interface, identical initial conditions.",
         size=16, color=THEME.ink_secondary, alpha=fade(ax, t, 0.2))

    duration = DURATIONS.get("scene_swarm", 8.0)
    positions, window = _swarm_frame(content.flock, t, duration)
    centre, span = _fit_view(np.vstack(window))

    # Draw into the right two-thirds of the frame.
    place = _projector((0.36, 0.14, 0.58, 0.58), centre, span)

    appear = ease_out(t / 0.8)
    for agent in range(positions.shape[0]):
        colour = THEME.agent_color(agent)
        trail = [place(frame[agent]) for frame in window]
        ax.plot(*zip(*trail), color=colour, linewidth=1.6, alpha=0.42 * appear,
                solid_capstyle="round", transform=ax.transAxes, zorder=3)
        x, y = trail[-1]
        ax.add_patch(disc(ax, x, y, 0.0062, facecolor=colour, edgecolor=THEME.plane,
                          linewidth=0.9, alpha=appear, zorder=5))

    lines = [
        ("no GPS", 0.9),
        ("no compass", 1.25),
        ("no radio between agents", 1.6),
        ("no velocity sensing", 1.95),
    ]
    for row, (label, start) in enumerate(lines):
        text(ax, 0.07, 0.66 - row * 0.075, label, size=21, family=MONO,
             color=THEME.agent_color(1), alpha=fade(ax, t, start))
    text(ax, 0.07, 0.30,
         "Only relative range and bearing \u2014\nand the flock still agrees on a heading.",
         size=17, color=THEME.ink_secondary, alpha=fade(ax, t, 2.5))
    if t > 3.1:
        text(ax, 0.07, 0.17, "polarisation  %.2f" % content.flock_order, size=25,
             weight="bold", family=MONO, color=THEME.agent_color(2),
             alpha=fade(ax, t, 3.1))
    text(ax, 0.07, 0.10, "Amorim, Nascimento, Chaudhary, Ferrante, Saska (2024)",
         size=12.5, color=THEME.muted, alpha=fade(ax, t, 3.4))


def scene_formation(ax, t, content):
    """Formation control: a shape, and what it costs to be able to hold it."""
    import numpy as np

    text(ax, 0.07, 0.90, "Formation control: hold a shape.", size=34, weight="bold",
         alpha=fade(ax, t, 0.0))
    text(ax, 0.07, 0.835,
         "What each agent can measure decides which symmetry it can fix.",
         size=16, color=THEME.ink_secondary, alpha=fade(ax, t, 0.2))

    duration = DURATIONS.get("scene_formation", 8.5)
    # Hold on the finished formation for the last third of the scene.
    progress = min(1.0, (t / max(duration, 1e-6)) * 1.5)
    history = content.formation.history
    index = min(len(history) - 1, int(progress * (len(history) - 1)))
    positions = history[index].positions
    target = content.formation_shape.centred(positions.shape[0], 2)
    centre, span = _fit_view(
        np.vstack([positions, target + positions.mean(axis=0)]), margin=1.25
    )

    place = _projector((0.05, 0.14, 0.46, 0.58), centre, span)

    # Ghost the target slots so the convergence is legible. They are offsets
    # from the formation centre, so they have to be moved to where the swarm
    # actually is before they mean anything.
    swarm_centre = positions.mean(axis=0)
    for slot in target:
        x, y = place(slot + swarm_centre)
        ax.add_patch(disc(ax, x, y, 0.0085, facecolor="none", edgecolor=THEME.grid,
                          linewidth=1.1, alpha=0.55 * fade(ax, t, 0.3), zorder=2))
    # Struts between the closest pairs *in the target shape*. Nine loose dots
    # do not read as a chevron; the same nine with their arms drawn do -- and
    # the struts are the formation's own structure, not decoration.
    order = np.argsort(
        np.linalg.norm(target[:, None, :] - target[None, :, :], axis=2)
        + np.eye(len(target)) * 1e9,
        axis=1,
    )
    holder = np.empty(len(target), dtype=int)      # slot -> agent
    holder[content.formation_slots] = np.arange(len(target))
    drawn = set()
    for i in range(len(target)):
        for j in order[i, :2]:
            pair = (min(i, int(j)), max(i, int(j)))
            if pair in drawn:
                continue
            drawn.add(pair)
            ends = [place(positions[holder[pair[0]]]),
                    place(positions[holder[pair[1]]])]
            ax.plot(*zip(*ends), color=THEME.ink_secondary, linewidth=1.3,
                    alpha=0.30 * ease_out(t / 0.9), transform=ax.transAxes, zorder=3)

    for agent in range(positions.shape[0]):
        x, y = place(positions[agent])
        ax.add_patch(disc(ax, x, y, 0.0082, facecolor=THEME.agent_color(agent),
                          edgecolor=THEME.plane, linewidth=1.0,
                          alpha=ease_out(t / 0.7), zorder=5))

    errors = content.formation_error
    if errors:
        current = errors[min(index, len(errors) - 1)]
        text(ax, 0.07, 0.09, "formation error  %.3f m" % current, size=19,
             family=MONO, weight="bold",
             color=THEME.agent_color(2) if current < 0.05 else THEME.ink_secondary,
             alpha=fade(ax, t, 0.6))

    rows = [
        ("displacement", "relative position", "translation", 1.1),
        ("distance", "range only", "+ rotation, reflection", 1.5),
        ("bearing", "direction only", "+ scale", 1.9),
    ]
    text(ax, 0.56, 0.70, "measures", size=13, color=THEME.muted, alpha=fade(ax, t, 1.0))
    text(ax, 0.79, 0.70, "free up to", size=13, color=THEME.muted, alpha=fade(ax, t, 1.0))
    for row, (law, measures, free, start) in enumerate(rows):
        y = 0.62 - row * 0.085
        alpha = fade(ax, t, start)
        text(ax, 0.56, y, law, size=17, weight="bold", family=MONO,
             color=THEME.agent_color(row), alpha=alpha)
        text(ax, 0.56, y - 0.038, measures, size=13.5, color=THEME.ink_secondary, alpha=alpha)
        text(ax, 0.79, y, free, size=14, color=THEME.ink_secondary, alpha=alpha)

    text(ax, 0.56, 0.28,
         "The less an agent senses, the longer it takes:\n3.6 s, 15.6 s, 41.1 s to converge.",
         size=15, color=THEME.ink_secondary, alpha=fade(ax, t, 2.4))
    text(ax, 0.56, 0.13,
         "A collinear target is not rigid \u2014\nthe library says so before you fly it.",
         size=14, color=THEME.muted, alpha=fade(ax, t, 3.0))


def scene_benchmark(ax, t, content):
    """Three solvers, measured -- the curves draw themselves in."""
    import math

    text(ax, 0.07, 0.90, "Three solvers. Measured, not asserted.", size=34, weight="bold",
         alpha=fade(ax, t, 0.0))
    text(ax, 0.07, 0.835,
         "pymapf.benchmark runs the sweep; pymapf.viz draws it.",
         size=16, color=THEME.ink_secondary, alpha=fade(ax, t, 0.2))

    # ---- runtime panel (log scale, drawn by hand so it can animate) --------
    px, py, pw, ph = 0.10, 0.24, 0.38, 0.44
    runtime = content.runtime
    xs = sorted({x for series in runtime.values() for x in series[0]})
    values = [v for series in runtime.values() for v in series[1] if v > 0]
    lo, hi = min(values), max(values)
    log_lo, log_hi = math.floor(math.log10(lo)), math.ceil(math.log10(hi))

    def to_x(v):
        return px + pw * (v - min(xs)) / max(1e-9, max(xs) - min(xs))

    def to_y(v):
        span = max(1e-9, log_hi - log_lo)
        return py + ph * (math.log10(max(v, 10 ** log_lo)) - log_lo) / span

    def fmt(seconds):
        if seconds >= 1:
            return "%gs" % seconds
        return "%gms" % round(seconds * 1000, 3)

    for decade in range(int(log_lo), int(log_hi) + 1):
        y = to_y(10.0 ** decade)
        ax.plot([px, px + pw], [y, y], color=THEME.grid, linewidth=0.7,
                transform=ax.transAxes)
        text(ax, px - 0.012, y, fmt(10.0 ** decade), size=11, color=THEME.muted, ha="right")
    for value in xs:
        text(ax, to_x(value), py - 0.045, str(value), size=11, color=THEME.muted, ha="center")
    text(ax, px + pw / 2, py - 0.10, "agents", size=13, color=THEME.ink_secondary, ha="center")
    text(ax, px - 0.012, py + ph + 0.06, "runtime  (log scale)", size=15, color=THEME.muted)

    reveal = ease_out((t - 0.5) / 2.0)
    # Series labels are nudged apart when two curves end at the same height.
    ends = []
    for order, name in enumerate(sorted(runtime)):
        color = THEME.agent_color(order)
        series_x, series_y = runtime[name]
        count = max(2, int(len(series_x) * reveal))
        points = [(to_x(x), to_y(y)) for x, y in zip(series_x[:count], series_y[:count])]
        if len(points) < 2:
            continue
        ax.plot(*zip(*points), color=color, linewidth=2.6, transform=ax.transAxes,
                solid_capstyle="round", zorder=4)
        ax.scatter(*zip(*points), s=26, color=color, edgecolor=THEME.surface,
                   linewidth=1.2, transform=ax.transAxes, zorder=5)
        label_y = points[-1][1]
        while any(abs(label_y - taken) < 0.035 for taken in ends):
            label_y += 0.035
        ends.append(label_y)
        text(ax, points[-1][0] + 0.014, label_y, name, size=14, color=THEME.ink_secondary)

    # ---- success-rate bars ------------------------------------------------
    bar_x = 0.60
    text(ax, bar_x, py + ph + 0.06, "instances solved", size=15, color=THEME.muted)
    for order, name in enumerate(sorted(content.success)):
        rate = content.success[name]
        color = THEME.agent_color(order)
        grown = ease_out((t - 1.2 - 0.15 * order) / 0.6)
        width = 0.19 * (rate / 100) * grown
        y = py + ph - 0.10 - 0.13 * order
        ax.add_patch(
            Rectangle((bar_x + 0.11, y), max(0.001, width), 0.055,
                      facecolor=color, edgecolor="none", transform=ax.transAxes)
        )
        text(ax, bar_x + 0.10, y + 0.028, name, size=14, color=THEME.ink_secondary, ha="right")
        if grown > 0.15:
            text(ax, bar_x + 0.12 + width, y + 0.028, "%.0f%%" % rate, size=13,
                 color=THEME.ink_secondary)

    text(ax, bar_x + 0.11, py + 0.02,
         "CBS times out where the others do not:\nthat is the price of optimality.",
         size=13.5, color=THEME.muted, alpha=fade(ax, t, 2.4))

    text(ax, 0.07, 0.10,
         "CBS is optimal  ·  Weighted CBS trades a bounded slice of cost for speed  ·  "
         "Prioritized planning is fastest and incomplete",
         size=14, color=THEME.muted, alpha=fade(ax, t, 3.0))


def scene_outro(ax, t, content):
    grid = content.hero.grid
    cell = 0.030
    origin = (0.5 - grid.width * cell / 2, 0.5 - grid.height * cell * ASPECT / 2)
    draw_map(ax, grid, origin, cell, alpha=0.12)
    for index, (name, path) in enumerate(content.hero_solution.paths.items()):
        draw_path(ax, grid, origin, cell, path, THEME.agent_color(index), alpha=0.18, width=2.0)

    text(ax, 0.5, 0.66, "pip install pymapf", size=46, weight="bold", ha="center",
         family=MONO, alpha=fade(ax, t, 0.1))
    text(ax, 0.5, 0.55, "solvers · scenarios · benchmarks · live views",
         size=21, color=THEME.ink_secondary, ha="center", alpha=fade(ax, t, 0.5))
    text(ax, 0.5, 0.44, "github.com/apla-toolbox/pymapf", size=17, color=THEME.agent_color(0),
         ha="center", family=MONO, alpha=fade(ax, t, 0.9))
    text(ax, 0.5, 0.30, "Apache 2.0 · a university project, rebuilt", size=14,
         color=THEME.muted, ha="center", alpha=fade(ax, t, 1.3))


# Scene, duration in seconds. The search and execute scenes derive their replay
# speed from these, so a plan always finishes on screen before the cut.
SCENES = [
    (scene_title, 4.5),
    (scene_origin, 6.0),
    (scene_rebuild, 8.0),
    (scene_search, 9.5),
    (scene_execute, 8.5),
    (scene_swarm, 9.0),
    (scene_formation, 9.5),
    (scene_benchmark, 7.5),
    (scene_outro, 5.0),
]
DURATIONS = {}


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def build_animation(content, seconds=None):
    figure = plt.figure(figsize=(WIDTH, HEIGHT), facecolor=THEME.plane)
    ax = figure.add_axes([0, 0, 1, 1])
    ax.set_facecolor(THEME.plane)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    DURATIONS.update({scene.__name__: duration for scene, duration in SCENES})
    total = sum(duration for _, duration in SCENES)
    if seconds:
        total = min(total, seconds)
    frames = int(total * FPS)

    def render(frame_index):
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor(THEME.plane)

        now = frame_index / FPS
        elapsed = 0.0
        for scene, duration in SCENES:
            if now < elapsed + duration:
                local = now - elapsed
                scene(ax, local, content)
                # Cross-fade the last 0.35s of every scene to black.
                remaining = duration - local
                if remaining < 0.35:
                    ax.add_patch(
                        Rectangle((0, 0), 1, 1, facecolor=THEME.plane,
                                  alpha=1 - remaining / 0.35, transform=ax.transAxes, zorder=50)
                    )
                if local < 0.3:
                    ax.add_patch(
                        Rectangle((0, 0), 1, 1, facecolor=THEME.plane,
                                  alpha=1 - local / 0.3, transform=ax.transAxes, zorder=50)
                    )
                break
            elapsed += duration
        else:
            SCENES[-1][0](ax, SCENES[-1][1], content)

        # Persistent corner mark.
        text(ax, 0.955, 0.045, "PyMAPF", size=12, color=THEME.axis, ha="right")
        return []

    animation = FuncAnimation(figure, render, frames=frames, interval=1000 // FPS, blit=False)
    return figure, animation, frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=os.path.join(ROOT, "docs", "assets", "pymapf-promo.mp4")
    )
    parser.add_argument("--gif", action="store_true", help="also write a looping GIF")
    parser.add_argument("--preview", type=float, default=None, help="render only N seconds")
    parser.add_argument("--dpi", type=int, default=80, help="80 -> 1280x720, 120 -> 1920x1080")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print("Rendering promo film")
    content = Content()
    figure, animation, frames = build_animation(content, args.preview)
    print("  %d frames at %d fps (%.1fs)" % (frames, FPS, frames / FPS))

    try:
        import imageio_ffmpeg

        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    from matplotlib.animation import FFMpegWriter

    started = time.perf_counter()
    writer = FFMpegWriter(
        fps=FPS,
        bitrate=6000,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-preset", "slow", "-movflags", "+faststart"],
    )
    animation.save(args.output, writer=writer, dpi=args.dpi)
    print("  wrote %s in %.0fs" % (args.output, time.perf_counter() - started))

    # A poster frame for the <video> element.
    poster = os.path.join(os.path.dirname(args.output), "promo-poster.png")
    render = animation._func
    render(int(1.9 * FPS))
    figure.savefig(poster, dpi=args.dpi, facecolor=THEME.plane)
    print("  wrote %s" % poster)

    if args.gif:
        gif = args.output.rsplit(".", 1)[0] + ".gif"
        animation.save(gif, writer="pillow", fps=12, dpi=48)
        print("  wrote %s" % gif)

    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
