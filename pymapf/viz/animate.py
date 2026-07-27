"""Animations: agents executing a plan, and the search that produced it.

Two views, both built on :class:`matplotlib.animation.FuncAnimation`:

* :func:`animate_solution` plays the plan -- agents glide between cells with
  fading trails and a timestep readout;
* :func:`animate_search` replays a :class:`~pymapf.core.trace.SearchTrace`, so
  you watch CBS discover conflicts and re-plan agents node by node.

:func:`save` writes either one to ``.mp4`` (ffmpeg) or ``.gif`` (pillow), and
:func:`to_jshtml` embeds one in a notebook.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ..core.grid import GridMap
from ..core.solver import Solution
from . import theme as theme_module
from .plots import _as_grid, _draw_map, _new_axes

__all__ = ["animate_solution", "animate_search", "save", "to_jshtml"]

# Sub-steps between two timesteps. Six is enough to read as motion at 12-20 fps
# without inflating frame counts on long plans.
SUBSTEPS = 6


def _ease(fraction: float) -> float:
    """Smoothstep: agents accelerate out of a cell and settle into the next."""
    return fraction * fraction * (3 - 2 * fraction)


def _interpolated(path: Sequence, frame: int, substeps: int):
    """Position along ``path`` at a sub-timestep frame index."""
    t, phase = divmod(frame, substeps)
    current = path[min(t, len(path) - 1)]
    nxt = path[min(t + 1, len(path) - 1)]
    alpha = _ease(phase / substeps)
    return (
        current[0] + (nxt[0] - current[0]) * alpha,
        current[1] + (nxt[1] - current[1]) * alpha,
    )


def animate_solution(
    solution: Solution,
    source,
    theme="dark",
    substeps: int = SUBSTEPS,
    trail: int = 8,
    title: Optional[str] = None,
    hold_frames: int = 12,
    figsize=None,
):
    """Animate agents executing ``solution``.

    Args:
        substeps: interpolation frames per timestep (higher = smoother).
        trail: how many timesteps of history stay visible behind each agent.
        hold_frames: extra frames at the end so the final state is readable
            before a looping GIF restarts.

    Returns the :class:`~matplotlib.animation.FuncAnimation`; keep a reference
    to it or it will be garbage collected before it renders.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.lines import Line2D

    resolved = theme_module.apply(theme)
    grid = _as_grid(source)
    ax = _new_axes(grid, None, resolved, figsize)
    _draw_map(ax, grid, resolved)

    names = list(solution.paths)
    colors = resolved.color_map(names)
    markers = resolved.marker_map(names)
    horizon = solution.makespan

    # Goal rings are static; draw once.
    for name in names:
        goal = solution.paths[name][-1]
        ax.plot(
            goal[1],
            goal[0],
            marker=markers[name],
            markersize=13,
            markerfacecolor="none",
            markeredgecolor=colors[name],
            markeredgewidth=1.8,
            alpha=0.55,
            linestyle="none",
            zorder=2,
        )

    trails: Dict[str, Line2D] = {}
    bodies: Dict[str, Line2D] = {}
    labels = {}
    for name in names:
        (trails[name],) = ax.plot(
            [], [], color=colors[name], linewidth=2.4, alpha=0.5, zorder=3
        )
        (bodies[name],) = ax.plot(
            [],
            [],
            marker=markers[name],
            markersize=11,
            color=colors[name],
            markeredgecolor=resolved.surface,
            markeredgewidth=1.6,
            linestyle="none",
            zorder=5,
        )
        labels[name] = ax.annotate(
            name,
            (0, 0),
            textcoords="offset points",
            xytext=(0, 11),
            ha="center",
            fontsize=8,
            color=resolved.ink_secondary,
            zorder=6,
        )

    heading = title if title is not None else (solution.algorithm or "plan")
    ax.set_title(heading, color=resolved.ink, pad=12)
    clock = ax.annotate(
        "",
        xy=(0, 0),
        xycoords="axes fraction",
        xytext=(0, -14),
        textcoords="offset points",
        fontsize=9,
        color=resolved.muted,
    )

    total = horizon * substeps + 1 + hold_frames

    def update(frame):
        frame = min(frame, horizon * substeps)
        t = frame // substeps
        for name in names:
            path = solution.paths[name]
            row, col = _interpolated(path, frame, substeps)
            bodies[name].set_data([col], [row])
            labels[name].xy = (col, row)
            start = max(0, t - trail)
            history = list(path[start : t + 1]) + [(row, col)]
            trails[name].set_data(
                [cell[1] for cell in history], [cell[0] for cell in history]
            )
        clock.set_text(
            "t = %d / %d    ·    cost %d    ·    %s"
            % (t, horizon, solution.sum_of_costs, solution.algorithm or "plan")
        )
        return list(bodies.values()) + list(trails.values())

    animation = FuncAnimation(
        ax.figure, update, frames=total, interval=1000 // 20, blit=False
    )
    plt.close(ax.figure)  # keep notebooks from double-rendering the still frame
    return animation


def animate_search(
    trace,
    source,
    theme="dark",
    max_nodes: Optional[int] = None,
    title: str = "Conflict-Based Search, live",
):
    """Replay a recorded search: the plan under consideration and its conflicts.

    Left: the current node's paths, with the conflict CBS just found marked.
    Right: the cost of each expanded node and a running conflict count -- the
    picture of how much work the instance actually required.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    resolved = theme_module.apply(theme)
    grid: GridMap = _as_grid(source)

    events = [e for e in trace if e.kind in ("expand", "conflict", "solved", "failed")]
    if max_nodes is not None:
        kept, expansions = [], 0
        for event in events:
            if event.kind == "expand":
                expansions += 1
                if expansions > max_nodes:
                    break
            kept.append(event)
        events = kept
    if not events:
        raise ValueError("trace contains no search events to animate")

    figure = plt.figure(figsize=(11, 5.2))
    grid_spec = figure.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.22)
    map_ax = figure.add_subplot(grid_spec[0, 0])
    chart_ax = figure.add_subplot(grid_spec[0, 1])

    map_ax.set_facecolor(resolved.surface)
    _draw_map(map_ax, grid, resolved)
    map_ax.set_xlim(-0.5, grid.width - 0.5)
    map_ax.set_ylim(grid.height - 0.5, -0.5)
    map_ax.set_aspect("equal")
    map_ax.set_xticks([])
    map_ax.set_yticks([])
    map_ax.set_title(title, color=resolved.ink, pad=10)

    chart_ax.set_facecolor(resolved.surface)
    chart_ax.set_title("Cost of expanded nodes", color=resolved.ink, pad=10)
    chart_ax.set_xlabel("high-level node", color=resolved.ink_secondary, fontsize=9)
    chart_ax.set_ylabel("sum of costs", color=resolved.ink_secondary, fontsize=9)
    chart_ax.grid(color=resolved.grid, linewidth=0.6)
    chart_ax.set_axisbelow(True)
    for spine in chart_ax.spines.values():
        spine.set_visible(False)

    (cost_line,) = chart_ax.plot([], [], color=resolved.categorical[0], linewidth=2.0)
    (cost_dot,) = chart_ax.plot(
        [],
        [],
        marker="o",
        markersize=8,
        color=resolved.categorical[0],
        markeredgecolor=resolved.surface,
        markeredgewidth=1.5,
        linestyle="none",
    )
    status = chart_ax.annotate(
        "",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 10),
        textcoords="offset points",
        fontsize=9,
        color=resolved.muted,
    )

    path_artists: List = []
    conflict_artists: List = []
    xs: List[int] = []
    ys: List[int] = []
    state = {"conflicts": 0, "solved": False}

    def draw_paths(paths, highlight=None):
        while path_artists:
            path_artists.pop().remove()
        names = list(paths)
        colors = resolved.color_map(names)
        for index, name in enumerate(names):
            path = paths[name]
            offset = 0.12 * (index - (len(names) - 1) / 2) / max(1, len(names) / 2)
            (line,) = map_ax.plot(
                [cell[1] + offset for cell in path],
                [cell[0] + offset for cell in path],
                color=colors[name],
                linewidth=2.0,
                alpha=0.95 if highlight in (None, name) else 0.45,
                zorder=3,
            )
            path_artists.append(line)
            (dot,) = map_ax.plot(
                path[0][1] + offset,
                path[0][0] + offset,
                marker="o",
                markersize=6,
                color=colors[name],
                markeredgecolor=resolved.surface,
                markeredgewidth=1.2,
                linestyle="none",
                zorder=4,
            )
            path_artists.append(dot)

    def update(index):
        event = events[index]
        if event.kind == "expand":
            paths = event.get("paths")
            if paths:
                draw_paths(paths)
            xs.append(event.get("node", len(xs) + 1))
            ys.append(event.get("cost", 0))
            cost_line.set_data(xs, ys)
            cost_dot.set_data(xs[-1:], ys[-1:])
            chart_ax.set_xlim(0, max(5, xs[-1] + 1))
            low, high = min(ys), max(ys)
            pad = max(1, (high - low) * 0.25)
            chart_ax.set_ylim(low - pad, high + pad)
        elif event.kind == "conflict":
            state["conflicts"] += 1
            cell = event.get("cell")
            if cell is not None:
                marker = map_ax.plot(
                    cell[1],
                    cell[0],
                    marker="X",
                    markersize=13,
                    color=theme_module.STATUS["critical"],
                    markeredgecolor=resolved.surface,
                    markeredgewidth=1.4,
                    linestyle="none",
                    zorder=6,
                )[0]
                conflict_artists.append(marker)
                # Keep only the most recent few so the map does not silt up.
                while len(conflict_artists) > 4:
                    conflict_artists.pop(0).remove()
        elif event.kind == "solved":
            state["solved"] = True
            paths = event.get("paths")
            if paths:
                draw_paths(paths)
            while conflict_artists:
                conflict_artists.pop().remove()

        if state["solved"]:
            status.set_text(
                "solved  ·  cost %d  ·  %d conflicts resolved"
                % (ys[-1] if ys else 0, state["conflicts"])
            )
            status.set_color(theme_module.STATUS["good"])
        else:
            status.set_text(
                "searching  ·  node %d  ·  %d conflicts found"
                % (len(xs), state["conflicts"])
            )
        return path_artists + [cost_line, cost_dot, status]

    animation = FuncAnimation(
        figure,
        update,
        frames=len(events) + 10,  # tail frames hold the solved state
        interval=220,
        blit=False,
        repeat_delay=1200,
    )
    # Frames beyond the event list should replay the last one, not crash.
    original = update

    def guarded(index):
        return original(min(index, len(events) - 1))

    animation._func = guarded
    plt.close(figure)
    return animation


def save(animation, path: str, fps: int = 20, dpi: int = 120, bitrate: int = 3200) -> str:
    """Write an animation to ``.mp4`` or ``.gif`` and return the path.

    MP4 needs ffmpeg on PATH (or ``imageio-ffmpeg`` installed, which we wire up
    automatically); GIF only needs pillow.
    """
    if path.endswith(".gif"):
        animation.save(path, writer="pillow", fps=fps, dpi=dpi)
        return path

    import matplotlib as mpl

    if not mpl.rcParams.get("animation.ffmpeg_path", "ffmpeg") or True:
        try:  # a bundled ffmpeg is the difference between "works" and "install ffmpeg"
            import imageio_ffmpeg

            mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # pragma: no cover - system ffmpeg is fine too
            pass

    from matplotlib.animation import FFMpegWriter

    writer = FFMpegWriter(
        fps=fps,
        bitrate=bitrate,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-preset", "slow"],
    )
    animation.save(path, writer=writer, dpi=dpi)
    return path


def to_jshtml(animation) -> str:
    """HTML/JS player for the animation -- what notebooks should display."""
    return animation.to_jshtml()
