"""Drawing continuous trajectories, in the plane or in a volume.

The other plotters in this package draw *plans*: cells, timesteps, a grid
underneath. A :class:`~pymapf.trajectory.JointPlan` has none of those -- it
is a set of polynomials over real space -- so it gets its own two functions:
one that draws the paths with the fleet's bodies at a chosen instant, and
one that shows what a plan is actually judged on, the pairwise separation
against the requirement over the whole flight.
"""

from __future__ import annotations

import math
from typing import Optional

from . import theme as theme_module

__all__ = ["plot_trajectories", "plot_separation", "animate_trajectories"]


def _bodies(plan):
    return {vehicle.name: vehicle.radius for vehicle in plan.vehicles}


def _limits(points, pad):
    lower = points.min(axis=0) - pad
    upper = points.max(axis=0) + pad
    return lower, upper


def plot_trajectories(
    plan,
    ax=None,
    theme: str = "dark",
    at: Optional[float] = None,
    title: Optional[str] = None,
    show_bodies: bool = True,
    show_endpoints: bool = True,
    dt: float = 0.02,
    elev: float = 26.0,
    azim: float = -58.0,
):
    """Every vehicle's flown path, in 2D or 3D.

    Args:
        plan: a :class:`~pymapf.trajectory.JointPlan`.
        at: draw the fleet's bodies at this time in seconds; ``None`` puts
            them at the closest approach, which is the instant worth looking
            at -- if the plan is safe anywhere, that is where you can see it.
        show_bodies: draw each vehicle at its true radius. A trajectory plot
            without the bodies cannot show whether a plan is safe; two paths
            that cross on the page may be seconds apart in time, and two that
            never touch may still be a metre too close.

    Returns the axes.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    resolved = theme_module.apply(theme)
    names = list(plan.trajectories)
    dimension = plan.trajectories[names[0]].dimension
    if dimension not in (2, 3):
        raise ValueError(
            "plot_trajectories draws 2D and 3D fleets, not %dD" % dimension
        )
    colors = resolved.color_map(names)
    times = plan.times(dt)
    tracks = {name: plan.trajectories[name].sample(times) for name in names}
    if at is None:
        at = plan.min_separation(dt)[1]

    if ax is None:
        # An even pixel count at the usual DPIs: an odd-width canvas makes
        # some GIF writers shear every frame, and the failure looks like a
        # bug in the trajectories rather than in the encoder.
        figure = plt.figure(figsize=(6.0, 6.0) if dimension == 2 else (6.4, 6.0))
        ax = figure.add_subplot(111, projection="3d" if dimension == 3 else None)
    ax.set_facecolor(resolved.surface)

    for name in names:
        track = tracks[name]
        coordinates = [track[:, k] for k in range(dimension)]
        ax.plot(*coordinates, color=colors[name], linewidth=1.6, alpha=0.9, zorder=3)
        if show_endpoints:
            start = [track[0, k] for k in range(dimension)]
            goal = [track[-1, k] for k in range(dimension)]
            ax.plot(
                *[[c] for c in start],
                marker="o",
                markersize=6,
                color=colors[name],
                zorder=4,
            )
            ax.plot(
                *[[c] for c in goal],
                marker="s",
                markersize=7,
                markerfacecolor="none",
                markeredgecolor=colors[name],
                markeredgewidth=1.4,
                zorder=4,
            )

    if show_bodies:
        radii = _bodies(plan)
        for name in names:
            centre = plan.trajectories[name](at)
            if dimension == 2:
                ax.add_patch(
                    plt.Circle(
                        tuple(centre),
                        radii[name],
                        facecolor=colors[name],
                        edgecolor="none",
                        alpha=0.85,
                        zorder=5,
                    )
                )
            else:
                ax.plot(
                    [centre[0]],
                    [centre[1]],
                    [centre[2]],
                    marker="o",
                    markersize=8,
                    color=colors[name],
                    zorder=5,
                )

    _frame(
        ax,
        np.concatenate([tracks[name] for name in names]),
        dimension,
        resolved,
        1.5 * max(_bodies(plan).values()) + 0.5,
        elev,
        azim,
    )
    if title:
        ax.set_title(title)
    return ax


def _frame(ax, points, dimension, resolved, pad, elev, azim) -> None:
    """Equal aspect, no ticks, and enough room for the bodies at the edges."""
    lower, upper = _limits(points, pad)
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_xticks([])
    ax.set_yticks([])
    if dimension == 3:
        ax.set_zlim(lower[2], upper[2])
        ax.set_zticks([])
        ax.view_init(elev=elev, azim=azim)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color((0, 0, 0, 0))
            pane.pane.set_edgecolor(resolved.grid)
    else:
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)


def plot_separation(
    plan, ax=None, theme: str = "dark", dt: float = 0.01, title: Optional[str] = None
):
    """Closest pair over time, against what the bodies require.

    This is the plot that says whether a plan is safe. The line is the
    minimum over every pair at each instant; the dashed rule is the largest
    requirement in the fleet. A line that touches the rule is a plan whose
    separation constraint is *active* -- the optimiser spent exactly as much
    detour as it had to, which is what optimal looks like here.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    resolved = theme_module.apply(theme)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 2.8))
    names = list(plan.trajectories)
    if len(names) < 2:
        raise ValueError(
            "plot_separation needs at least two vehicles; this plan has %d, "
            "so there is no pair to measure" % len(names)
        )
    times = plan.times(dt)
    tracks = {name: plan.trajectories[name].sample(times) for name in names}
    closest = np.full(len(times), np.inf)
    requirement = 0.0
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            distance = np.linalg.norm(tracks[first] - tracks[second], axis=1)
            closest = np.minimum(closest, distance)
            requirement = max(requirement, plan.required_separation(first, second))
    ax.set_facecolor(resolved.surface)
    ax.plot(times, closest, color=resolved.agent_color(0), linewidth=1.6)
    ax.axhline(requirement, color=resolved.muted, linewidth=0.9, linestyle="--")
    ax.text(
        times[0],
        requirement,
        " bodies touch",
        color=resolved.muted,
        fontsize=8,
        va="bottom",
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("closest pair (m)")
    ax.set_ylim(0.0, float(np.nanmax(closest[np.isfinite(closest)])) * 1.05)
    ax.set_title(
        title
        or "Closest pair: never below %.2f m, required %.2f m"
        % (float(closest.min()), requirement)
    )
    return ax


def animate_trajectories(
    plan,
    theme: str = "dark",
    fps: int = 20,
    trail: float = 2.5,
    title: Optional[str] = None,
    hold_frames: int = 16,
    elev: float = 26.0,
    azim: float = -58.0,
):
    """The fleet flying its trajectories, with fading trails.

    Args:
        trail: seconds of history drawn behind each vehicle.
        hold_frames: extra frames at the end so a looping GIF pauses on the
            final configuration.

    Returns the ``FuncAnimation``; keep a reference to it or it is collected
    before it renders. Save it with :func:`pymapf.viz.save_animation`.
    """
    import numpy as np
    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt

    resolved = theme_module.apply(theme)
    names = list(plan.trajectories)
    dimension = plan.trajectories[names[0]].dimension
    if dimension not in (2, 3):
        raise ValueError(
            "animate_trajectories draws 2D and 3D fleets, not %dD" % dimension
        )
    colors = resolved.color_map(names)
    radii = _bodies(plan)
    dt = 1.0 / fps
    frames = int(math.ceil(plan.duration / dt)) + hold_frames
    times = plan.times(dt)
    tracks = {name: plan.trajectories[name].sample(times) for name in names}

    ax = plot_trajectories(
        plan, theme=theme, show_bodies=False, title=title, elev=elev, azim=azim, dt=dt
    )
    figure = ax.figure
    for name in names:  # the paths become a faint backdrop
        for line in ax.lines:
            line.set_alpha(0.25)
    bodies, trails, readout = _artists(ax, names, colors, radii, dimension, resolved)
    closest = [math.inf]
    span = max(1, int(trail / dt))

    def update(frame):
        index = min(frame, len(times) - 1)
        for name in names:
            point = tracks[name][index]
            if dimension == 2:
                bodies[name].center = (point[0], point[1])
            else:
                bodies[name].set_data_3d([point[0]], [point[1]], [point[2]])
            history = tracks[name][max(0, index - span) : index + 1]
            if dimension == 2:
                trails[name].set_data(history[:, 0], history[:, 1])
            else:
                trails[name].set_data_3d(history[:, 0], history[:, 1], history[:, 2])
        points = np.array([tracks[name][index] for name in names])
        if len(names) > 1:
            difference = points[:, None, :] - points[None, :, :]
            distance = np.sqrt((difference**2).sum(axis=2)) + np.eye(len(names)) * 1e9
            closest[0] = min(closest[0], float(distance.min()))
        readout.set_text(
            "t = %5.2f s    closest so far %.2f m" % (times[index], closest[0])
        )
        return list(bodies.values()) + list(trails.values()) + [readout]

    animation = FuncAnimation(
        figure, update, frames=frames, interval=1000 * dt, blit=False
    )
    # As in `animate_solution`: the animation keeps a reference to the figure,
    # and closing it stops a notebook from also rendering the still frame and
    # stops a batch script from accumulating open figures.
    plt.close(figure)
    return animation


def _artists(ax, names, colors, radii, dimension, resolved):
    """One body, one trail and one readout, empty and ready to be moved."""
    import matplotlib.pyplot as plt

    bodies, trails = {}, {}
    for name in names:
        if dimension == 2:
            body = plt.Circle(
                (0, 0), radii[name], facecolor=colors[name], edgecolor="none", zorder=6
            )
            ax.add_patch(body)
        else:
            (body,) = ax.plot(
                [], [], [], marker="o", markersize=9, color=colors[name], zorder=6
            )
        bodies[name] = body
        trails[name] = ax.plot(
            *([[], [], []] if dimension == 3 else [[], []]),
            color=colors[name],
            linewidth=2.0,
            alpha=0.85,
            zorder=5,
        )[0]
    readout = (ax.text2D if dimension == 3 else ax.text)(
        0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9, color=resolved.ink
    )
    return bodies, trails, readout
