"""Live views: watch a solve as it happens, in a window or in the terminal.

Both classes here are plain observers -- pass one as the ``observer`` argument
of :func:`pymapf.solve` and it redraws while the search runs::

    scenario = pymapf.build_scenario("bottleneck")
    with LiveSolveView(scenario) as view:
        solution = pymapf.solve(scenario.to_problem(), "cbs", observer=view)

:class:`LiveConsoleView` needs no display at all, which makes it the one that
works over SSH and in CI logs.
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List, Optional

from ..core.grid import GridMap
from ..scenarios import Scenario
from . import theme as theme_module

__all__ = ["LiveSolveView", "LiveConsoleView"]


def _grid_of(source) -> GridMap:
    return source.grid if hasattr(source, "grid") else source


class LiveSolveView:
    """Interactive matplotlib view that redraws as the solver emits events.

    Args:
        source: the scenario/problem being solved (for the map).
        throttle: minimum seconds between redraws. Drawing is far slower than
            expanding a node, so an unthrottled view turns a 10 ms solve into a
            10 s one; the default keeps the window responsive and honest.
        theme: ``"dark"``, ``"light"`` or a :class:`~pymapf.viz.theme.Theme`.
    """

    def __init__(
        self, source, throttle: float = 0.05, theme="dark", title: str = "Live solve"
    ):
        import matplotlib.pyplot as plt

        from .plots import _draw_map, _new_axes

        self.resolved = theme_module.apply(theme)
        self.grid = _grid_of(source)
        self.throttle = throttle
        self._last_draw = 0.0
        self._artists: List = []
        self._conflict_artists: List = []
        self._plt = plt

        plt.ion()
        self.ax = _new_axes(self.grid, None, self.resolved)
        _draw_map(self.ax, self.grid, self.resolved)
        self.ax.set_title(title, color=self.resolved.ink, pad=12)
        self._status = self.ax.annotate(
            "waiting for the first node...",
            xy=(0, 0),
            xycoords="axes fraction",
            xytext=(0, -14),
            textcoords="offset points",
            fontsize=9,
            color=self.resolved.muted,
        )
        self.expansions = 0
        self.conflicts = 0
        self._show()

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "LiveSolveView":
        return self

    def __exit__(self, *exc_info) -> None:
        self.finish()

    # -- observer ----------------------------------------------------------
    def __call__(self, event) -> None:
        if event.kind == "expand":
            self.expansions += 1
            paths = event.get("paths")
            if paths:
                self._draw_paths(paths)
            self._status.set_text(
                "node %d  ·  cost %s  ·  %d conflicts"
                % (self.expansions, event.get("cost", "?"), self.conflicts)
            )
        elif event.kind == "agent_planned":
            self._draw_paths({event["agent"]: event["path"]}, replace=False)
        elif event.kind == "conflict":
            self.conflicts += 1
            self._mark_conflict(event.get("cell"))
        elif event.kind == "solved":
            paths = event.get("paths")
            if paths:
                self._draw_paths(paths)
            self._clear_conflicts()
            self._status.set_text(
                "solved  ·  cost %s  ·  makespan %s  ·  %d nodes"
                % (event.get("cost"), event.get("makespan"), self.expansions)
            )
            self._status.set_color(theme_module.STATUS["good"])
            self._show(force=True)
            return
        elif event.kind == "failed":
            self._status.set_text("failed: %s" % event.get("reason", ""))
            self._status.set_color(theme_module.STATUS["critical"])
            self._show(force=True)
            return
        self._show()

    # -- drawing -----------------------------------------------------------
    def _draw_paths(self, paths: Dict[str, List], replace: bool = True) -> None:
        if replace:
            while self._artists:
                self._artists.pop().remove()
        names = list(paths)
        colors = self.resolved.color_map(names)
        for index, name in enumerate(names):
            path = paths[name]
            offset = 0.12 * (index - (len(names) - 1) / 2) / max(1, len(names) / 2)
            (line,) = self.ax.plot(
                [cell[1] + offset for cell in path],
                [cell[0] + offset for cell in path],
                color=colors[name],
                linewidth=2.0,
                zorder=3,
            )
            self._artists.append(line)

    def _mark_conflict(self, cell) -> None:
        if cell is None:
            return
        (marker,) = self.ax.plot(
            cell[1],
            cell[0],
            marker="X",
            markersize=13,
            color=theme_module.STATUS["critical"],
            markeredgecolor=self.resolved.surface,
            markeredgewidth=1.4,
            linestyle="none",
            zorder=6,
        )
        self._conflict_artists.append(marker)
        while len(self._conflict_artists) > 4:
            self._conflict_artists.pop(0).remove()

    def _clear_conflicts(self) -> None:
        while self._conflict_artists:
            self._conflict_artists.pop().remove()

    def _show(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self._last_draw < self.throttle:
            return
        self._last_draw = now
        try:
            self.ax.figure.canvas.draw_idle()
            self.ax.figure.canvas.flush_events()
        except Exception:  # pragma: no cover - headless backends
            pass

    def finish(self, keep_open: bool = False) -> None:
        """Stop interactive mode; optionally block until the window is closed."""
        self._plt.ioff()
        if keep_open:  # pragma: no cover - interactive only
            self._plt.show()


# ANSI helpers for the console view. Truecolor is nearly universal in modern
# terminals, and degrading to plain characters costs one flag.
def _ansi(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return "\033[38;2;%d;%d;%dm" % (r, g, b)


_RESET = "\033[0m"


class LiveConsoleView:
    """Render the search into the terminal, in place, with ANSI colors.

    No display server, no backend, no window: this is the view that works when
    the solve is running on a robot, in a container, or over SSH.
    """

    def __init__(
        self,
        source,
        throttle: float = 0.08,
        color: bool = True,
        stream=None,
        theme="dark",
    ):
        self.grid = _grid_of(source)
        self.scenario = source if isinstance(source, Scenario) else None
        self.throttle = throttle
        self.color = color and (stream or sys.stdout).isatty()
        self.stream = stream or sys.stdout
        self.theme = theme_module.get_theme(theme)
        self.expansions = 0
        self.conflicts = 0
        self._last_draw = 0.0
        self._lines_drawn = 0
        self._paths: Dict[str, List] = {}
        self._conflict_cell = None
        self._done = False

    def __enter__(self) -> "LiveConsoleView":
        return self

    def __exit__(self, *exc_info) -> None:
        if not self._done:
            self._render(force=True)

    def __call__(self, event) -> None:
        if event.kind == "expand":
            self.expansions += 1
            paths = event.get("paths")
            if paths:
                self._paths = paths
        elif event.kind == "agent_planned":
            self._paths[event["agent"]] = event["path"]
        elif event.kind == "conflict":
            self.conflicts += 1
            self._conflict_cell = event.get("cell")
        elif event.kind in ("solved", "failed"):
            paths = event.get("paths")
            if paths:
                self._paths = paths
            self._conflict_cell = None
            self._done = True
            self._render(force=True, footer=self._footer(event))
            return
        self._render()

    def _footer(self, event) -> str:
        if event.kind == "solved":
            text = "solved · cost %s · makespan %s · %d nodes" % (
                event.get("cost"),
                event.get("makespan"),
                self.expansions,
            )
            return self._paint(text, theme_module.STATUS["good"])
        return self._paint(
            "failed: %s" % event.get("reason", ""), theme_module.STATUS["critical"]
        )

    def _paint(self, text: str, hex_color: str) -> str:
        return "%s%s%s" % (_ansi(hex_color), text, _RESET) if self.color else text

    def _render(self, force: bool = False, footer: Optional[str] = None) -> None:
        now = time.perf_counter()
        if not force and now - self._last_draw < self.throttle:
            return
        self._last_draw = now

        names = list(self._paths)
        colors = self.theme.color_map(names)
        # Later agents win a shared cell; the head of each path is drawn last so
        # current positions are always visible.
        canvas = [
            [
                ("#" if not self.grid.is_free((r, c)) else "·", None)
                for c in range(self.grid.width)
            ]
            for r in range(self.grid.height)
        ]
        for name in names:
            for cell in self._paths[name]:
                if self.grid.is_free(cell):
                    canvas[cell[0]][cell[1]] = ("•", colors[name])
        for name in names:
            path = self._paths[name]
            head, tail = path[0], path[-1]
            canvas[head[0]][head[1]] = (name[0].lower(), colors[name])
            canvas[tail[0]][tail[1]] = (name[0].upper(), colors[name])
        if self._conflict_cell is not None:
            r, c = self._conflict_cell
            canvas[r][c] = ("X", theme_module.STATUS["critical"])

        lines = []
        for row in canvas:
            cells = []
            for char, hex_color in row:
                if hex_color and self.color:
                    cells.append("%s%s%s" % (_ansi(hex_color), char, _RESET))
                elif hex_color is None and char == "#" and self.color:
                    cells.append("%s#%s" % (_ansi(self.theme.muted), _RESET))
                else:
                    cells.append(char)
            lines.append(" ".join(cells))
        lines.append(
            footer
            if footer is not None
            else "searching · node %d · %d conflicts"
            % (self.expansions, self.conflicts)
        )

        if self._lines_drawn:
            self.stream.write("\033[%dA" % self._lines_drawn if self.color else "\n")
        self.stream.write("\n".join(lines) + "\n")
        self.stream.flush()
        self._lines_drawn = len(lines)
