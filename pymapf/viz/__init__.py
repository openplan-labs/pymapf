"""Visualisation for PyMAPF: static figures, animations, live views and charts.

matplotlib is an *optional* dependency -- the solvers themselves need nothing
beyond the standard library, and that is what lets them run in the browser under
Pyodide. Importing this package raises a clear, actionable error if matplotlib
is missing rather than a bare ``ModuleNotFoundError`` from three frames deep.

Quick tour::

    import pymapf
    from pymapf import viz

    scenario = pymapf.build_scenario("warehouse", n_agents=8)
    solution = pymapf.solve(scenario.to_problem(), "prioritized")

    viz.plot_solution(solution, scenario)                 # static figure
    viz.save(viz.plot_timeline(solution), "timeline.png")  # who waits, when
    anim = viz.animate_solution(solution, scenario)        # agents in motion
    viz.save_animation(anim, "plan.gif", fps=16)
"""

from __future__ import annotations

try:  # pragma: no cover - exercised by the import itself
    import matplotlib  # noqa: F401
except ModuleNotFoundError as error:  # pragma: no cover
    raise ModuleNotFoundError(
        "pymapf.viz needs matplotlib, which is optional: the solvers run "
        "without it. Install the extra with `pip install pymapf[viz]` (or "
        "`pip install matplotlib`)."
    ) from error

from .animate import animate_search, animate_solution, save as save_animation, to_jshtml
from .charts import (
    dashboard,
    plot_cost_comparison,
    plot_cost_curve,
    plot_scaling,
    plot_success_rate,
)
from .live import LiveConsoleView, LiveSolveView
from .plots import (
    compare_solutions,
    plot_congestion,
    plot_grid,
    plot_scenario,
    plot_solution,
    plot_spacetime,
    plot_timeline,
    save,
)
from .theme import DARK, LIGHT, STATUS, Theme, apply, get_theme

__all__ = [
    # static
    "plot_grid",
    "plot_scenario",
    "plot_solution",
    "plot_congestion",
    "plot_spacetime",
    "plot_timeline",
    "compare_solutions",
    "save",
    # animation
    "animate_solution",
    "animate_search",
    "save_animation",
    "to_jshtml",
    # live
    "LiveSolveView",
    "LiveConsoleView",
    # charts
    "plot_scaling",
    "plot_cost_comparison",
    "plot_success_rate",
    "plot_cost_curve",
    "dashboard",
    # theme
    "Theme",
    "DARK",
    "LIGHT",
    "STATUS",
    "apply",
    "get_theme",
]
