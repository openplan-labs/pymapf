"""Smoke tests for the visualisation layer.

These assert that every figure builds, carries the artists it should, and can be
written to disk -- pixels are not compared, but a broken axis, an exception in a
draw callback or a silently empty plot all fail here.
"""

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import pymapf
from pymapf import viz
from pymapf.benchmark import compare_algorithms, scaling_study


@pytest.fixture(scope="module")
def scenario():
    return pymapf.build_scenario("warehouse", n_agents=5, seed=1)


@pytest.fixture(scope="module")
def solution(scenario):
    return pymapf.solve(scenario.to_problem(), "prioritized")


@pytest.fixture(scope="module")
def trace(scenario):
    recorder = pymapf.SearchTrace()
    pymapf.solve(scenario.to_problem(), "cbs", observer=recorder, time_limit=5)
    return recorder


def test_theme_colors_are_stable_per_agent_index():
    theme = viz.get_theme("dark")
    mapping = theme.color_map(["a", "b", "c"])
    assert mapping["a"] == theme.agent_color(0)
    assert mapping["c"] == theme.agent_color(2)
    # A ninth agent reuses a hue but changes marker, so identity survives.
    assert theme.agent_color(8) == theme.agent_color(0)
    assert theme.agent_marker(8) != theme.agent_marker(0)


def test_unknown_theme_is_rejected():
    with pytest.raises(ValueError):
        viz.get_theme("neon")


def test_plot_grid_draws_obstacles(scenario):
    ax = viz.plot_grid(scenario.grid, title="map")
    assert ax.patches  # one rectangle per wall
    assert ax.get_title() == "map"


def test_plot_scenario_marks_starts_and_goals(scenario):
    ax = viz.plot_scenario(scenario)
    # two markers per agent (start + goal)
    assert len(ax.lines) >= 2 * scenario.n_agents


def test_plot_solution_draws_one_rail_per_agent(scenario, solution):
    ax = viz.plot_solution(solution, scenario)
    assert len(ax.lines) >= len(solution.paths)
    assert "cost" in ax.get_xlabel()


def test_plot_solution_accepts_a_problem_as_source(scenario, solution):
    ax = viz.plot_solution(solution, scenario.to_problem())
    assert ax is not None


def test_plot_congestion_has_a_colorbar(scenario, solution):
    ax = viz.plot_congestion(solution, scenario)
    assert ax.images
    assert ax.figure.axes[-1].get_ylabel() == "agent-timesteps"


def test_plot_spacetime_is_three_dimensional(scenario, solution):
    ax = viz.plot_spacetime(solution, scenario)
    assert hasattr(ax, "get_zlim")


def test_plot_timeline_has_a_row_per_agent(scenario, solution):
    ax = viz.plot_timeline(solution)
    assert len(ax.get_yticks()) == len(solution.paths)


def test_compare_solutions_handles_a_failed_run(scenario, solution):
    figure = viz.compare_solutions({"ok": solution, "failed": None}, scenario)
    assert len(figure.axes) == 2
    texts = [t.get_text() for t in figure.axes[1].texts]
    assert "no solution" in texts


def test_save_writes_a_file(tmp_path, scenario, solution):
    destination = viz.save(
        viz.plot_solution(solution, scenario), str(tmp_path / "s.png")
    )
    assert path.getsize(destination) > 0


def test_animate_solution_produces_frames(scenario, solution):
    animation = viz.animate_solution(solution, scenario, substeps=2, hold_frames=1)
    # matplotlib renamed save_count -> _save_count; accept either.
    assert getattr(animation, "_save_count", None) or getattr(
        animation, "save_count", None
    )
    animation._init_draw()
    artists = animation._func(3)
    assert artists


def test_animation_saves_a_gif(tmp_path, scenario, solution):
    animation = viz.animate_solution(solution, scenario, substeps=1, hold_frames=0)
    destination = viz.save_animation(
        animation, str(tmp_path / "plan.gif"), fps=8, dpi=50
    )
    assert path.getsize(destination) > 0


def test_animate_search_replays_a_trace(scenario, trace):
    animation = viz.animate_search(trace, scenario, max_nodes=5)
    animation._func(0)
    animation._func(2)
    assert len(animation._fig.axes) == 2


def test_animate_search_rejects_an_empty_trace(scenario):
    with pytest.raises(ValueError):
        viz.animate_search(pymapf.SearchTrace(), scenario)


def test_charts_render_from_a_report():
    report = compare_algorithms(["corner_swap", "empty_room"], ["cbs", "prioritized"])
    ax = viz.plot_cost_comparison(report)
    assert ax.patches
    assert len(ax.get_xticks()) == 2

    rates = viz.plot_success_rate(report)
    assert rates.patches


def test_scaling_chart_uses_a_log_axis():
    report = scaling_study(
        "random_obstacles",
        agent_counts=(2, 4),
        algorithms=("prioritized",),
        seeds=(0,),
    )
    ax = viz.plot_scaling(report, y="runtime")
    assert ax.get_yscale() == "log"
    assert ax.lines


def test_cost_curve_chart(trace):
    ax = viz.plot_cost_curve({"cbs": trace})
    assert ax.lines


def test_dashboard_has_four_panels():
    report = scaling_study(
        "random_obstacles",
        agent_counts=(2, 4),
        algorithms=("cbs", "prioritized"),
        seeds=(0,),
        solver_kwargs={"time_limit": 5.0},
    )
    figure = viz.dashboard(report)
    assert len(figure.axes) >= 4


def test_live_console_view_renders_without_a_display(scenario):
    import io

    stream = io.StringIO()
    view = viz.LiveConsoleView(scenario, throttle=0.0, color=False, stream=stream)
    with view:
        pymapf.solve(scenario.to_problem(), "prioritized", observer=view)
    output = stream.getvalue()
    assert "solved" in output
    assert "#" in output  # the map itself was drawn


def test_live_console_view_reports_failure():
    import io

    from pymapf.core import Agent, GridMap, MAPFProblem

    # A one-cell-wide corridor where two agents must swap: no plan exists.
    grid = GridMap([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
    problem = MAPFProblem(
        grid, [Agent("a", (1, 0), (1, 2)), Agent("b", (1, 2), (1, 0))]
    )

    stream = io.StringIO()
    view = viz.LiveConsoleView(grid, throttle=0.0, color=False, stream=stream)
    with view:
        pymapf.solve(problem, "cbs", observer=view, max_expansions=20)
    assert "failed" in stream.getvalue()
