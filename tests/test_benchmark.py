import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pytest

import pymapf
from pymapf.benchmark import (
    BenchmarkReport,
    RunResult,
    aggregate,
    compare_algorithms,
    run_once,
    scaling_study,
)


def test_run_once_reports_metrics():
    scenario = pymapf.build_scenario("corner_swap", n_agents=4)
    row = run_once(scenario, "cbs")
    assert row.solved and row.valid
    assert row.sum_of_costs > 0
    assert row.makespan > 0
    assert row.runtime >= 0
    assert row.n_agents == 4


def test_repeats_produce_a_stdev():
    scenario = pymapf.build_scenario("empty_room", n_agents=2, seed=5)
    row = run_once(scenario, "prioritized", repeats=3)
    assert row.repeats == 3
    assert row.runtime_stdev >= 0


def test_unsupported_solver_kwargs_are_dropped():
    # `time_limit` exists on CBS but not on prioritized planning; a benchmark
    # configured once must run both.
    scenario = pymapf.build_scenario("empty_room", n_agents=2, seed=1)
    assert run_once(scenario, "prioritized", time_limit=5).solved


def test_compare_algorithms_covers_the_cross_product():
    report = compare_algorithms(["corner_swap", "empty_room"], ["cbs", "prioritized"])
    assert len(report) == 4
    assert report.algorithms == ["cbs", "prioritized"]
    assert report.scenarios == ["corner_swap", "empty_room"]


def test_progress_callback_is_invoked_per_run():
    seen = []
    compare_algorithms(["empty_room"], ["cbs", "prioritized"], on_result=seen.append)
    assert len(seen) == 2
    assert all(isinstance(row, RunResult) for row in seen)


def test_scaling_study_labels_instances_and_keeps_agent_counts():
    report = scaling_study(
        "random_obstacles",
        agent_counts=(2, 3),
        algorithms=("prioritized",),
        seeds=(0,),
    )
    assert len(report) == 2
    assert sorted(row.n_agents for row in report) == [2, 3]
    assert all("seed=0" in row.scenario for row in report)


def test_solver_kwargs_reach_the_solver_not_the_builder():
    report = scaling_study(
        "empty_room",
        agent_counts=(2,),
        algorithms=("cbs",),
        seeds=(0,),
        solver_kwargs={"time_limit": 5.0},
        height=8,
        width=8,
    )
    assert len(report) == 1


def test_report_filtering_and_success_rate():
    report = BenchmarkReport(
        [
            RunResult("s1", "cbs", 2, True, 0.1, 10, 5, 3, True),
            RunResult("s2", "cbs", 2, False, 0.2),
            RunResult("s1", "pp", 2, True, 0.05, 12, 6, 2, True),
        ]
    )
    assert report.success_rate("cbs") == 0.5
    assert report.success_rate("pp") == 1.0
    assert report.success_rate("missing") == 0.0
    assert len(report.filter(algorithm="cbs")) == 2
    assert report.series("cbs", "n_agents", "sum_of_costs") == ([2], [10])


def test_aggregate_averages_over_seeds():
    report = BenchmarkReport(
        [
            RunResult("a", "cbs", 4, True, 0.10, 10),
            RunResult("b", "cbs", 4, True, 0.20, 20),
            RunResult("c", "cbs", 8, True, 0.40, 40),
            RunResult("d", "cbs", 8, False, 9.99),  # unsolved runs are excluded
        ]
    )
    xs, means = aggregate(report, "n_agents", "runtime")["cbs"]
    assert xs == [4, 8]
    assert means[0] == pytest.approx(0.15)
    assert means[1] == pytest.approx(0.40)


def test_table_renders_every_row():
    report = compare_algorithms(["empty_room"], ["cbs"])
    table = report.table()
    assert "scenario" in table
    assert "empty_room" in table
    assert len(table.splitlines()) == 3  # header, rule, one row


def test_to_csv_and_json(tmp_path):
    report = compare_algorithms(["empty_room"], ["cbs"])
    destination = report.to_csv(str(tmp_path / "results.csv"))
    with open(destination) as handle:
        lines = handle.read().splitlines()
    assert lines[0].startswith("scenario,algorithm")
    assert len(lines) == 2
    assert report.to_json()[0]["algorithm"] == "cbs"
