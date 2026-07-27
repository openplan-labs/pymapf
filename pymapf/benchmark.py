"""A small, dependency-free benchmark harness for MAPF solvers.

Comparing algorithms is the whole point of a framework, so measuring them
should not require a notebook and a stopwatch. This module runs solvers over
:mod:`pymapf.scenarios` instances, records cost/time/expansion metrics, and
returns tabular results that :mod:`pymapf.viz.charts` turns into figures.

Only the standard library is used (``time.perf_counter`` for timing), so the
same benchmarks run in CI and in the browser under Pyodide::

    from pymapf.benchmark import compare_algorithms
    report = compare_algorithms(["corner_swap", "warehouse"], ["cbs", "prioritized"])
    print(report.table())
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from .core.solver import get_solver
from .scenarios import Scenario, build_scenario

__all__ = [
    "RunResult",
    "BenchmarkReport",
    "run_once",
    "compare_algorithms",
    "scaling_study",
]


@dataclass
class RunResult:
    """Metrics for one (scenario, algorithm) run."""

    scenario: str
    algorithm: str
    n_agents: int
    solved: bool
    runtime: float
    sum_of_costs: Optional[int] = None
    makespan: Optional[int] = None
    expansions: int = 0
    valid: bool = False
    runtime_stdev: float = 0.0
    repeats: int = 1
    note: str = ""

    def as_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BenchmarkReport:
    """A list of :class:`RunResult` rows plus the usual ways to look at them."""

    rows: List[RunResult] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def add(self, row: RunResult) -> None:
        self.rows.append(row)

    def filter(self, **criteria) -> "BenchmarkReport":
        """Rows matching every ``field=value`` pair."""
        return BenchmarkReport(
            [
                row
                for row in self.rows
                if all(getattr(row, key) == value for key, value in criteria.items())
            ]
        )

    @property
    def algorithms(self) -> List[str]:
        return sorted({row.algorithm for row in self.rows})

    @property
    def scenarios(self) -> List[str]:
        # Insertion order, de-duplicated: scenarios usually have a meaningful
        # order (increasing difficulty) that sorting would destroy.
        seen = []
        for row in self.rows:
            if row.scenario not in seen:
                seen.append(row.scenario)
        return seen

    def success_rate(self, algorithm: str) -> float:
        rows = [r for r in self.rows if r.algorithm == algorithm]
        if not rows:
            return 0.0
        return sum(1 for r in rows if r.solved) / len(rows)

    def series(self, algorithm: str, x: str, y: str):
        """``(xs, ys)`` for solved runs of ``algorithm``, sorted by ``x``."""
        points = [
            (getattr(r, x), getattr(r, y))
            for r in self.rows
            if r.algorithm == algorithm and r.solved and getattr(r, y) is not None
        ]
        points.sort(key=lambda p: p[0])
        return [p[0] for p in points], [p[1] for p in points]

    def to_csv(self, path: str) -> str:
        import csv

        fields = list(RunResult.__dataclass_fields__)
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row.as_dict())
        return path

    def to_json(self) -> List[Dict]:
        return [row.as_dict() for row in self.rows]

    def table(self) -> str:
        """A fixed-width text table -- what scripts print to the terminal."""
        headers = [
            "scenario",
            "algorithm",
            "agents",
            "solved",
            "cost",
            "makespan",
            "expansions",
            "ms",
        ]
        lines = [
            [
                row.scenario,
                row.algorithm,
                str(row.n_agents),
                "yes" if row.solved else "no",
                "-" if row.sum_of_costs is None else str(row.sum_of_costs),
                "-" if row.makespan is None else str(row.makespan),
                str(row.expansions),
                "%.1f" % (1000 * row.runtime),
            ]
            for row in self.rows
        ]
        widths = [
            max(len(headers[i]), max((len(line[i]) for line in lines), default=0))
            for i in range(len(headers))
        ]
        rule = "-+-".join("-" * w for w in widths)
        out = [
            " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
            rule,
        ]
        out += [
            " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(line))
            for line in lines
        ]
        return "\n".join(out)


def _supported(algorithm: str, kwargs: Dict) -> Dict:
    """Drop constructor kwargs a given solver does not accept.

    Benchmarks are configured once and run across every algorithm, so
    ``time_limit=5`` must not blow up on a solver that has no such knob.
    """
    import inspect

    from .core.solver import _REGISTRY

    cls = _REGISTRY.get(algorithm.lower())
    if cls is None:
        return kwargs
    accepted = inspect.signature(cls.__init__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in accepted}


def run_once(
    scenario: Scenario,
    algorithm: str = "cbs",
    repeats: int = 1,
    **solver_kwargs,
) -> RunResult:
    """Solve ``scenario`` with ``algorithm`` and measure it.

    ``repeats > 1`` re-runs the solve and keeps the *median* runtime, which is
    what you want on a noisy machine (a mean is dragged around by a single
    scheduling hiccup). Solver options that an algorithm does not accept (say
    ``time_limit`` for prioritized planning) are ignored rather than raising.
    """
    problem = scenario.to_problem()
    solver = get_solver(algorithm, **_supported(algorithm, solver_kwargs))

    timings: List[float] = []
    solution = None
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        solution = solver.solve(problem)
        timings.append(time.perf_counter() - started)

    return RunResult(
        scenario=scenario.name,
        algorithm=algorithm,
        n_agents=scenario.n_agents,
        solved=solution is not None,
        runtime=statistics.median(timings),
        runtime_stdev=statistics.stdev(timings) if len(timings) > 1 else 0.0,
        repeats=len(timings),
        sum_of_costs=solution.sum_of_costs if solution else None,
        makespan=solution.makespan if solution else None,
        expansions=solution.expansions if solution else 0,
        valid=solution.is_valid() if solution else False,
        note="" if solution else "no solution returned",
    )


def compare_algorithms(
    scenarios: Sequence,
    algorithms: Sequence[str] = ("cbs", "prioritized"),
    repeats: int = 1,
    on_result: Optional[Callable[[RunResult], None]] = None,
    **solver_kwargs,
) -> BenchmarkReport:
    """Run every algorithm on every scenario.

    ``scenarios`` accepts :class:`~pymapf.scenarios.Scenario` objects or names
    of registered builders. ``on_result`` is called after each run, which is how
    long benchmarks report progress instead of going quiet for a minute.
    """
    report = BenchmarkReport()
    for entry in scenarios:
        scenario = build_scenario(entry) if isinstance(entry, str) else entry
        for algorithm in algorithms:
            row = run_once(scenario, algorithm, repeats=repeats, **solver_kwargs)
            report.add(row)
            if on_result is not None:
                on_result(row)
    return report


def scaling_study(
    builder: str = "random_obstacles",
    agent_counts: Iterable[int] = (2, 4, 6, 8, 10),
    algorithms: Sequence[str] = ("cbs", "prioritized"),
    seeds: Sequence[int] = (0, 1, 2),
    repeats: int = 1,
    on_result: Optional[Callable[[RunResult], None]] = None,
    solver_kwargs: Optional[Dict] = None,
    **builder_kwargs,
) -> BenchmarkReport:
    """Measure how each algorithm scales with the number of agents.

    Every ``(agent count, seed)`` pair is a fresh instance solved by every
    algorithm; averaging over seeds is what keeps one unlucky map from
    dominating the curve.

    Keyword arguments go to the *scenario builder*; solver options (such as
    ``time_limit``) go in the ``solver_kwargs`` dict, since the two namespaces
    would otherwise collide.
    """
    solver_kwargs = solver_kwargs or {}
    report = BenchmarkReport()
    for count in agent_counts:
        for seed in seeds:
            scenario = build_scenario(
                builder, n_agents=count, seed=seed, **builder_kwargs
            )
            for algorithm in algorithms:
                row = run_once(scenario, algorithm, repeats=repeats, **solver_kwargs)
                row.scenario = "%s/n=%d/seed=%d" % (builder, count, seed)
                row.n_agents = count
                report.add(row)
                if on_result is not None:
                    on_result(row)
    return report


def aggregate(report: BenchmarkReport, x: str = "n_agents", y: str = "runtime"):
    """Mean of ``y`` per ``(algorithm, x)`` -- the shape charts want.

    Returns ``{algorithm: (xs, means)}`` over solved runs only.
    """
    buckets: Dict[str, Dict[float, List[float]]] = {}
    for row in report.rows:
        if not row.solved or getattr(row, y) is None:
            continue
        buckets.setdefault(row.algorithm, {}).setdefault(getattr(row, x), []).append(
            float(getattr(row, y))
        )
    out = {}
    for algorithm, by_x in buckets.items():
        xs = sorted(by_x)
        out[algorithm] = (xs, [statistics.fmean(by_x[value]) for value in xs])
    return out
