# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0]

### Added

- **Interactive playground** (`docs/`, published to GitHub Pages): edit a map,
  pick a solver and watch the search resolve conflicts in the browser. It runs
  the library's own source under Pyodide in a web worker, with a JavaScript port
  of the solvers (`docs/mapf.js`) as an instant-response fallback. The Python
  tab runs arbitrary user code against the real package; the benchmark tab runs
  a sweep and charts it live.
- **Weighted CBS** (`pymapf.algorithms.WeightedCBS`, registered as `"wcbs"`):
  bounded-suboptimal focal search. Returns a solution costing at most
  `weight x optimal`, typically orders of magnitude faster than optimal CBS.
- **Search instrumentation** (`pymapf.core.trace`): solvers accept an
  `observer` and stream `SearchEvent`s (`root`, `expand`, `conflict`, `branch`,
  `agent_planned`, `solved`, `failed`). `SearchTrace` records them with
  aggregates (`summary()`, `cost_curve()`); observing is opt-in and costs one
  branch per event when unused.
- **Scenario library** (`pymapf.scenarios`): six deterministic instance
  families -- `empty_room`, `random_obstacles`, `warehouse`, `maze`,
  `bottleneck`, `corner_swap` -- plus `from_ascii`/`to_ascii` for hand-written
  maps and a name-based registry (`build_scenario`, `available_scenarios`).
- **Benchmark harness** (`pymapf.benchmark`): `run_once`, `compare_algorithms`,
  `scaling_study` and `aggregate`, returning a `BenchmarkReport` with a text
  table, CSV and JSON export. Median-of-repeats timing; solver options that an
  algorithm does not accept are dropped rather than raising.
- **Visualisation** (`pymapf.viz`, optional `[viz]` extra): `plot_solution`,
  `plot_scenario`, `plot_congestion`, `plot_spacetime` (3D space-time cube),
  `plot_timeline` (moving vs waiting), `compare_solutions`, benchmark charts
  (`plot_scaling`, `plot_cost_comparison`, `plot_success_rate`,
  `plot_cost_curve`, `dashboard`), animations (`animate_solution`,
  `animate_search`, GIF/MP4 export) and live views (`LiveSolveView` for a
  window, `LiveConsoleView` for a terminal). One shared, colorblind-safe theme
  in light and dark.
- `Solution.position_at`, `Solution.congestion`, `Solution.as_dict` and
  `Solution.runtime`; `count_conflicts` in `pymapf.core.solver`.
- `time_limit` on CBS and weighted CBS: a hard instance now reports a failure
  with a reason instead of running unbounded.
- Scripts: `generate_gallery.py` (every figure in the docs),
  `make_promo.py` (the promo film -- every number in it is measured at render
  time), `build_web_bundle.py` (the playground's copy of the library).

### Changed

- **CBS expands best-first on `(cost, conflicts)`** instead of cost alone. Among
  equal-cost nodes it now prefers the one closest to conflict-free, which stops
  it breadth-first-ing through equal-cost plateaus on corridor-heavy maps.
- **Packaging**: `install_requires` is now empty -- the solver framework has no
  third-party dependencies. matplotlib, numpy and scipy moved to the `viz`,
  `decentralized`, `legacy` and `all` extras. Existing users of the
  decentralized planners should install `pymapf[all]`.
- `MAPFSolver.solve` takes an optional `observer` argument. Custom solvers that
  define `solve(self, problem)` keep working unless an observer is passed.

## [0.2.0]

### Added

- **Centralized MAPF framework** (`pymapf.core`): an algorithm-agnostic layer
  that makes PyMAPF usable as a library long term.
  - `GridMap`: a deterministic, explicit occupancy grid (build a specific
    scenario instead of the random-wall-only `World`); includes
    `GridMap.from_world`.
  - `MAPFProblem`, `Agent`, `Solution` (with `makespan`, `sum_of_costs`,
    `is_valid`, `first_conflict`), and `Constraints`.
  - Pluggable heuristics (`manhattan`, `euclidean`, `chebyshev`, `octile`) with
    name/callable resolution, replacing the global `common.HEURISTIC` flag.
  - Abstract `MAPFSolver` plus a name-based solver **registry**
    (`register_solver`, `get_solver`, `available_solvers`) so new algorithms are
    discoverable and swappable.
  - Conflict detection utilities (`find_first_conflict`, `Conflict`).
- **New algorithms** (`pymapf.algorithms`):
  - `space_time_astar`: a constraint-aware, provably terminating low-level
    space-time A* shared by the solvers.
  - `PrioritizedPlanning` (`"prioritized"`): cooperative A* with space-time
    reservations.
  - `ConflictBasedSearch` (`"cbs"`): the canonical two-level optimal
    (sum-of-costs) MAPF algorithm.
- Top-level convenience API: `pymapf.solve(problem, algorithm="cbs", **kwargs)`
  and re-exports of the core framework types.
- Deterministic test suite for the new modules (heuristics, grid, low-level
  search, prioritized planning, CBS, and the solver registry).

### Changed

- `pymapf.__version__` bumped to `0.2.0`.

### Notes

- The existing reactive/decentralized planners (`MultiAgentNMPC`,
  `MultiAgentVelocityObstacle`) and the legacy `CooperativeAStar` are unchanged
  and remain available.
