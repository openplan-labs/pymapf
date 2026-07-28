# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0]

Formation control, and the post-2020 flocking models from the Albani / Ferrante
/ Manoni / Saska group.

### Added

- **`pymapf.swarm.formation`** -- formation control on the displacement /
  distance / bearing taxonomy of Oh, Park and Ahn (2015). Four controllers,
  seven shapes, and the rigidity theory that says when each one can work.
  - `DisplacementFormation` (alias `formation`) -- relative positions in a
    shared frame; fixes the formation up to translation. Converges in 3.6 s.
  - `DistanceFormation` -- range only, no shared frame; fixes it up to
    translation, rotation and reflection (Krick et al. 2009). Reaches every
    desired distance to 1e-13.
  - `BearingFormation` -- direction only, what a camera measures; fixes it up to
    translation and *scale* (Zhao and Zelazo 2016), with an optional
    `scale_gain` to pin the size.
  - `LeaderFollower` -- leaders track the mission, followers hold offsets
    (Balch and Arkin 1998).
  - `FormationShape` objects -- line, V (with sweep and dihedral), circle, grid,
    cube, sphere, custom -- with `register_shape`/`get_shape`/`available_shapes`,
    the same registry pattern as behaviors and coverage controllers.
  - `assign_slots` -- exact Hungarian assignment of agents to slots, O(n^3) and
    dependency-free. Assigning by index makes agents cross the formation to
    reach a slot someone else is standing next to.
  - `is_infinitesimally_rigid` -- rank test on the rigidity matrix. Predicts the
    only two configurations in the whole sweep where distance and bearing
    control fail: a collinear target in 2D and a planar target in 3D. Those
    controllers now warn before running rather than converging quietly to the
    wrong shape.
  - `formation_error` -- fits the shape under exactly the symmetry group the
    controller's *sensing* leaves free (rotation, scale, reflection each
    optional), solving pose and correspondence jointly.
- **Two post-2020 flocking models** (ten total):
  - `MinimalisticFlocking` -- Amorim, Nascimento, Chaudhary, Ferrante and Saska
    (2024). Relative range and bearing only: no GPS, no compass, no
    communication, no velocity sensing, and a cohesive flock still emerges and
    agrees on a direction nobody transmitted. Order 0.99 with zero separation
    violations.
  - `DistributedThreeDimensional` -- Albani, Manoni, Saska and Ferrante (2022).
    Proximal control made anisotropic, because a multirotor is: climbing is
    expensive and a drone below another sits in its downwash. Settles into a
    lattice with vertical-to-horizontal spread 0.44 against 0.71 for the
    isotropic law.
- `docs/survey.md` gains section 6.2c on formation control, with measured
  convergence times for all four controllers and the three findings below.

### Fixed

- **The Lennard-Jones well was centred at the wrong distance** in every proximal
  controller. A potential written with length parameter `sigma` has its force
  zero at `2^(1/m) sigma`, not at `sigma` -- so passing `reference_distance`
  straight in built a controller whose rest spacing was 41% wider than its own
  configuration. In open space the gap compounds until the outer agents leave
  interaction range. `equilibrium_sigma()` solves for the minimum instead.
- **A range-limited interaction graph is the wrong constraint set for distance
  and bearing control.** It is not rigid in general, and it *changes* as the
  swarm moves, so the constraints being descended shift underneath the descent:
  pairs pushed apart to their desired distance, left sensing range, and the
  pairs that should have pulled them back were never in it. Formation error grew
  from 3.6 to 26.8. The graph is now built once from the target shape and
  augmented until rigid.
- **A waypoint was applied as a per-agent attraction**, which is a contraction,
  which is a deformation. It squashed every formation and collapsed the
  bearing-based one onto the waypoint entirely -- scale being exactly the
  freedom bearings do not constrain. It is now a common translation computed
  from the formation centroid, which lies in the null space of all four laws.
- **A leader with no mission chased the swarm centroid**, closing a feedback
  loop through its own followers: they track it, it tracks them, and the
  formation never settles. Leaders now hold station.
- **`formation_error` fitted the pose before knowing the correspondence**,
  scoring exactly-converged formations as failures -- a distance controller
  satisfying its entire target distance matrix to 1e-13 was reported at error
  3.58.

### Changed

- `MinimalisticFlocking` defaults to a topological neighbourhood with k = 8
  rather than the k = 3 of `ActiveElastic`. A bounded attraction needs more
  incident edges than a spring does: over ten seeds with twenty agents in the
  plane the flock fragmented in 7 runs at k = 6 against 1 at the default here.
  In 3D -- what the paper is about -- every seed converges either way.

## [0.5.0]

An object-oriented swarm layer, more flocking and coverage algorithms, and
Gaussian-mixture distribution control.

### Added

- **`pymapf.swarm`** -- the decentralized side rebuilt on the same conventions as
  the planners: an abstract base class per family, a name registry, and swappable
  strategy objects.
  - `Behavior` + `register_behavior`/`get_behavior`/`available_behaviors`, so a
    controller is chosen with a string and compared with a loop.
  - `CompositeBehavior`: new controllers by weighted composition rather than by
    writing a new class.
  - `Neighborhood` strategies -- metric, topological (Ballerini et al. 2008),
    forward-cone, and Gaussian-kernel (Manoni et al. 2022) -- usable by any
    behavior. Measured: topological k=5 gives better spacing than a metric radius
    (2.17 m vs 1.60 m) with a third of the connectivity.
  - `SwarmSimulator` with reflecting bounds, obstacles, observers and metrics.
- **Four more flocking models** (eight total):
  - `CuckerSmale` -- power-law velocity consensus (Cucker and Smale 2007).
  - `ProximalControl` -- Lennard-Jones proximal potential plus distance-dependent
    alignment allowance, in the Vasarhelyi et al. (2018) style.
  - `ActiveElastic` -- Ferrante et al. (2012, 2013). The swarm as an active
    elastic solid: alignment *emerges* from the elastic modes, with no agent ever
    sensing a neighbour's velocity or heading. Measured at order 0.95 with zero
    separation violations.
  - `GaussianKernelFlocking` -- Manoni, Albani et al. (2022) kernel arbitration,
    with an adaptive kernel width.
- **Coverage over pluggable domains** (`pymapf.swarm.domain`): planar, disk,
  sphere, hemisphere, annulus and arbitrary mesh. One Lloyd implementation now
  deploys a team on any of them (84-96% cost reduction across the six).
- **Five coverage controllers** (`pymapf.swarm.coverage`): `lloyd`,
  `limited_range`, `adaptive` (estimates the density online, Schwager et al.
  2009), `gmm` (splits the team across mixture components) and `time_varying`
  (pursues moving targets, Manoni et al. 2024).
- **Density fields** (`pymapf.swarm.density`): `GaussianMixtureDensity` with
  responsibilities, sampling and **EM fitting**, plus time-varying and sampled
  fields. EM recovers generating means to within 0.15 on 500 samples, and
  covering the *fitted* density is as good as covering the true one.
- **Swarm distribution control** (`pymapf.swarm.distribution`): `DensityMatching`
  (kernel-density gradient flow) and `MixtureAssignment` (probabilistic guidance
  in the spirit of Bandyopadhyay et al. 2017). The latter hits the mixing weights
  exactly -- 12/12/6 agents for a 0.4/0.4/0.2 mixture.

### Changed

- `pymapf.decentralized.flocking` and `.coverage` are now thin functional
  façades over `pymapf.swarm`; there is one implementation of every model. The
  old API is unchanged and still tested.

### Fixed

- **Mixture-based team assignment never split a team.** Assigning each agent to
  its most-responsible component leaves a clustered fleet entirely on one
  component, and quota-pressure scaling cannot fix it because responsibilities
  are near-degenerate (1e-30 vs 1). Replaced with a capacity-constrained greedy
  allocation, which produces the requested split exactly.
- **Adaptive coverage made its own estimate worse over time.** Fitting only the
  current agent positions is under-determined and self-confirming. It now
  accumulates measurements in a bounded memory -- the cheapest stand-in for the
  persistence-of-excitation condition the original analysis assumes.
- **Unnormalised Gaussian-kernel weights dispersed the flock** (cohesion 48 m
  against 3 m for the metric neighbourhood): the kernel silently scaled the whole
  interaction down, letting self-propulsion outrun cohesion. Weights are now
  normalised to mean 1, so the kernel arbitrates rather than weakens.
- **Active elastic flocking collapsed under a metric neighbourhood**: bounded
  springs plus ~10 neighbours means summed attraction beats local repulsion --
  the same failure mode as Olfati-Saber at r/d = 2. The default is now
  topological, and the model is implemented at velocity level as the papers
  formulate it.

## [0.4.0]

Modern MAPF algorithms, general graphs, decentralized swarm control, a
referenced bibliography, an extended survey, and an experimental section.

### Added

- **Modern solvers**, each with its citation in `REFERENCES.md`:
  - `PIBT` (`"pibt"`) -- priority inheritance with backtracking (Okumura et al.,
    AIJ 2022). One timestep at a time, O(agents x degree) per step.
  - `LaCAM` (`"lacam"`) -- lazy constraints addition search (Okumura, AAAI
    2023). Complete; solved 6 of 7 instances where PIBT livelocked and CBS
    proved a solution existed.
  - `LargeNeighborhoodSearch` (`"lns"`) -- anytime destroy/repair with adaptive
    operator weights (Li et al., IJCAI 2021). Takes PIBT's warehouse plan from
    cost 175 to 113 in two seconds.
  - `sipp` -- safe interval path planning (Phillips and Likhachev, ICRA 2011).
    Verified against space-time A* on 400 randomised constraint sets (identical
    costs) and 115x faster on a long-horizon instance.
- **General graphs** (`pymapf.core.graph.ExplicitGraph`): every solver now runs
  on arbitrary graphs -- roadmaps, warehouse topologies, PRMs -- not just grids.
  Duck-type compatible with `GridMap`, so nothing else changed.
- **Single-agent search primitives** (`pymapf.algorithms.search`): Dijkstra, A*,
  weighted A*, focal search, and `distance_table`/`true_distance` (an exact,
  admissible heuristic, and the only one available on a graph without
  coordinates).
- **Decentralized swarm control**:
  - `decentralized.flocking` -- boids (Reynolds 1987), Vicsek (1995),
    Olfati-Saber (2006) with the paper's action and bump functions, and
    acceleration-based bird-inspired flocking (Iacone, Lejeune, Manoni,
    Manfredi and Albani, 2024), plus a simulator with order/cohesion/safety
    metrics.
  - `decentralized.coverage` -- Voronoi/Lloyd coverage (Cortes et al. 2004),
    limited-range coverage (Bertoncelli, Belal et al., DARS 2022) and
    hemispherical surface coverage (Belal et al., ANTS 2026).
- **`pymapf.experimental`** -- three measured variants registered under `x-*`
  names: congestion-aware PIBT, delay-targeted LNS, and restart-based LaCAM,
  with `python -m pymapf.experimental.study` to reproduce every number.
- **`REFERENCES.md`** -- every algorithm mapped to its source, including what
  is surveyed but not implemented, and an implementation-notes section stating
  every deviation from the cited work.
- **`docs/survey.md`** -- an extension of *Survey of the Multi-Agent Pathfinding
  Solutions* (Lejeune and Sarkar, 2021) covering 2021-2026, the decentralized
  swarm line, and an experiments section reporting measured results including
  the negative ones.

### Changed

- `LaCAM(anytime=True)` spends its leftover budget on randomised restarts rather
  than continuing the search in place. The in-place continuation was measured
  over 28 paired instances and won **zero** of them; restarts won 26.
- `LargeNeighborhoodSearch` exposes `operators()` / `_pick_neighborhood()` so
  destroy operators can be added by subclassing instead of copying the loop.

### Fixed

- **PIBT could return an invalid plan.** Two bugs: an agent could swap with a
  peer that had already committed to its vertex (an edge conflict priority
  inheritance does not rule out on its own), and a failed inheritance chain
  leaked the assignments made deeper in the recursion. Assignments are now
  journalled and rolled back as a unit, and every step is verified before it is
  returned. A 306-run sweep across all scenarios is now clean.
- **Flocking: an unbounded waypoint term starved collision avoidance.** A
  distant migration point consumed the whole acceleration budget, so the final
  clamp scaled separation to nothing. Navigational authority is now capped at a
  share of the budget.
- **Olfati-Saber flocking collapsed.** Its alpha-lattice assumes an interaction
  range of ~1.2x the reference distance; at this library's default sensing range
  (2x) summed attraction beats local repulsion and the flock merges into a
  point. The ratio is now part of the controller.

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
