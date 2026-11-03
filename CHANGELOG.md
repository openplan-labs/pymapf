# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases are cut by [release-please](https://github.com/googleapis/release-please)
from Conventional Commit messages (see CONTRIBUTING.md). Each release entry
holds the notes written under *Unreleased* followed by the generated list of
commits.

## [Unreleased]

### Added

- **Joint trajectory optimisation in continuous space** (`pymapf.trajectory`):
  `plan_joint_trajectories` optimises every vehicle's piecewise-polynomial
  trajectory in R^n in one problem -- minimum snap, subject to endpoints and
  rest conditions, C^k continuity, pairwise separation by the vehicles'
  bodies, sphere obstacles, and speed and acceleration limits. Non-convex
  separation is handled by sequential convex programming with supporting
  half-spaces (conservative, so a feasible subproblem solution is genuinely
  collision-free) inside a trust region. The equality constraints are
  eliminated once into a null-space basis, so each iteration is a solve in
  the reduced coordinates. Checked against SLSQP end to end: the same optimum
  to 3.8e-11 relative.
  - The separation constraints hold *between* the sample times too, by a
    bound on the dip derived from each pair's own relative speed. Without it a
    head-on pair that clears 0.800 m at every sample reaches 0.742 m between
    two of them, which a test measures both ways.
  - Speed and acceleration limits are met exactly by dilating the fleet's
    clock, which preserves every pairwise separation because it scales every
    vehicle equally.
  - `waypoints_from_solution` seeds the optimiser from any discrete
    `Solution`, so the fleet inherits the homotopy a complete MAPF solver
    chose before the continuous optimiser smooths it.
- `pymapf.trajectory.solve_qp`: a dense quadratic program solver in numpy
  alone -- augmented Lagrangian on the inequalities with a damped
  semismooth-Newton inner solve -- and `PiecewisePolynomial`, the
  normalised-time trajectory representation the layer is built on.
- `viz.plot_trajectories`, `viz.plot_separation` and
  `viz.animate_trajectories` draw continuous fleets in 2D and 3D, with the
  bodies at the closest approach and the pairwise separation against what the
  bodies require.
- **Quadrotor docking** (`pymapf.aerial`): a hovering fleet is assigned to
  ground stations, routed and flown down on collision-free, kinematically
  feasible trajectories. `Airspace.build` discretises a box of air into a
  `VoxelGrid` with the ground blocked except at the pads and, with `floor`,
  the layers below it open only in the vertical corridor above each pad;
  `assign_stations` is the Hungarian algorithm on flight distance (exact);
  `plan_docking` stacks it with CBS on the volume and `plan_trajectories`
  with a safety distance derived from the bodies, and `DockingPlan` reports
  the measured closest approach against the required one. Demo: eight
  vehicles, ten pads, a building in the way, 0.70 m required, 1.50 m measured.
- **Straight runs in the scheduler.** `plan_trajectories(...,
  merge_straight=True)` flies consecutive collinear moves as one run under
  one profile and passes the vertices between them at speed; each segment
  lists them on `via`. The temporal network stays a longest-path problem
  because a run's profile is fixed once its length is. Safety constraints
  are now derived per *pair of runs* that share a vertex, from a simulation
  of both vehicles' motion over the interval both are moving -- the same
  simulation the rest-to-rest hand-over used, minus an assumption that made
  runs sharing an endpoint look like a permanent collision. On the warehouse
  plans, a quarter off the makespan under an acceleration limit at the same
  0.5 separation.
- `pymapf.core.assignment.hungarian`: the optimal assignment in pure Python,
  shared by the aerial layer and the swarm formation slots.
- **Demos for the four new layers.** `scripts/generate_feature_demos.py`
  renders one animation and one figure each for kinodynamic execution, 3D
  volumes, decentralized navigation and lifelong MAPF into `.docs/assets`;
  they are embedded in the README sections and the site's gallery. The README
  gains the *From plans to trajectories* section that the feature bullet
  promised.
- **`pymapf.kinodynamic`** -- discrete plans become trajectories a robot can
  follow. `plan_trajectories(solution, limits=KinematicLimits(v_max, a_max,
  safety_distance))` keeps the order in which the plan has agents visit each
  vertex, discards its unit-step timing, and re-derives the timing under
  physical limits as the longest path through a simple temporal network
  (Hönig et al. 2016, MAPF-POST). Moves are rest-to-rest trapezoidal profiles,
  so the speed and acceleration limits hold at every instant; the tests check
  both by finite differences. Rotations on a cycle work, and a margin too long
  for one raises `InfeasibleScheduleError` rather than a schedule that never
  finishes. Heterogeneous fleets pass `limits_by_agent`; general graphs take
  coordinates from `ExplicitGraph.positions`. Pure standard library.
  - The safety margin is derived **per hand-over**, which the paper does not
    do. `required_margin` simulates the leaving and arriving moves under
    their actual profiles and the angle between them and finds the smallest
    delay at which the pair never comes within `safety_distance`. The
    full-speed rule it replaces let a right-angle hand-over between
    rest-to-rest agents get within 0.199 of a 0.5 that was asked for; with the
    derived margin the sampled minimum over a whole warehouse plan is 0.500,
    with or without an acceleration limit.
  - `TrajectorySet.min_separation()` samples what the schedule actually
    achieved in space, because the model covers hand-overs at shared vertices
    and not two agents passing on adjacent ones -- on a unit grid the latter
    keep one cell, and above that the sample is the check.
- **Three-dimensional grids.** `VoxelGrid` is a stack of layers with the same
  two-question interface as `GridMap` -- *is this free?*, *what is adjacent?* --
  so every solver, the heuristics, the benchmark harness and the trajectory
  scheduler run on a volume unchanged; the tests parametrise over every
  registered solver to prove it. 6-connected by default, 26-connected with
  `allow_diagonals`, under the planar corner-cutting rule generalised to every
  face, edge and corner of the box a diagonal cuts through. The geometric
  heuristics take any dimension. Three families join the registry --
  `empty_volume`, `random_blocks`, `stacked_floors` (floors joined by a few
  shafts: the 3D bottleneck) -- and `to_ascii` renders a volume one layer at a
  time. `viz.plot_solution_3d` draws routes threading blocked voxels; the
  planar plotters refuse a volume and say which function to use instead, and
  so does `MAPFEnv`, which stays planar. Measured: eight agents in a 6x6
  footprint cost CBS a median of 794 expansions on the plane and 3 with three
  layers, because the third axis is a way around.
- **Decentralized navigation** (`pymapf.swarm.navigation`): four control laws
  in which every agent has a goal of its own and decides from what it can
  see -- **ORCA** (van den Berg et al. 2011), **buffered Voronoi cells** (Zhou
  et al. 2017), **artificial potential fields** (Khatib 1986) and the
  **social force model** (Helbing and Molnár 1995) -- as `Behavior`
  subclasses registered by name, running in the existing `SwarmSimulator`
  under the existing safety metrics, in any dimension. `circle_swap` is the
  standard benchmark. ORCA moves from *related* to *implemented* in
  `REFERENCES.md`.
  - The two constraint methods choose a velocity by `project_onto_polytope`,
    an exact projection onto half-spaces and a ball: RVO2's incremental
    linear programs generalised to *n* dimensions by recursion, checked
    against SLSQP on random polytopes to 1e-11. The first draft used
    Dykstra's alternating projections and let agents brush to 0.67 of a 1.0
    separation, because the iterate stalled short of a thin cell while the
    convergence test called it done. The exact solver holds 1.02 in every
    run, and is faster.
  - The measurements are in the README and pinned by tests, failures
    included: ORCA's twelve-agent ring deadlock, buffered Voronoi's deadlock
    on any symmetric crossing, and the potential field's local minimum
    behind an obstacle.
- **Lifelong MAPF in the RL environment.** `MAPFEnv(..., lifelong=True)`
  hands an agent a new goal the moment it reaches one -- reachable, free,
  claimed by nobody, drawn from the environment's own RNG so a seed fixes the
  sequence of tasks as well as the map -- and runs to the horizon; the
  episode summary reports `throughput` and `goals_completed`. This is the
  objective deployed fleets are measured on and the top open problem in the
  survey's second edition. Rewards are computed against the goal that was
  actually reached before the re-tasking, and `ShapedReward` now caches its
  distance fields per goal cell rather than per agent, so the potential
  follows the current goal instead of rewarding a walk back to the first.
- `pymapf.rl.baselines`: planners with the policy interface, bound to an
  environment and run through the same loop as a network. `PIBTPolicy`
  re-decides every step with the PIBT paper's lifelong priority rule;
  `ReplanPolicy` re-plans with any registered solver whenever a goal changes;
  `RandomPolicy` is the floor.
- `compare_lifelong` scores policies and planner baselines by throughput on
  shared instances, per agent per hundred steps so instances of different
  size read alike. Trainers record `throughput` in their history and keep
  the best policy by it when the environment is lifelong. Measured on the
  8-agent warehouse: PIBT 8.5 goals per agent per 100 steps, replanning
  LaCAM 8.4 at six times the cost, random 0.1, all collision-free but the
  last.
- **Contributor guide and community files.** `CONTRIBUTING.md` (setup, the
  checks CI runs, and what a new algorithm needs to be merged),
  `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `SECURITY.md`, a pull
  request template, and issue templates for new algorithms and for reporting
  use in research or teaching.
- README: a *Why PyMAPF* section stating the project's two goals (a framework
  to build on, and implementations that explain), and a *Publication* section
  with the plan to archive releases on Zenodo and submit a software paper to
  JOSS.
- `Solution.violations(problem)` and `Solution.is_valid(problem)`: check a
  plan against the whole MAPF definition -- one path per agent, starting on
  its start and ending on its goal, every step a wait or a move to a free
  neighbour (diagonals only when the problem allows them), and no vertex or
  swap conflict -- and return readable messages. `is_valid()` with no
  argument keeps its old, conflicts-only meaning.
- `heuristics.default_heuristic` and `resolve_heuristic`, exported from
  `pymapf.core`; `true_distance(graph)` without a goal serves every goal,
  running one Dijkstra per distinct goal and caching it.
- `register_solver(name, replace=True)` to override a registered solver
  deliberately.
- `scripts/build_web_bundle.py --check` unpacks the bundle and imports it
  under `python -I -S` (standard library only, as in Pyodide), then solves a
  tiny instance with every registered solver. The Pages workflow runs it.
- `tests/test_optimality.py`: CBS against an independent brute-force
  joint-state Dijkstra on 80 tiny instances (4-connected and diagonal), with
  WCBS(1.5) checked against the same optimum. The WCBS bound tests used CBS
  as their reference, so a CBS bug would have hidden in both.
- `Behavior.set_timestep(dt)`: the simulator tells every behaviour the step it
  integrates with (composites forward it to their parts).
- `CoverageController.reset()` (called by `CoverageSimulator.run()`) and
  `CoverageController.cost_weights()` hooks.
- `is_infinitesimally_bearing_rigid` (Zhao and Zelazo 2016) and a per-controller
  `is_rigid()` hook on the formation controllers.

### Changed

- `CITATION.cff` now describes the software itself, so GitHub's "Cite this
  repository" button cites PyMAPF rather than the 2021 survey. The survey is
  kept as a reference.
- CI runs on pull requests, including those from forks. It used to run on
  `push` only, which never fires for a fork's branch.
- MAPPO's critic input is now the global state plus a one-hot of the agent
  being valued, so `critic_dim` is `state_size + n_agents`. The agents share a
  state but not a reward; fed the bare state, the critic could only learn the
  average of their values (three agents with rewards -0.01, -0.12 and -0.48
  all got -0.1755). This is the agent-specific global state of Yu et al.
  (2022). Checkpoints saved by an earlier MAPPO do not load into this one.
- Every search entry point (`space_time_astar`, `sipp`, `astar`,
  `weighted_astar`, `focal_astar`) and solver (CBS, WCBS, prioritized, LNS)
  now defaults to `heuristic=None`, resolved to the admissible heuristic for
  the map and move set: Manhattan on a 4/6-connected lattice, Chebyshev with
  diagonals, and cached exact distances on a general graph. Passing a name or
  callable works as before. LNS gained a `heuristic` argument for its SIPP
  repairs.
- `MAPFProblem` rejects two agents sharing a start or a goal with a
  `ValueError` naming both; such an instance is unsolvable from the outset.
  `Agent` stores list cells (as JSON hands them over) as tuples.
- `benchmark.run_once` counts a run as solved (and valid) only if the plan
  passes `Solution.violations`; an invalid plan is reported with
  `solved=False` and its first violation in `note`.
- `benchmark._supported` still drops an option the chosen solver lacks (a
  benchmark is configured once for every algorithm), but raises for an option
  *no* registered solver accepts, so `time_limt=5` no longer runs silently
  with the default. It follows `**kwargs` up the MRO, so a subclass that
  forwards to its base is checked against the base's parameters.
- `register_solver` accepts re-registering the *same* class (same module and
  qualified name), which is what re-running a notebook cell does; a different
  class under a taken name still raises, now naming the existing one.
- The bundle builder derives its module list from the import graph of
  `pymapf` and `pymapf.benchmark` (function-level imports and parent packages
  included) and refuses any reachable non-stdlib import, instead of a
  hand-kept list. `.docs/pymapf-bundle.json` regenerated: 0.7.0 -> 0.9.0,
  19 -> 26 modules.
- Documentation: WeightedCBS is BCBS(w, 1) in Barer et al. (2014) -- focal
  high level bounded by the cheapest OPEN node, optimal low level -- not the
  ECBS high level (module docstring, REFERENCES.md, README, JS comment). The
  LNS docstring now says a repair is accepted only on a strict improvement,
  as the code does. `find_first_conflict` documents that with diagonals two
  agents crossing through one 2x2 block are not a conflict in this model.
- `CompositeBehavior` now requires its parts to agree on `output` (all
  accelerations or all velocities) and raises `ValueError` otherwise; it takes
  that common output, and adopts its first part's `SwarmParams` when none are
  passed.
- `SwarmSimulator(instance)` without `params` keeps the instance's parameters
  instead of replacing them with defaults; explicit `params` still win.
- `register_shape` raises on a duplicate name, like `register_behavior`.
- The legacy `MultiAgentVelocityObstacle`, `MultiAgentNMPC` and
  `CooperativeAStar` no longer create `./logs`, call `logging.basicConfig` or
  install `coloredlogs` on the root logger; each module logs through
  `logging.getLogger(__name__)` and `log_level` only sets the level of that
  planner's own package logger. `CooperativeAStar.register_agent` ignores (with
  a warning) an agent that shares a start or goal with one already registered.
- `NMPCAgent` takes a `seed=0` keyword for its warm-start generator.

### Fixed

- The playground footer and the promo film said the project was Apache 2.0
  licensed. It is MIT, as `LICENSE` and `pyproject.toml` say.
- **The goal bonus could be farmed.** It was paid on every arrival and never
  taken back, and an agent on its goal pays no step cost, so stepping off and
  back on was worth +8.6 over twenty steps where parking was worth 0, under
  both `sparse` and `shaped`. The optimal policy never parked. The bonus is
  now repaid on leaving, which makes it a potential like the shaping term.
- **Truncated episodes were treated as finished.** The trainer cut GAE at a
  truncation exactly as at a real terminal state, valuing the future at zero.
  The last reward of a truncated stream now carries `gamma * V(final state)`.
  `VectorMAPFEnv` already kept the final observation for this and nothing read
  it; it now also keeps `final_state`, so MAPPO bootstraps too. Lifelong
  episodes only ever end by truncation, so every one of them was affected.
- Lifelong mode counted an agent with nowhere new to go (every free cell
  taken) as completing its goal again on every step: 102 completions for 4
  arrivals in a full 2x3 room. Only arrivals count now.
- `action_space(agent)` and `observation_space(agent)` return the same object
  on every call, as PettingZoo expects, so seeding a space has an effect.
- A randomised `reset` draws another scenario seed when one cannot place the
  team (8 in 3,000 for six agents on a dense 8x8) instead of raising partway
  through a training run.
- `evaluate` caught any `TypeError` from a policy's `act` and retried without
  `deterministic`, hiding real errors and silently changing the mode. It now
  checks the signature. The `pymapf.rl` docstring's example passed
  `trainer.policy`, which crashes; it passes the trainer.
- Visualisation: `plot_separation` on one vehicle, `plot_spacetime` on an
  empty plan, `animate_trajectories` leaving its figure open, and `save()`
  overwriting a user's `animation.ffmpeg_path` unconditionally.
- **CBS, WCBS, prioritized planning, LNS and A\* were not optimal with
  `allow_diagonals=True`.** The default heuristic was Manhattan distance,
  which overestimates once a diagonal (one timestep) covers two Manhattan
  units: on a 4x5 map the optimum is 5 and CBS returned 6, and space-time A\*
  was suboptimal on 4 of 378 random diagonal instances. The default is now
  Chebyshev with diagonals (0 of 378). The JS port resolves its default the
  same way; on 300 random instances (70% with diagonals) the two engines
  agree on every CBS, WCBS and prioritized cost, where the old JS disagreed
  with the fixed Python on 8 of 150.
- Solvers failed on an `ExplicitGraph` with default arguments: Manhattan
  distance on string nodes raised `TypeError`. The default there is now a
  per-goal cached true distance. `distance_table` searched forward from the
  goal, which overestimates on one-way edges (on the cycle a -> b -> c -> a it
  put b two moves from c instead of one); it now follows reversed edges on a
  directed graph. The `ExplicitGraph` docstring no longer refers to a
  `zero_heuristic` that does not exist.
- The playground bundle could not `import pymapf`: `pymapf/__init__.py`
  imports `kinodynamic` and `aerial` and `core/__init__.py` imports `voxel`
  (and `aerial` needs `core/assignment`), none of which the builder's list
  carried (`ModuleNotFoundError: pymapf.core.voxel`). The CI "check" only
  asserted every file name ended in `.py`.
- `sipp` ignored back-to-back edge constraints: it stepped past the first
  blocked crossing time once and never re-checked, so with a move forbidden
  at t = 1 and t = 2 on a 1x2 grid it crossed at t = 2. It now waits while
  the edge is blocked and both safe intervals allow it.
- `focal_astar` broke its `w` bound: it never reopened an expanded node
  reached by a cheaper path (FOCAL expands out of f order, so an expanded
  node's g need not be optimal) and never admitted to FOCAL the nodes a
  rising bound newly covered. On a 12x12 map with `w = 1.5` it returned 21
  steps where the optimum is 13. It is now the textbook focal search: 0
  violations over 2513 random instances x 3 weights.
- `space_time_astar` ignored a vertex constraint on the start at t = 0, while
  `sipp` returned `None`; both now return `None` (and so does the JS port).
- `ExplicitGraph.free_cells` subtracted blocked names that are not nodes, so
  it could go negative; the solvers use it as their time horizon, and CBS
  then found no path on a two-node graph.
- `PrioritizedPlanning` compared `priority` with the agents as sets, so a
  repeated name passed and that agent was planned twice; it now requires each
  name exactly once.
- `from_ascii` silently kept only the last of two equal start (or goal)
  letters; it now raises naming both cells.
- `bottleneck(n_agents=0)` built one agent and `n_agents` above the rooms'
  capacity (45 for `room=5`) raised `IndexError`; both now raise a
  `ValueError` stating the range.
- `Solution.sum_of_costs`' docstring claimed "time before settling at the
  goal" while the code sums `len(path) - 1`; it now says what it computes and
  why that equals the standard sum of costs for every built-in solver (LaCAM
  and PIBT trim trailing goal waits). Reported costs are unchanged.
- **Legacy Velocity Obstacle avoided nothing**: `MultiAgentVelocityObstacle`
  unpacked `simulate_step`'s `(history, state, velocity)` in the wrong order,
  so other agents appeared as `[vx, vy, x, y, vx, vy]`, and the list of other
  agents was rebuilt from empty each step, so the first agent never saw
  anybody. A head-on swap passed at 0.06 m between centres against a radius sum
  of 1.0; every agent now decides from one snapshot of everyone's previous
  state and the pair keeps 1.11 m apart. `MultiAgentNMPC` had the same
  "first agent is blind" defect and is fixed the same way.
- **Legacy NMPC obstacle prediction**: `NMPCAgent` evaluated
  `obstacles[:, step, :]`, discarded it and looped over the time axis, so two
  obstacles produced a silently meaningless cost and three raised a broadcast
  error. It now rolls each obstacle's state at the current step over the
  horizon. `__hash__` returned a `str` (agents could not go in a set) and the
  SLSQP warm start drew from the global unseeded `np.random`; runs are now
  reproducible.
- **Legacy cooperative A\* returned conflicting plans**: no swap (edge)
  check, the diagonal and non-diagonal conflict branches were identical, an
  agent accepted a goal that an earlier-planned agent drove through
  afterwards, and `World.get_start_goal` only re-drew when start *and* goal
  had both been used. Over 150 seeded 10x10 six-agent instances the plans held
  61 swapping pairs and 53 vertex conflicts (29 shared starts). The planner now
  checks vertex, swap and diagonal-crossing conflicts, only stops at a goal
  nobody passes through later, keys every search node by `(x, y, t)` (the
  start was keyed by state and the rest by cell, so a later arrival could
  reopen a closed node), and starts and goals are unique: no conflicts on the
  same instances with or without diagonals, and 122 of 1800 agents left
  without a path instead of 339.
- `velocity_obstacle.py`, `nmpc.py` and `centralized/animator.py` called
  `matplotlib.use("TkAgg")` at import whenever `DISPLAY` was set, overriding
  `MPLBACKEND=Agg`, failing without tkinter and switching the backend under
  `pymapf.viz` too. Removed.
- `SwarmSimulator(Boids(params=SwarmParams(migration_point=(200, 0))))`
  dropped the waypoint (centroid x -1.5 after 200 steps instead of 144), and
  params set on a `CompositeBehavior` never reached its parts.
- `CompositeBehavior` declared `output="acceleration"` whatever its parts
  returned and never called their per-step `commands()` hook: ActiveElastic
  flew at order 0.98 / speed 2.08 alone and 0.41 / 5.02 when wrapped, and a
  `DisplacementFormation` part never reassigned slots. A one-part composite
  now flies exactly like its part.
- `BufferedVoronoi`, `ORCA` and `ActiveElastic` kept their own `dt=0.1`
  whatever the simulator integrated with; the 12-agent BVC circle swap at
  `dt=0.25` had 52 colliding steps (0 now).
- `SwarmSimulator.run()` re-seeds the noise generator, so two runs of one
  simulator are identical (they differed by up to 1.02 m).
- `TimeVaryingCoverage` differenced against its previous *extrapolated*
  target, feeding its own feed-forward back in (`e_t = -lookahead e_(t-1)`):
  on a static density targets kept flipping 1.16 either side of the
  centroids and the team never stopped moving (0.32 per step after 100
  iterations). It now differences the raw centroids.
- `AdaptiveCoverage.cost` scored against its own moving estimate, so a run
  that removed 82% of the true locational cost reported an improvement of
  -190%; it now scores against the true density when it has one. Controller
  memory no longer leaks between `CoverageSimulator.run()` calls (a second
  adaptive run started at cost 3264 instead of 2054).
- `pymapf.decentralized.flocking.simulate` looked a callable up by
  `__name__`: a custom controller function raised, and one called `boids`
  silently ran the library's Boids. Custom functions now run as given.
- `BearingFormation` warned that a planar circle in 3D "will not converge"
  (then held it with every radius at 1.70 and zero z spread): it was judged by
  distance rigidity. It now uses bearing rigidity, for the warning and when
  augmenting a sparse interaction graph.
- Formation controllers dropped `spacing` unless the shape name was
  lower-case (`shape="V"`); `AnnulusDomain.project` left a point at the exact
  centre inside the hole; the `pymapf.swarm` docstring example looped over
  every behaviour, two of which raise `TypeError` without a target density.
- Docstrings: `GaussianKernelNeighborhood` promised a command continuous in
  the positions (the default cutoff is 2 sigma, where the kernel is still
  0.14, and mean-normalisation rescales every weight when the neighbour set
  changes); `MinimalisticFlocking` called its repulsion unbounded (it is
  clipped at `4 * max_acceleration`); `CuckerSmale`'s formula omitted the
  `1/|N_i|` average.
- Tests: `assert a.all() == b.all()` (always true) in the velocity-agent and
  NMPC-agent tests is now a real array comparison; the cooperative A\* tests
  are seeded, and the legacy visualisation tests write their GIFs to
  `tmp_path` instead of the working directory.

## [0.9.0]

NumPy 2 support, and the end of the Python versions that were holding it back.

### Fixed

- **pymapf works on NumPy 2.** `velocity_obstacle` built a 2-element direction
  vector and passed it to `np.cross`, which NumPy 2.0 removed support for; the
  vector is now padded with an explicit `z = 0.0`, spelling out the behaviour
  1.x supplied silently. Anyone on NumPy 2 was hitting this on every velocity
  obstacle step.

### Changed

- **Python 3.10 is the new floor**, and the matrix runs 3.10 through 3.13.
  3.8 and 3.9 are past end of life, and the pinned scientific stack no longer
  ships wheels for them. `requires-python` and the classifiers moved with it.
- Pinned dependencies are now declared per interpreter, because NumPy and SciPy
  drop interpreters faster than this project does: each floor gets the last
  release that still supports it, rather than one pin that forces the whole
  matrix down to the oldest.
- The logo is served from the `branding` repository rather than a third-party
  SVG host, and all URLs follow the organisation's rename to `openplan-labs`.

### Removed

- `pip-tests.yml`, a workflow that targeted a `master` branch this repository
  does not have and pinned interpreters it no longer supports. It had not run
  successfully in years.
- Mergify. Its auto-merge rule required an approving review that a
  single-maintainer project does not produce, so every recent pull request was
  merged by hand while the check reported neutral.

## [0.8.0]

The learning layer reaches the published site, and the artifact it reads becomes
readable.

**The importable package is byte-identical to 0.7.0** -- `git diff v0.7.0..v0.8.0
-- pymapf/` is empty. Nothing here requires an upgrade for code reasons. What
changed is the documentation site, the published benchmark data, the release
pipeline and a README table that was wrong.

### Fixed

- **`.docs/assets/rl-benchmark.json` was not valid JSON.** A method that solves
  nothing has no mean cost, and `json.dump` writes a bare `NaN` for it -- 27
  times in that file. Python's own loader accepts `NaN`, so a round-trip in
  Python looks clean; it is not JSON, `JSON.parse` rejects the *entire*
  document, and the file had therefore been unreadable by the page meant to
  display it since the day it was written. `scripts/train_rl.py` now maps
  non-finite floats to `null` and passes `allow_nan=False` so it cannot regress
  silently.
- **The README's RL table had drifted from the run that produced it**, claiming
  47% / 1.10x / 3.05x where the artifact said 45% / 1.11x / 2.94x. Corrected,
  and the table now says where its numbers come from. The new site section
  renders from the artifact rather than restating it, because transcription is
  what caused the drift.
- **A `.gitignore` rule of `*.gif` was swallowing the two gallery animations.**
  Both are published artifacts -- the README embeds one, the playground embeds
  both, and Pages uploads the checked-out `.docs/` tree -- so an ignored GIF was
  a broken image on the live site rather than a missing local file.
- **A failed PyPI upload used to cancel the GitHub Release.** `github-release`
  needed `publish-pypi`, so an unregistered trusted publisher would have taken
  the release down with it, and a tag cannot cleanly be re-pushed to try again.
  It is gated on `verify` now and runs either way, with notes that state whether
  the upload succeeded instead of printing a `pip install` line that would not
  work.

### Added

- **A Learning section on the playground site.** The page previously contained
  no occurrence of `rl`, `ippo`, `mappo`, `reinforcement` or `learning`, while
  shipping the RL film, its poster and the benchmark JSON with nothing linking
  any of them. The section carries the film, a table across all four benchmark
  settings, and the findings. `survey-v2.md` and `research-notes.md` were
  orphaned the same way and are now linked from the nav and the README.
- `tests/test_docs_assets.py` -- every `.json` under `.docs/` must parse with
  `NaN`/`Infinity` **refused** (via `parse_constant`; `strict=False` is the
  opposite of what is wanted), the benchmark must contain both a learner and
  CBS, and every asset `index.html` references must exist or it would 404 on
  Pages.

### Changed

- `actions/checkout` is on v7 across all five workflows, `actions/setup-python`
  on v7, `codecov/codecov-action` on v7, `stefanzweifel/git-auto-commit-action`
  on v7. `pip-tests.yml` had pinned **`actions/checkout@master`** -- a floating
  ref, resolved at run time, from a third-party repository, running with the
  workflow's token. Renovate does not rewrite floating refs, so it would have
  survived the bot's own upgrade PR.
- The pinned CI `numpy` moves to 1.22.0, clearing
  [CVE-2021-34141](https://nvd.nist.gov/vuln/detail/CVE-2021-34141). That is the
  last numpy supporting Python 3.8, which is the floor of the test matrix;
  scipy 1.18, matplotlib 3.11 and termcolor 3 were all declined for requiring
  3.12, 3.11 and 3.9 respectively.

## [0.7.0]

Multi-agent reinforcement learning on the library's own MAPF instances,
benchmarked against its own optimal planner. First release published from CI.

### Added

- **`pymapf.rl`** -- the MAPF instances as a multi-agent RL environment, built
  so the learned side and the planning side are directly comparable rather than
  merely adjacent.
  - `MAPFEnv` follows the **PettingZoo Parallel API** without importing
    PettingZoo, so it runs in a bare CI job; `to_pettingzoo()` wraps it in the
    real `ParallelEnv` when that is installed, and `SingleAgentGym` gives a
    Gymnasium-style view for single-agent algorithms.
  - Transition semantics are MAPF's, not a gridworld's approximation: vertex
    conflicts, **edge (swap) conflicts**, and refusals that **cascade** to a
    fixed point. A rollout is therefore a valid plan by construction -- validity
    is 100% in the benchmark even for a random policy, because it cannot be
    otherwise.
  - `MAPFEnv.solution()` returns a `pymapf.Solution`, so a learned policy is
    scored by the same `sum_of_costs` and the same `is_valid()` as CBS.
  - `IPPO` and `MAPPO`, shared-parameter PPO with GAE. They differ by one class
    attribute -- whether the critic sees the global state -- because that is the
    entire difference between the two algorithms.
  - **No dependency beyond numpy.** The default backend is an MLP with
    hand-written backpropagation and Adam, gradient-checked against finite
    differences, so the learning code is tested in CI rather than only shipped.
    A torch backend implements the same objective for scale.
  - Registries throughout: `register_observation`, `register_reward`,
    `register_algorithm`, matching the solvers and the swarm layer.
  - `ShapedReward` uses the library's exact backward-Dijkstra distance oracle as
    a potential, so the shaping is policy-invariant (Ng et al. 1999) *and* routes
    around walls -- a stronger potential than a learned-MAPF paper normally has.
  - `compare()` scores policies and planners on identical seeded instances and
    reports true suboptimality against the optimum.
- `scripts/train_rl.py` reproduces every number in `.docs/survey.md` § 7.6.
- `scripts/make_rl_promo.py` renders a 62-second film for the learning layer.
  The policy is trained while the film renders, the split-screen scene is that
  policy acting on one shared instance in both action modes, and the 70/30
  failure split is measured over 80 instances during the render -- so the film
  cannot drift from the library's behaviour.
- New extras: `rl` (numpy only), `rl-torch`, `rl-ecosystem`.
- **`.docs/survey-v2.md`** -- a second edition of the survey. It revises the
  framing rather than the measurements: lifelong MAPF has replaced one-shot MAPF
  as the objective the field optimises (throughput, not sum-of-costs);
  guidance-graph optimisation is a third lever that improves the *environment*
  rather than the solver, and is orthogonal to everything already implemented
  here; and learning has settled into supplying heuristics, priorities and
  neighbourhoods *inside* a sound search rather than replacing it -- which our
  own RL numbers support from the negative side.
- **`.docs/research-notes.md`** -- the annotated literature scan behind it,
  recording for each citation how far it was actually read. Nothing in it is
  reproduced here unless marked [impl].
- **`.github/workflows/release.yml`** -- pushing a `v*` tag now tests on 3.8-3.10,
  builds, installs the built wheel into a clean interpreter on 3.8 and 3.12 and
  solves a real instance with it, publishes to PyPI through Trusted Publishing
  (OIDC -- no API token in this repository), and opens a GitHub Release with the
  artifacts attached and this changelog's section as the notes. The build fails
  if the tag disagrees with `pymapf.__version__`, because a version number on
  PyPI can never be reused. `workflow_dispatch` offers TestPyPI as a dry run.

### Changed

- **Packaging moved from `setup.py` to `pyproject.toml`** (PEP 621). The version
  had lived in two places and had already drifted apart once -- `setup.py` said
  0.3.0 while the package reported 0.5.0 -- so it is now read from
  `pymapf.__version__` alone. `pip install pymapf` remains dependency-free.
- `docs/` is now **`.docs/`**. Every reference moved with it, including the Pages
  workflow's path filter, bundle check and upload path. Publishing is unaffected
  because this repository deploys through `actions/upload-pages-artifact` rather
  than the "deploy from a branch, /docs folder" setting, which offers only `/`
  or `/docs` and could not have followed the move.

### Findings

- **How a policy is sampled matters more than which algorithm trained it.** The
  same weights score 45% solved at 1.11x optimal under argmax, and 100% at 2.94x
  when sampled. Over 80 instances the 33 greedy failures split into two modes:
  70% are collision-free period-2 orbits where the agents never touch, and 30%
  are period-1 freezes colliding every step -- a genuine livelock. None involves
  a wall, and in every case both agents solve the instance alone. `compare()`
  reports both action modes by default because either alone is half the result.
- **IPPO and MAPPO are indistinguishable here** (45% vs 44%, 1.11x vs 1.11x),
  matching the MAPPO paper's own conclusion.
- **On the hardest setting the optimal planner is the one that fails**: CBS
  closed 1% of bottleneck instances in five seconds; PIBT closed 100% at 1.07x.
- Two non-fixes, recorded so they are not retried: KL early stopping never fires
  (measured KL 0.0004-0.014 against a 0.02 threshold, runs bit-identical), and
  the entropy coefficient moves the final solve rate by one point across
  0.01/0.03/0.05.

### Fixed

- Orthogonal initialisation returned a **non-contiguous** array, so
  `param.reshape(-1)` handed back a copy and anything writing through that view
  updated nothing -- which is how a finite-difference gradient check reports a
  zero gradient for a weight that is fine. Found by writing the gradient check.
- The same routine mishandled non-square shapes outright: `np.linalg.qr` returns
  the reduced factorisation, so a wide layer got a square weight matrix and
  failed to broadcast on the first forward pass.

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
- `.docs/survey.md` gains section 6.2c on formation control, with measured
  convergence times for all four controllers and the three findings below.
- **Two new promo scenes** covering the swarm side: the minimalistic flocking
  model, and formation control with its taxonomy table. Both draw live
  simulation output, so the film cannot drift from the library's behaviour.

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
- **`.docs/survey.md`** -- an extension of *Survey of the Multi-Agent Pathfinding
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

- **Interactive playground** (`.docs/`, published to GitHub Pages): edit a map,
  pick a solver and watch the search resolve conflicts in the browser. It runs
  the library's own source under Pyodide in a web worker, with a JavaScript port
  of the solvers (`.docs/mapf.js`) as an instant-response fallback. The Python
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
