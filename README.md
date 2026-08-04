<div align="center">
    
<img src="https://svgshare.com/i/TFJ.svg" alt="Logo" width="300">     
    
</div>

<div align="center">

# PyMAPF

✨ A Python toolbox for Multi-Agents Planning (Centralized and Decentralized) ✨

</div>

<div align="center">
    
![tests](https://github.com/APLA-Toolbox/pymapf/workflows/tests/badge.svg?branch=main)
![pip-package](https://github.com/APLA-Toolbox/pymapf/workflows/.github/workflows/pip-tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/APLA-Toolbox/pymapf/branch/main/graph/badge.svg?token=63GHA9JUND)](https://codecov.io/gh/APLA-Toolbox/pymapf)
[![CodeFactor](https://www.codefactor.io/repository/github/apla-toolbox/pymapf/badge)](https://www.codefactor.io/repository/github/apla-toolbox/pymapf)
[![Percentage of issues still open](http://isitmaintained.com/badge/open/APLA-Toolbox/pymapf.svg)](http://isitmaintained.com/project/APLA-Toolbox/pymapf "Percentage of issues still open")
![PipPerMonths](https://img.shields.io/pypi/dm/pymapf.svg)
[![Pip version fury.io](https://badge.fury.io/py/pymapf.svg)](https://pypi.python.org/pypi/pymapf/)
[![GitHub license](https://img.shields.io/github/license/Apla-Toolbox/pymapf.svg)](https://github.com/Apla-Toolbox/pymapf/blob/master/LICENSE)
[![GitHub contributors](https://img.shields.io/github/contributors/Apla-Toolbox/pymapf.svg)](https://GitHub.com/Apla-Toolbox/pymapf/graphs/contributors/)

</div>

<div align="center">
    
[Report Bug](https://github.com/APLA-Toolbox/pymapf/issues) · [Request Feature](https://github.com/APLA-Toolbox/pymapf/issues)

Loved the project? Please consider [donating](https://www.buymeacoffee.com/dq01aOE) to help it improve!

</div>

## Features 🌱

- 🎮 **[Interactive playground](https://apla-toolbox.github.io/pymapf/)** — run the solvers in your browser, watch the search resolve conflicts live
- 🧭 Centralized planners: **CBS**, **Weighted CBS**, Prioritized Planning, **PIBT**, **LaCAM**, **MAPF-LNS** — every one referenced in [REFERENCES.md](REFERENCES.md)
- 🕸️ Works on **arbitrary graphs**, not just grids (roadmaps, warehouse topologies, PRMs)
- 🐦 **Decentralized swarm control**, all object-oriented and registry-based: 10 flocking models (boids, Vicsek, Cucker–Smale, Olfati-Saber, proximal, active-elastic, acceleration-based, Gaussian-kernel, minimalistic, distributed-3D), 5 coverage controllers over 6 pluggable domains, and Gaussian-mixture distribution control
- 📐 **Formation control** on the displacement / distance / bearing taxonomy, with exact Hungarian slot assignment and a rigidity test that tells you when a target shape is holdable at all
- 🧠 **Multi-agent RL** (`pymapf.rl`): the MAPF instances as a PettingZoo-parallel environment, IPPO and MAPPO that train with **no dependency beyond numpy**, and a benchmark scored against the *optimal* CBS solution rather than another heuristic
- 🔬 **[Extended survey](.docs/survey.md)** of MAPF 2021→2026 plus an experimental section with measured (and negative) results — and a [second edition](.docs/survey-v2.md) that revises the framing around lifelong MAPF, guidance-graph optimisation and learning-inside-search, with the [literature scan](.docs/research-notes.md) behind it
- 🧩 Pluggable solver framework with a name-based registry, pluggable heuristics and deterministic maps
- 🔭 **Observable search**: every solver streams `SearchEvent`s — record them, animate them, or watch them live
- 🗺️ **Six reproducible scenario families** (empty room, random obstacles, warehouse, maze, bottleneck, corner swap) plus ASCII maps
- 📊 **Benchmark harness** with CSV/JSON export and ready-made charts
- 🎬 **Visualisation**: static plots, congestion heatmaps, space-time cubes, timelines, GIF/MP4 animations, live views (window *or* terminal)
- 🔎 Reactive distributed planners (Nonlinear Model Predictive Control, Velocity Obstacles)
- 🪶 Zero runtime dependencies in the core — the solvers are pure standard library

<div align="center">

<img src=".docs/assets/animated-search.gif" alt="Conflict-based search resolving conflicts" width="640">

<em>CBS finding and resolving conflicts, node by node — produced by <code>pymapf.viz.animate_search</code></em>

</div>

## Install 🖇️

```bash
pip install pymapf                 # the solver framework — no dependencies
pip install "pymapf[viz]"          # + plots, animations and live views
pip install "pymapf[all]"          # + the decentralized/legacy planners
```

From a clone:

```bash
git clone https://github.com/apla-toolbox/pymapf && cd pymapf
pip install -e ".[all,dev]"
pytest
```

## Usage 📑

### Quickstart 🧭

Cells are `(row, col)`; a truthy grid value marks an obstacle.

```python
import pymapf

grid = pymapf.GridMap([
    [0, 0, 0],
    [0, 1, 0],   # a wall in the middle
    [0, 0, 0],
])
problem = pymapf.MAPFProblem(grid, [
    pymapf.Agent("a", start=(0, 0), goal=(2, 2)),
    pymapf.Agent("b", start=(2, 0), goal=(0, 2)),
])

print(pymapf.available_solvers())          # ['cbs', 'prioritized', 'wcbs']

solution = pymapf.solve(problem, "cbs")
print(solution.sum_of_costs, solution.makespan, solution.is_valid())
for name, path in solution.paths.items():
    print(name, path)                      # path[t] is the cell at timestep t
```

### Choosing a solver 🧠

| Solver | Name | Guarantee | Use it when |
|---|---|---|---|
| Conflict-Based Search | `"cbs"` | optimal sum-of-costs | you need the best plan and can pay for it |
| Weighted CBS (ECBS) | `"wcbs"` | cost ≤ `w` × optimal | you need most of the quality, much faster |
| **LaCAM** | `"lacam"` | complete | large fleets, milliseconds, quality refined later |
| **PIBT** | `"pibt"` | none (incomplete) | thousands of agents, one timestep at a time |
| Prioritized Planning | `"prioritized"` | none (incomplete) | the classic baseline |
| **MAPF-LNS** | `"lns"` | anytime, never worse than its initial plan | you have a deadline and want the best plan by then |

Measured on the 8-agent warehouse instance: CBS spends 6 470 expansions and
5.3 s for cost 100; weighted CBS reaches 104 in 23 expansions and 18 ms; LaCAM
returns a valid plan in 4 ms; LNS takes PIBT's 175 down to 113 in two seconds.
The full picture, including where each one fails, is in
[`.docs/survey.md`](.docs/survey.md).

CBS is exponential in the number of conflicts, so give it a budget on hard maps:

```python
solution = pymapf.solve(problem, "cbs", time_limit=5.0)      # None if it runs out
bounded  = pymapf.solve(problem, "wcbs", weight=1.5)         # within 50% of optimal
```

### Any graph, not just grids 🕸️

```python
from pymapf import ExplicitGraph

graph = ExplicitGraph.undirected(
    [("dock", "aisle1"), ("aisle1", "aisle2"), ("aisle2", "pack"), ("dock", "pack")]
)
problem = pymapf.MAPFProblem(graph, [
    pymapf.Agent("r1", "dock", "pack"),
    pymapf.Agent("r2", "pack", "dock"),
])
pymapf.solve(problem, "lacam")      # PIBT and LaCAM use exact graph distances
```

### Scenarios 🗺️

Reproducible instances, deterministic in their seed:

```python
scenario = pymapf.build_scenario("warehouse", n_agents=8, seed=3)
solution = pymapf.solve(scenario.to_problem(), "wcbs")

print(pymapf.available_scenarios())
# ['bottleneck', 'corner_swap', 'empty_room', 'maze', 'random_obstacles', 'warehouse']
```

Or write the map by hand — lowercase is a start, uppercase the matching goal:

```python
from pymapf.scenarios import from_ascii

scenario = from_ascii("""
##########
#a......A#
#.######.#
#B......b#
##########
""")
```

### Watching the search 🔭

Solvers push `SearchEvent`s to any callable you pass as `observer`:

```python
trace = pymapf.SearchTrace()
solution = pymapf.solve(scenario.to_problem(), "cbs", observer=trace)

print(trace.summary())
# {'events': 67, 'expansions': 16, 'conflicts': 15, 'solved': True, 'cost': 48, ...}
```

Live, while it runs — in a window, or in the terminal over SSH:

```python
from pymapf.viz import LiveSolveView, LiveConsoleView

with LiveConsoleView(scenario) as view:                    # no display needed
    pymapf.solve(scenario.to_problem(), "cbs", observer=view)
```

### Visualisation 🎬

```python
from pymapf import viz

viz.save(viz.plot_solution(solution, scenario), "plan.png")
viz.save(viz.plot_congestion(solution, scenario), "congestion.png")   # traffic hot spots
viz.save(viz.plot_spacetime(solution, scenario), "spacetime.png")     # the 3D search cube
viz.save(viz.plot_timeline(solution), "timeline.png")                 # who waits, when

viz.save_animation(viz.animate_solution(solution, scenario), "plan.gif", fps=16)
viz.save_animation(viz.animate_search(trace, scenario), "search.mp4")
```

| | | |
|---|---|---|
| ![solution](.docs/assets/solution.png) | ![congestion](.docs/assets/congestion.png) | ![space-time](.docs/assets/spacetime.png) |
| `plot_solution` | `plot_congestion` | `plot_spacetime` |

### Benchmarking 📊

```python
from pymapf.benchmark import compare_algorithms, scaling_study
from pymapf import viz

report = compare_algorithms(["warehouse", "maze"], ["cbs", "wcbs", "prioritized"], time_limit=2.0)
print(report.table())
report.to_csv("results.csv")

scaling = scaling_study("random_obstacles", agent_counts=(2, 4, 6, 8, 10, 12))
viz.dashboard(scaling, report).savefig("dashboard.png")
```

![dashboard](.docs/assets/dashboard.png)

### Adding your own solver 🔌

```python
from pymapf.core import MAPFSolver, Solution, register_solver
from pymapf.algorithms import space_time_astar


@register_solver("selfish")
class Selfish(MAPFSolver):
    """Every agent takes its own shortest path and ignores the others."""

    def solve(self, problem, observer=None):
        paths = {
            agent.name: space_time_astar(problem.grid, agent.start, agent.goal)
            for agent in problem.agents
        }
        return Solution(paths=paths, algorithm=self.name)


pymapf.solve(problem, "selfish").first_conflict()   # spoiler: there is one
```

### Swarms: flocking, coverage and distribution 🐦

Same conventions as the planners — an abstract base class per family, a name
registry, swappable strategy objects.

```python
from pymapf.swarm import SwarmSimulator, available_behaviors

for name in available_behaviors():          # 10 flocking + 4 formation + 2 distribution
    result = SwarmSimulator(name, n_agents=20).run(steps=300)
    print(name, result.metrics.summary())
```

**Who each agent sees** is a strategy object, and it changes the collective
behaviour as much as the control law does:

```python
from pymapf.swarm import SwarmSimulator, TopologicalNeighborhood

SwarmSimulator("acceleration", neighborhood=TopologicalNeighborhood(k=5))
# better spacing (2.17 m vs 1.60 m) with a third of the connectivity
```

**Composition** rather than new classes:

```python
from pymapf.swarm import CompositeBehavior, CuckerSmale, AccelerationFlocking

blend = CompositeBehavior([(CuckerSmale(), 0.5), (AccelerationFlocking(), 1.0)])
```

**Coverage** is written against a domain, so one controller serves every shape:

```python
from pymapf.swarm import CoverageSimulator

for domain in ("planar", "disk", "sphere", "hemisphere", "annulus"):
    print(domain, CoverageSimulator("lloyd", domain=domain, n_agents=9).run(steps=40).improvement)
```

Controllers: `lloyd`, `limited_range`, `adaptive` (learns the density online),
`gmm` (splits the team across mixture components), `time_varying` (pursues
moving targets).

**Gaussian mixtures** are the importance model *and* the target distribution:

```python
from pymapf.swarm import GaussianMixtureDensity, SwarmSimulator

target = GaussianMixtureDensity(means=[(-8, 0), (8, 4), (0, -9)],
                                covariances=[3., 3., 2.], weights=[.4, .4, .2])
sim = SwarmSimulator("mixture_assignment", n_agents=30, mixture=target)
sim.run(steps=400)          # allocation lands on 12 / 12 / 6 — exactly the quota

fitted = GaussianMixtureDensity.fit(observations, k=2)   # EM from measurements
```

**Formation control** is organised by *what each agent can measure* — the
displacement / distance / bearing taxonomy — because that is what decides which
symmetry you can fix:

```python
from pymapf.swarm import SwarmSimulator, is_infinitesimally_rigid, get_shape

for law in ["displacement_formation", "distance_formation",
            "bearing_formation", "leader_follower"]:
    sim = SwarmSimulator(law, n_agents=9, shape="v", spacing=3.0)
    result = sim.run(steps=800)
    print(law, sim.behavior.error(result.final))     # graded under the
                                                     # symmetries it can't see
```

| Law | Agent measures | Formation fixed up to | Converges in |
|---|---|---|---|
| `displacement_formation` | relative position, shared frame | translation | 3.6 s |
| `distance_formation` | range only | translation, rotation, reflection | 15.6 s |
| `bearing_formation` | direction only (cameras) | translation, **scale** | 41.1 s |
| `leader_follower` | offset from a leader | translation | 4.7 s |

The less each agent senses, the longer it takes — that is the taxonomy restated
as a cost. Distance and bearing control also need the constraint graph to be
**rigid**, and the library says so before you fly it:

```python
line = get_shape("line", spacing=3.0).centred(6, 2)
is_infinitesimally_rigid(line, [(i, j) for i in range(6) for j in range(i + 1, 6)])
# False — a collinear target has flex modes no first-order controller can see
```

The functional API in `pymapf.decentralized.flocking` / `.coverage` still works;
it now delegates to this layer.

### Reinforcement learning 🧠

The same instances the planners solve, as a multi-agent environment — and the
reason to have it here rather than in a separate repo is that a rollout comes
back as a `pymapf.Solution`, so a learned policy and CBS are scored by
*identical* code:

```python
from pymapf.rl import MAPFEnv, make_trainer, compare

env = MAPFEnv("random_obstacles", n_agents=4, height=10, width=10)
trainer = make_trainer("mappo", env)      # or "ippo"
trainer.learn(total_steps=400_000)        # ~7k steps/s, numpy only

for row in compare(env, {"mappo": trainer}, episodes=100):
    print(row["method"], row["success_rate"], row["suboptimality"])
```

It follows the **PettingZoo Parallel API** without importing PettingZoo, so it
runs in a bare environment and still drops into any MARL library
(`env.to_pettingzoo()` when you want the real base class). Observations,
rewards and algorithms are registries like everything else:

```python
from pymapf.rl import register_observation, LocalWindow

@register_observation("my_encoding")
class MyEncoder(LocalWindow):
    ...
```

Three things it gets from living inside the library:

- **exact reward shaping.** `ShapedReward` uses the backward-Dijkstra distance
  oracle the solvers already use, so the potential is the true remaining cost
  rather than a Manhattan guess — and being potential-based, it is
  policy-invariant (Ng et al. 1999).
- **conflict-freedom by construction.** Vertex, edge and cascading conflicts are
  resolved with MAPF's rules, so *any* rollout is a valid plan. Validity is
  100% in the table below because it cannot be otherwise.
- **true suboptimality.** CBS is optimal, so the ratio is measured against
  ground truth, not against another heuristic.

Measured on `empty_room`, 2 agents, 400k steps of IPPO — and this is the result
worth knowing about:

| method | solved | cost | vs optimal |
|---|---|---|---|
| IPPO, greedy (argmax) | 45% | 10.5 | **1.11x** |
| IPPO, sampled | **100%** | 27.3 | 2.94x |
| CBS (optimal) | 100% | 9.6 | 1.00x |

<sub>Read from [`.docs/assets/rl-benchmark.json`](.docs/assets/rl-benchmark.json),
which `scripts/train_rl.py` writes. The [playground](https://apla-toolbox.github.io/pymapf/#learning)
renders all four settings from that same file.</sub>

The same weights, evaluated two ways — and the gap has **two** causes, measured
over 80 instances (33 greedy failures, no wall contacts, and in every case both
agents solve that instance fine *alone*):

- **70%** are collision-free **period-2 orbits**. The agents never touch. The
  argmax makes each a deterministic function of an observation that contains the
  other agent, and the pair settles onto a closed loop.
- **30%** are **period-1 freezes** with a collision on every step — a genuine
  livelock, two agents each wanting the cell the other holds. The same failure
  PIBT has, reached by a different route.

Both have the same cure: sampling is the only noise in the system, so it always
escapes — and it also wanders, hence 3x the cost. Reporting either number alone
would be reporting half the result, so `compare()` reports both by default.

There is a short film for this layer — `.docs/assets/pymapf-rl-promo.mp4`, built
by `scripts/make_rl_promo.py`. It trains the policy while it renders, so the
split-screen is that policy acting on one shared instance, and the 70/30 split
is measured over 80 instances during the render rather than quoted.

### Reactive planners 🔎

```python
from pymapf.decentralized import MultiAgentNMPC
from pymapf.decentralized.position import Position
import numpy as np

sim = MultiAgentNMPC()
sim.register_agent("r2d2", Position(0, 3), Position(10, 7))
sim.register_agent("bb8", Position(0, 7), Position(5, 10))
sim.register_agent("c3po", Position(10, 7), Position(5, 0))
sim.register_obstacle(2, np.pi / 4, Position(0, 0))
sim.run_simulation()
sim.visualize("filename_test", 10, 10)
```

```python
from pymapf.decentralized.velocity_obstacle import MultiAgentVelocityObstacle
from pymapf.decentralized.position import Position

sim = MultiAgentVelocityObstacle(simulation_time=8.0)
sim.register_agent("r2d2", Position(0, 3), Position(10, 7))
sim.register_agent("bb8", Position(0, 7), Position(5, 10))
sim.register_agent("c3po", Position(10, 7), Position(5, 0))
sim.run_simulation()
sim.visualize("filename_test_2", 10, 10)
```

### Scripts 💨

```bash
python scripts/generate_gallery.py     # every figure in .docs/assets
python scripts/make_promo.py           # the promo film
python scripts/make_rl_promo.py        # the learning-layer film
python scripts/train_rl.py             # train IPPO/MAPPO, benchmark vs CBS
python scripts/build_web_bundle.py     # refresh the playground's copy of the library
python scripts/switch_positions_nmpc.py
```

### The playground 🌐

`.docs/` is a static site that runs PyMAPF in the browser under Pyodide — the
same source files, loaded into a WebAssembly interpreter, with a JavaScript port
of the solvers as an instant-response fallback. Serve it locally with:

```bash
python -m http.server -d docs 8000
```

## Cite 📰

If you use the project in your work, please consider citing it with:
```
@misc{https://doi.org/10.13140/rg.2.2.14030.28486,
  doi = {10.13140/RG.2.2.14030.28486},
  url = {http://rgdoi.net/10.13140/RG.2.2.14030.28486},
  author = {Erwin Lejeune and Sampreet Sarkar},
  language = {en},
  title = {Survey of the Multi-Agent Pathfinding Solutions},
  publisher = {Unpublished},
  year = {2021}
}
```

List of publications & preprints using `pymapf` (please open a pull request to add missing entries):

* [Survey of MAPF solutions](https://www.researchgate.net/publication/348716625_Survey_of_the_Multi-Agent_Pathfinding_Solutions) (January 2021)

## Contribute 🆘

Open an issue to state clearly the contribution you want to make. Upon aproval send in a PR with the Issue referenced. (Implement Issue #No / Fix Issue #No).

## Maintainers Ⓜ️

- Erwin Lejeune
- Sampreet Sarkar
