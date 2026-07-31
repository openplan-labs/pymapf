# Multi-Agent Path Finding and Decentralized Coordination: a 2026 update

*An extension of “Survey of the Multi-Agent Pathfinding Solutions” (Lejeune and
Sarkar, 2021, DOI 10.13140/RG.2.2.14030.28486), covering what changed in the
five years since, and adding the decentralized swarm-control line of work that
the original left out.*

Every algorithm named here is cited in [`REFERENCES.md`](../REFERENCES.md).
Algorithms marked **[impl]** are implemented in this repository and were run to
produce the numbers in [§7](#7-experiments); everything else is surveyed only.

---

## 1. What this update is for

The 2021 survey covered the field as it stood after a decade of conflict-based
search: CBS and its refinements, prioritized planning, and the reactive
methods. Two things have happened since.

The first is a **change in what “large” means**. In 2021 a hard instance had
tens of agents and an optimal solver was the object of study. By 2023 a single
laptop core could solve instances with *thousands* of agents in under a second —
not by making optimal search faster, but by abandoning optimality up front and
searching the joint configuration space with a cheap, complete fallback. LaCAM
and PIBT did that, and they reset the baseline for everything else.

The second is that **anytime became the default contract**. Rather than "give me
the optimal plan" or "give me any plan", the useful interface turned out to be
"here is a deadline, give me the best plan you can by then". MAPF-LNS, LNS2 and
LaCAM\* all fit that shape, and it is the shape a robot fleet actually needs.

A third strand — learning-based MAPF — grew rapidly and, as of 2025, still does
not beat search at scale. That is worth stating plainly because the 2021 survey
predated most of it.

Finally, the original survey treated MAPF as the whole problem. In practice the
same fleets that need discrete plans also need continuous, decentralized
control: flocking to move as a group, coverage to spread over an area. That
work is surveyed in [§6](#6-decentralized-coordination), including the
acceleration-based flocking and limited-range coverage developed at TII's
Autonomous Robotics Research Centre.

## 2. Problem, restated

An instance is a graph `G = (V, E)`, `k` agents, a start and a goal per agent.
Time is discrete; at each timestep an agent moves along an edge or waits. A
solution is a set of paths that is free of **vertex conflicts** (two agents on
one vertex) and **edge conflicts** (two agents swapping along one edge). The
two standard objectives are **sum of costs** and **makespan**.

Stern et al. (2019) is the reference for the variant taxonomy, and this
library follows its definitions: it implements the classic setting with `stay
at target` semantics, unit costs, and both conflict types. Optimal MAPF is
NP-hard for both objectives (Yu and LaValle 2013), which is the fact every
practical method is designed around.

## 3. Optimal MAPF: refinement, then a ceiling

**CBS** **[impl]** remains the structural centre of optimal MAPF: a two-level
search where the high level branches on conflicts and the low level replans one
agent under constraints. The decade after it was a sequence of refinements to
the same skeleton:

| Year | Refinement | What it attacks |
|---|---|---|
| 2015 | ICBS — cardinal conflict prioritisation, meta-agents | which conflict to branch on |
| 2015 | Bypass | branching when a same-cost alternative exists |
| 2018 | CBSH — admissible high-level heuristics | the high level's blindness |
| 2019 | Disjoint splitting | redundancy between the two child subtrees |
| 2019–21 | Corridor / rectangle / target symmetry reasoning | equal-cost symmetric subtrees |
| 2022 | Branch-and-cut-and-price | the LP relaxation of the whole problem |

Together these bought orders of magnitude, and they matter — but they did not
move the ceiling far in *number of agents*. Optimal MAPF on a dense instance
with hundreds of agents remains out of reach, and the field responded by
mostly leaving optimality behind.

This library implements plain CBS plus the conflict-count tie-break, and none
of the rest; the effect is visible in [§7](#7-experiments), where CBS times out
on corridor maps that the suboptimal solvers dispatch in milliseconds. That is
not a defect of CBS, it is the point: the refinements above are what a
production optimal solver needs, and skipping them is what "plain CBS" costs.

## 4. The scalability shift: PIBT and LaCAM

**PIBT** **[impl]** (Okumura et al. 2019, 2022) plans *one timestep at a time*.
Each agent proposes the neighbour closest to its goal; conflicts are resolved by
**priority inheritance** — a blocked high-priority agent lends its priority to
whoever is in the way, recursively, with backtracking when the chain fails.
Cost per step is `O(k · degree)`. It is not complete: it can livelock where an
agent must retreat far from its goal.

**LaCAM** **[impl]** (Okumura 2023) makes that completeness gap disappear. It
searches the space of **configurations** (one vertex per agent), but generates
successors *lazily*: each search node holds a queue of partial assignments, pops
one, and asks PIBT to fill in every other agent in a single sweep. One expansion
= one PIBT call = one successor. Because the queue eventually enumerates every
combination, the search is complete; because PIBT's first suggestion is usually
good, the queue is barely touched.

The follow-ups matter:

* **LaCAM\*** (IJCAI 2023) keeps searching after the first solution and relaxes
  g-values over the explored configuration graph, converging to optimal.
* **LaCAM3 / Engineering LaCAM\*** (AAMAS 2024) adds swap operations,
  monte-carlo configuration generation and refinement; this is the version that
  current comparisons treat as the search-based state of the art.

Measured here (§7), LaCAM's *first* solution is exactly PIBT-quality — as
expected, since PIBT generates it — and its value is completeness: on instances
where PIBT livelocks and CBS proves a solution exists, LaCAM found one in
**6 of 6** cases.

## 5. Anytime MAPF

**MAPF-LNS** **[impl]** (Li et al. 2021) starts from any initial plan and
repeats: destroy a small neighbourhood of agents, re-plan them against the
frozen rest, accept if the sum of costs improved. **MAPF-LNS2** (2022) adds
**SIPPS**, a soft-constraint safe-interval search that returns *colliding*
paths scored by collision count, which lets repair make progress on instances
where a conflict-free repair does not exist yet.

Two implementation details do the heavy lifting:

* **SIPP** **[impl]** as the repair search. Space-time A* pays for every
  timestep of the horizon; SIPP collapses time into safe intervals per vertex.
  Measured here: identical optimal costs on 400 randomised constraint sets, and
  **115× faster** on a long-horizon instance (229 ms → 2 ms).
* **Adaptive operator weights**, so the destroy operator that has been paying
  off recently gets chosen more often.

LNS is the strongest single lever in this library: it takes PIBT's plan on the
warehouse instance from cost 175 to **113** in two seconds.

## 6. Decentralized coordination

MAPF assumes a graph, a central planner, and full information. A drone fleet
has none of those. This is the part the 2021 survey did not cover.

### 6.1 Flocking

Forty years of models, all implemented here as interchangeable objects and run
on identical initial conditions (20 agents, lattice spawn, 300 steps, no
waypoint):

| Model | order | cohesion | min sep | speed | neighbours | violations |
|---|---|---|---|---|---|---|
| boids (Reynolds 1987) | 0.57 | 2.33 | 1.29 | 0.40 | 18.5 | 149 |
| vicsek (1995) | 1.00 | 7.07 | 2.29 | 4.00 | 4.8 | 0 |
| cucker_smale (2007) | 1.00 | 5.32 | 1.59 | 0.24 | 6.4 | 0 |
| olfati_saber (2006) | 0.00 | 5.24 | 2.97 | 0.00 | 7.5 | 0 |
| proximal (Vásárhelyi 2018) | 0.59 | 64.53 | 2.34 | 4.00 | 2.2 | 0 |
| **active_elastic** (Ferrante 2012/13) | 0.99 | 6.10 | 2.66 | 1.99 | 3.0 | 0 |
| **acceleration** (Iacone et al. 2024) | 1.00 | 3.08 | 1.64 | 3.20 | 14.8 | 0 |
| **gaussian_kernel** (Manoni et al. 2022) | 1.00 | 4.11 | 2.25 | 3.20 | 10.3 | 0 |
| **minimalistic** (Amorim et al. 2024) | 0.99 | 4.24 | 2.35 | 2.00 | 8.0 | 0 |
| **distributed_3d** (Albani et al. 2022) | 0.59 | 64.53 | 2.34 | 4.00 | 2.2 | 0 |

The last two are 2022–2024 work from the Albani / Ferrante / Manoni / Saska
group, and both are *three-dimensional* methods, so the planar table above
understates them. Repeating it in 3D:

| Model | order | cohesion | min sep | speed | neighbours | violations |
|---|---|---|---|---|---|---|
| proximal | 0.98 | 2.94 | 2.27 | 4.00 | 16.6 | 0 |
| active_elastic | 0.74 | 18.53 | 2.42 | 2.00 | 3.0 | 0 |
| acceleration | 1.00 | 2.42 | 1.73 | 3.20 | 19.0 | 0 |
| **minimalistic** | 0.88 | 3.26 | 2.27 | 1.96 | 8.0 | 0 |
| **distributed_3d** | 0.99 | 2.65 | 1.72 | 4.00 | 17.8 | 0 |

Read as statements about what each model is *for*:

* **Vicsek** aligns perfectly and never collides because it has no attraction at
  all — and disperses for the same reason.
* **Cucker–Smale** reaches velocity consensus (order 1.00) but at the *average*
  initial velocity, which is near zero: it is a consensus law, not a complete
  flocking controller. Compose it with a spacing behavior — which the object
  model makes a one-liner — and it becomes one.
* **Olfati-Saber** converges to an exact α-lattice at the reference spacing
  (3.00 m) and then stops. An α-lattice is a *static* formation; collective
  motion needs the navigational feedback term.
* **Proximal control** is designed for confined environments, and it shows: in
  the open plane it disperses (cohesion 65). Give it a boundary or a migration
  target — the conditions the paper assumes — and it is tight (cohesion 4.9,
  order 0.91), and in 3D it needs neither (cohesion 2.94). This is a property of
  the bounded, decaying Lennard-Jones attraction, not a defect of the
  implementation.
* **Active elastic** (Ferrante et al.) reaches order 0.95 with zero violations
  *without ever measuring a neighbour's velocity or heading*. Alignment is not
  computed; it emerges from the elastic modes. That is the result that matters
  for cheap hardware: heading sensing is the expensive part of a flocking robot.
* **Acceleration-based** (Iacone, Lejeune, Manoni, Manfredi, Albani) is the only
  model here that holds full polarisation *and* legal spacing *and* keeps flying
  at 3.2 m/s with nothing to chase. Self-propulsion and drag are what buy that.
* **Gaussian-kernel arbitration** (Manoni, Albani et al.) matches it on order and
  safety with half the effective neighbours, because the kernel decides who
  matters rather than a hard radius.
* **Minimalistic flocking** (Amorim, Nascimento, Chaudhary, Ferrante, Saska
  2024) is the floor of the field and the most interesting row in the table.
  Each agent measures relative range and bearing to its neighbours and nothing
  else — no GPS, no compass, no communication, no velocity sensing — and the
  group still reaches order 0.99 with zero violations, travelling in a common
  direction that *nobody transmitted*. With no channel to agree on a heading,
  agreement can only come out of the dynamics, and it does. Validated in the
  original work with nine UAVs over a desert.
* **Distributed 3D flocking** (Albani, Manoni, Saska, Ferrante 2022) is the one
  model here that treats the vertical axis as a different problem, because for a
  multirotor it is: climbing is expensive, and a drone below another sits in its
  downwash. In 3D it reaches order 0.99 at full cruise speed and settles into a
  lattice with a vertical-to-horizontal spread of 0.44, against 0.71 for the
  isotropic law — flat and wide, which is the shape a rotorcraft swarm should
  hold. In the plane it reduces exactly to `proximal`, which is why the two
  share a row above: with no vertical axis there is nothing to shape, and the
  anisotropic terms are inert by construction.

The neighbourhood rule turns out to matter as much as the control law. Holding
the acceleration model fixed and changing only who each agent sees:

| Neighbourhood | order | cohesion | min sep | neighbours |
|---|---|---|---|---|
| metric, r = 6 m | 1.000 | 3.21 | 1.60 | 14.1 |
| topological, k = 5 | 0.998 | 4.72 | 2.17 | 5.0 |
| cone, 135° | 0.835 | 22.86 | 1.87 | 5.5 |
| Gaussian kernel | 0.993 | 12.02 | 2.40 | 6.9 |

Topological sensing gives *better* spacing (2.17 m vs 1.60 m) with a third of
the connectivity — the same finding that motivated the topological hypothesis
for real starlings.

### 6.2 Coverage

**Lloyd's algorithm** on a Voronoi partition (Lloyd 1982; Cortés et al. 2004) is
the canonical deployment law: own what you are nearest to, move to its weighted
centroid, repeat. Written against a *domain* rather than the plane, one
implementation deploys a team on any surface — 9 agents, 40 iterations:

| Domain | cost reduction |
|---|---|
| rectangle | 84% |
| disk | 91% |
| sphere | 88% |
| **hemisphere** | 92% |
| annulus (a perimeter) | 96% |

Four variants matter in practice, all from the TII/UNIMORE line of work:

* **Limited range** (Bertoncelli, Belal, Albani, Pratissoli, Sabattini): an
  agent owns its Voronoi cell *intersected with a disc*. Measured at 61% against
  84% unlimited — not a worse algorithm, a harder problem, because a team that
  starts clustered has no gradient to spread along at all.
* **Adaptive** (Schwager et al.): the density is *estimated online* from what
  agents measure. 28% reduction, and the gap to the informed case is the honest
  price of not being handed the map.
* **Hemispherical surfaces** (Belal, Manoni, Albani, Sabattini, ANTS 2026):
  geodesic cells and spherical centroids on a dome.
* **Time-varying targets** (Manoni et al.): the centroid has moved by the time
  you arrive, so the controller pursues with a feed-forward term rather than
  converging.

### 6.2b Distribution control and Gaussian mixtures

Assigning a goal per agent does not scale. Distribution control replaces "agent
47 goes here" with "the swarm should look like *this density*", and a Gaussian
mixture is the natural way to write that density down: it is multi-modal, it can
be **fitted** to observations by EM, **sampled** from, and evaluated anywhere.

Steering 30 agents to a three-component target (mean distance from a target
sample to the nearest agent — lower is better):

| Controller | error before | after | min separation | allocation |
|---|---|---|---|---|
| `density_matching` | 2.29 | 1.41 | 1.33 | — |
| `mixture_assignment` | 2.29 | **0.90** | 2.09 | **12 / 12 / 6** (quota 12 / 12 / 6) |

`mixture_assignment` hits the mixing weights *exactly*, because allocation is a
capacity-constrained assignment rather than an argmax over responsibilities.
That distinction is the whole finding: the obvious implementation — assign each
agent to its most-responsible component — never splits a team at all, since
agents that launch together are all nearest the same component and hysteresis
locks it in. Scaling the scores by a quota pressure does not fix it either;
responsibilities are near-degenerate (1e-30 vs 1) and scaling zero is still
zero. Only an explicit capacity constraint produces the split.

`density_matching`'s residual separation of 1.33 m against a 1.5 m requirement is
not a tuning failure: matching a density and enforcing a minimum separation are
**conflicting objectives** when the target's peak density is higher than the
separation distance permits. The controller trades them, and the trade is
visible in the number.

End to end: sample 500 observations from an unknown field, fit a two-component
mixture by EM (recovered means (3.93, 3.90) and (15.15, 14.04) against a truth
of (4, 4) and (15, 14)), and hand the *fitted* density to a coverage controller
— 83% cost reduction, the same as covering the true density.

### 6.2c Formation control

Flocking asks a swarm to move together and coverage asks it to spread out.
Formation control asks for something stricter — a *specified geometry*, held
while the group moves — and the interesting question is not how to hold it but
**what each agent is allowed to measure**. Oh, Park and Ahn's 2015 survey
organises the field on exactly that axis, and the taxonomy is predictive: what
you can sense determines which symmetry you can fix, and no gain tuning changes
it.

| Constraint | Agent measures | Needs | Formation fixed up to | Class |
|---|---|---|---|---|
| Displacement | relative position in a shared frame | a compass or common heading | translation | `DisplacementFormation` **[impl]** |
| Distance | inter-agent range | nothing shared | translation, rotation, **reflection** | `DistanceFormation` **[impl]** |
| Bearing | direction to neighbours | nothing shared | translation, **scale** | `BearingFormation` **[impl]** |
| Leader–follower | offset from a designated leader | leader tracking | translation | `LeaderFollower` **[impl]** |

Nine agents, identical lattice spawn, averaged over the V, circle and grid
targets. `error` is the mean per-agent distance from the best-fitting placement
of the shape — fitted, crucially, under the symmetry group that controller is
*entitled* to (see below). Convergence time is to 1% of the initial error.

| Controller | initial error | final error | time to 1% (s) | min sep |
|---|---|---|---|---|
| displacement | 1.90 | 0.0000 | 3.6 | 3.00 |
| distance | 1.81 | 0.0000 | 15.6 | 3.00 |
| bearing | 2.00 | 0.0005 | 41.1 | 1.31 |
| leader–follower | 1.90 | 0.0000 | 4.7 | 3.00 |

The ordering is the taxonomy restated as a cost: **the less you sense, the
longer it takes**. Displacement control converges in 3.6 s because every agent
already knows its absolute slot. Distance control takes 4× longer for the same
final error, because the group must first agree on an orientation nobody can
measure. Bearing control takes 11× longer still, and its closest approach is the
worst of the four — with scale unconstrained, the formation breathes on its way
in, and a contracting phase is exactly when agents get close.

Three findings came out of building this, and all three were originally
mistaken for controller failures.

**1. The error metric must quotient out the symmetry the sensing cannot fix.**
A distance-based controller that lands on a *mirror image* of the target has
succeeded — distances cannot see handedness. A bearing-based controller that
lands on a half-size copy has succeeded — bearings cannot see scale. Graded
against a fixed pose, both looked stuck: the distance controller reported an
error of 3.58 on a V formation whose entire desired distance matrix it was
satisfying **to 10⁻¹³**. Each controller here declares its own group, and
`formation_error` fits under exactly that group. This is not a testing detail;
it is the same statement as the taxonomy, made measurable.

A related subtlety: pose and correspondence have to be solved *together*.
Procrustes needs to know which agent holds which slot; the assignment needs to
know the pose. Fitting either first with the other guessed scores an
exactly-converged formation as a failure, and alternating from a single start
can lock onto a local optimum — a 3×3 grid rotated by 1.1 rad reports an error
of 1.10 instead of 0. Alternating from several seed rotations fixes it.

**2. A range-limited interaction graph is the wrong constraint set.** Distance
and bearing control only pin a formation down if the constraint graph is
*rigid*, and the proximity graph of whoever is in sensing range is not rigid in
general — worse, it changes as the swarm moves, so the constraint set being
descended shifts underneath the descent. The observed failure was spectacular
and entirely explicable: pairs pushed apart to their desired distance, left
sensing range, and the pairs that should have pulled them back were never in it.
Error grew from 3.6 to **26.8**. Building the graph once from the target shape,
and augmenting a sparse one with the shortest missing edges until
`is_infinitesimally_rigid` accepts it, converges to zero.

The rigidity check earns its place by predicting the residual failures too. A
*collinear* target in 2D and a *planar* target in 3D are not infinitesimally
rigid at any edge count — the complete graph on six collinear points has
rigidity-matrix rank 5 against the 9 required — so a first-order controller has
flex modes it cannot see, and stalls. Those are the only two cases in the whole
sweep where the distance and bearing controllers fail to converge, and the
library now warns before running rather than converging quietly to the wrong
thing.

**3. A waypoint is a translation, not an attraction.** The obvious way to fly a
formation somewhere — pull every agent toward the target point — is a
*contraction*, and a contraction is a deformation. It fights the shape term for
every controller, and it destroys the bearing controller outright: scale is
precisely the degree of freedom bearings do not constrain, so with nothing
resisting, all nine agents collapsed onto the waypoint and the formation
reported convergence at zero size. Computing the command once from the
*centroid* error and issuing it identically to everybody puts it in the null
space of all four laws, which are translation-invariant by construction. The
formation then slides to the waypoint without deforming at all.

Two smaller notes. The textbook distance potential, Krick et al.'s
`Σ(|p_ij|² − d_ij²)²`, is cubic in the error: an agent one formation-width off
requests roughly a hundred times the acceleration it can deliver, and what the
vehicle executes is saturated bang-bang. `Σ(|p_ij| − d_ij)²` has the same
equilibria and the same rigidity theory with a proportional demand, and is the
default here; the original is a constructor argument. And slot assignment is
solved exactly (Hungarian, O(n³), dependency-free) rather than by index —
assigning by index makes agents cross the formation to reach a slot someone else
is standing next to, which is both slower and the main source of collisions
while forming up.

### 6.3 Reactive avoidance

**Velocity obstacles** **[impl]** (Fiorini and Shiller 1998) and their
reciprocal forms (RVO 2008, ORCA 2011) remain the standard local layer, and
**NMPC** **[impl]** (Kamel et al. 2017) the standard optimisation-based one. In
a deployed system these sit *under* a planner: MAPF or LaCAM decides the
routes, and the reactive layer absorbs the disturbances that discrete plans
cannot model.

## 7. Experiments

Every number below was produced by this repository — `python -m
pymapf.experimental.study` and `scripts/generate_gallery.py` — on the scenario
families in `pymapf.scenarios`. Comparisons are **paired**: same instance, same
seed, same budget, ratio reported per instance, because MAPF instance difficulty
spans orders of magnitude and unpaired means mostly measure which instances were
sampled.

Results are reported with their failures. Three of the five experiments below
produced a *negative or mixed* result, and those are the useful ones.

### 7.1 Where plain CBS stops

On the warehouse instance with 8 agents, optimal CBS needs **6 470 node
expansions and 5.3 s** for cost 100. Weighted CBS (`w = 1.5`) reaches cost 104
in **23 expansions and 18 ms** — 4% more cost for a 290× reduction in work. On
the bottleneck and maze families, plain CBS does not finish inside 5 s at all,
while PIBT and LaCAM return valid plans in ~4 ms.

### 7.2 SIPP versus space-time A*

400 randomised constraint sets on 10×10 maps: **identical path costs in every
case**, zero constraint violations. On an instance with a vertex blocked for 400
timesteps, space-time A* takes 229 ms and SIPP 2 ms — the difference between
paying per timestep and paying per safe interval.

### 7.3 Completeness: PIBT versus LaCAM

Searching for instances where PIBT fails but a solution provably exists (CBS
found one), LaCAM solved **6 of 6**, at 1.3–2.2× the optimal cost. This is the
concrete value of LaCAM's lazy constraint tree: PIBT alone livelocks on exactly
these instances.

*Bugs this experiment exposed, both fixed:* priority inheritance can produce an
edge conflict when an earlier-decided agent has already committed to the
caller's vertex, and a failed inheritance chain leaks assignments made deeper in
the recursion unless the whole chain is journalled and rolled back. Both
produced invalid plans that a validity sweep caught (306 runs, now all valid).

### 7.4 The experimental variants

Three deliberate deviations, each measured against the algorithm it modifies,
plus one question about heuristics. Paired ratios; **ratio < 1 means the
variant is cheaper**. Raw data: `docs/assets/experiments.{json,csv}`.

**A. Congestion-aware PIBT.** Add a static penalty — how many agents'
individually-shortest paths cross a vertex — to PIBT's distance-based
preference. 25 instances:

| α | mean ratio | median | win/tie/loss | instances solved |
|---|---|---|---|---|
| 0.0 (= PIBT) | 1.000 | 1.000 | 0/25/0 | 69% |
| 0.1 | 0.940 | 0.938 | 18/2/5 | **78%** |
| 0.3 | 0.940 | 0.938 | 18/2/5 | 78% |
| 0.6 | 0.940 | 0.938 | 18/2/5 | 78% |
| 1.0 | 0.940 | 0.938 | 18/2/5 | 78% |
| 2.0 | 0.918 | 0.931 | 14/1/2 | 56% |

6% cheaper *and* 9 points more instances solved. But look at the middle four
rows: **α from 0.1 to 1.0 gives byte-identical results.** That is the actual
finding. Distances on a unit grid are integers, so any penalty smaller than 1
can only ever break *ties* between equidistant neighbours — it never trades
distance for congestion. The entire gain comes from preferring the less
contested of two equally-good moves. At α = 2.0 the penalty finally starts
overriding distance, and PIBT's success rate collapses from 78% to 56%,
because the thing that makes PIBT terminate is that agents move *toward* their
goals.

So: a real improvement, but not the one hypothesised, and with a sharp cliff
immediately past it.

**B. Delay-targeted LNS.** Pick the destroy neighbourhood around the agent
whose path most exceeds its individual lower bound. 25 instances at a 0.5 s
budget: **mean ratio 1.000, median 1.000, 5 wins / 18 ties / 2 losses.**

No effect. The hypothesis — that delay identifies where the slack is — is
plausible and is what MAPF-LNS's own agent-based operator does, but at this
budget the stock operators already find the same improvements, and the roulette
weighting means a fourth operator mostly dilutes the other three. A negative
result, reported as one.

**C. What to do with LaCAM's leftover budget.** 28 instances, equal budget:

| Use of the budget | mean ratio | median | win/tie/loss |
|---|---|---|---|
| first solution only | 1.000 | 1.000 | 0/28/0 |
| continue the search (our LaCAM\* approximation) | **1.000** | 1.000 | **0/28/0** |
| randomised restarts, keep best | 0.808 | 0.896 | 26/28 → 26/2/0 |
| LNS refinement of the first solution | **0.786** | 0.878 | 25/3/0 |

Two things here. First, our in-search continuation is **worthless** — 0 wins in
28. It relaxes g-values over the explored configuration graph, but LaCAM's
depth-first stack rarely re-reaches the goal configuration by a cheaper route,
so the relaxation has nothing to propagate. This is a limitation of *our*
approximation, not of LaCAM\*, which implements a full cost-propagation scheme
we did not; it is recorded here because the honest version of "we implemented
LaCAM\*" is "we implemented part of it and it did not work".

Second, the two cheap alternatives both win decisively: restarting the
randomised search buys 19%, and simply handing the first solution to LNS buys
21%. If you have a budget and this library, spend it on LNS.

**This measurement changed the library.** `LaCAM(anytime=True)` no longer
continues the search in place — it now spends the remaining budget on
randomised restarts, because that is what the numbers supported. On the
warehouse instance the effect is cost 175 → 114. Shipping the version that won
0 of 28 would have been shipping a placebo.

**D. True-distance heuristic in the CBS low level.** Exact goal distances (one
backward Dijkstra per goal) instead of Manhattan, 33 instances:
**runtime ratio 0.955** (19 wins / 14 losses), **expansion ratio 1.022** with 28
of 33 instances tied exactly.

Essentially no effect, and worth understanding why: on these map sizes
Manhattan is already close to exact, and CBS's cost is dominated by *high-level*
node count, which the low-level heuristic does not change. The one maze
instance in the sample ran 35% faster — the regime where the exact heuristic
should pay — but n = 1 is an anecdote, not a result. The heuristic's real
justification in this library is different: it is the only admissible option on
a general graph with no coordinates, which is why PIBT and LaCAM use it
unconditionally.

### 7.5 Flocking controllers, measured

20 agents, identical lattice spawn, 300 steps, 10 Hz. `order` is Vicsek
polarisation; `min sep` is the closest approach in steady state (the separation
distance is 1.5 m).

| Controller | order | cohesion | min sep | speed | steady violations |
|---|---|---|---|---|---|
| boids | 0.65 | 2.23 | 0.87 | 0.41 | 151 |
| vicsek | 1.00 | 6.30 | 2.12 | 4.00 | 0 |
| olfati_saber | 0.00 | 5.11 | 3.00 | 0.00 | 0 |
| **acceleration** | **1.00** | **3.25** | **1.56** | **3.20** | **0** |

*(free flock, no waypoint)*

Read carefully, this table is a set of statements about what each model is
*for*. Vicsek aligns perfectly and never collides because it has no attraction
at all — it also disperses (cohesion 6.3). Olfati-Saber converges to an exact
α-lattice at the reference spacing (3.00 m) and then **stops**: an α-lattice is
a static formation, and collective motion needs the navigational feedback.
Boids stalls without a waypoint and violates spacing constantly at these gains.
Only the acceleration-based model holds full polarisation *and* legal spacing
*and* keeps flying at 3.2 m/s with nothing to chase — which is precisely what
self-propulsion and drag are there to do.

Two findings worth recording, both discovered by measurement:

1. **Olfati-Saber's interaction range is part of the algorithm, not a free
   parameter.** At `r/d = 2` (this library's default sensing range) the lattice
   *collapses* — every agent sees enough distant neighbours that summed
   attraction beats local repulsion, and min separation goes to 0.00 m. At the
   paper's `r ≈ 1.2 d` it produces an exact lattice. The failure is silent:
   the flock looks cohesive, it is simply all in one place.
2. **The σ-norm gradient vanishes at contact.** `n_ij` scales with the
   separation, so repulsion peaks at intermediate range and *fades* as two
   agents converge. A pair driven together by other terms can merge and stay
   merged. The acceleration model does not have this failure mode because its
   short-range term is an unbounded `1/d²`.

A third, mundane but expensive: an unsaturated waypoint term
`gain · (target − position)` grows without bound with distance, so with a
waypoint 60 m away it consumed the entire acceleration budget and the clamp
scaled collision avoidance to nothing. Bounding navigational authority to a
fixed share of the budget fixed it for every controller.

### 7.6 Learned MAPF, measured against the optimum

The 2021 survey covered learning-based MAPF (§4) but could not benchmark it,
because a learned-MAPF number is normally a *success rate* — the optimum is too
expensive to obtain per instance. That is exactly the thing this repository
already owns, so the tables below report **true suboptimality**: the learned
sum-of-costs over the CBS sum-of-costs, computed by the planner's own code on
the same instance.

IPPO and MAPPO (`pymapf.rl`), shared parameters, 400k environment-agent steps
each, egocentric 9×9 observation, potential-based shaping from the exact
distance oracle. 100 held-out instances per setting.

| setting | method | solved | cost | vs optimal |
|---|---|---|---|---|
| 8×8, 2 agents | ippo (greedy) | 45% | 10.5 | **1.11x** |
| | ippo (sampled) | **100%** | 27.3 | 2.94x |
| | mappo (greedy) | 44% | 10.5 | 1.11x |
| | mappo (sampled) | **100%** | 27.4 | 2.97x |
| | CBS | 100% | 9.6 | 1.00x |
| | PIBT | 100% | 9.9 | 1.03x |
| 10×10, 4 agents | ippo (greedy) | 10% | 30.6 | 1.38x |
| | ippo (sampled) | 94% | 128.3 | 5.33x |
| | mappo (sampled) | 94% | 122.2 | 5.24x |
| | CBS | 100% | 24.1 | 1.00x |
| 10×10 + obstacles | ippo (sampled) | 65% | 164.6 | 6.68x |
| | mappo (sampled) | 67% | 166.2 | 6.84x |
| | CBS | 100% | 26.7 | 1.00x |
| | PIBT | 97% | 29.4 | 1.11x |
| bottleneck, 4 agents | ippo (sampled) | 58% | 299.1 | 7.43x |
| | mappo (sampled) | 0% | — | — |
| | CBS | **1%** | 58.0 | 1.00x |
| | PIBT | 100% | 77.7 | 1.07x |

Four things fall out of this, and only the first is the one people usually
report.

**1. How you sample the policy matters more than which algorithm trained it.**
The same weights, evaluated two ways, are either a 45%-success policy at 1.11x
optimal or a 100%-success policy at 2.94x. Taking the argmax makes the policy
deterministic, and two deterministic agents that both want the same cell are
refused by the conflict rules, revert, and then choose exactly the same thing
again — a **livelock**, and precisely the failure mode PIBT has on 25% of
warehouse instances (§7.3). Sampling breaks the symmetry, so the flock always
eventually gets through; it also wanders, hence three times the cost. Reporting
either number alone reports half the result, so the harness reports both.

**2. IPPO and MAPPO are indistinguishable here.** 45% versus 44%, 1.11x versus
1.11x, 94% versus 94%. That is the MAPPO paper's own finding restated: the
centralized critic is not where the performance comes from. It is worth saying
plainly because the two differ in this codebase by a single class attribute, so
there is no implementation gap hiding a real one.

**3. The learned policies are conflict-free by construction, not by training.**
Validity is 100% in every row including the random baseline, because the
environment resolves vertex, edge and cascading conflicts with MAPF's rules
rather than letting agents overlap and penalising it afterwards. This is a
modelling choice with teeth: it means a policy can never score well by cheating
on the physics, and it means "success rate" and "collision rate" are not
trading off against each other.

**4. On the hardest setting the *optimal* planner is the one that fails.** CBS
closed 1% of the bottleneck instances inside five seconds. PIBT closed 100% of
them at 1.07x. The learned policy closed 58% at 7.43x. If you only ever compare
against an optimal baseline you will not be able to run this row at all — which
is the practical argument for keeping a suboptimal planner in the comparison,
and the reason the harness takes a list of baselines rather than one.

Two things that did **not** work, recorded so they are not re-attempted. The
training curve peaks near 100% solved around 100k steps and settles near 50%,
and neither of the standard remedies touches it: KL-based early stopping never
fires, because the measured per-update KL stays between 0.0004 and 0.014 and
the conventional threshold is 0.02 — runs with and without it are bit-identical.
Raising the entropy coefficient from 0.01 through 0.03 to 0.05 moves the final
solve rate from 52% to 53% to 53%. The curve is not the policy degrading; it is
the greedy-evaluation livelock of finding (1) arriving as the policy sharpens.

## 8. Open problems

1. **Optimality at scale.** LaCAM\* converges to optimal in the limit, but the
   gap between "first solution in 5 ms" and "provably optimal" is still where
   most of the research effort sits.
2. **Learning has not overtaken search.** As of 2025, MAPF-GPT and its
   relatives are competitive on small dense instances and behind on large ones.
   The interesting question is no longer "can learning do MAPF" but "which part
   of a search-based solver should be learned" — LNS2+RL (learning the destroy
   operator) is the most promising shape.
3. **Lifelong and execution-time robustness.** RHCR handles continuously
   arriving tasks; ADG handles execution delays. A system that does both, under
   kinematic constraints, is still assembled by hand per deployment.
4. **The planning/control seam.** The discrete plan and the continuous
   controller are still designed separately and glued at runtime. Nothing in
   this survey — including our own work — makes the guarantees compose.
5. **Reproducibility.** Most published MAPF results come from separate C++
   codebases with separate benchmark harnesses. Cross-implementation
   comparisons remain rare, and this library is not big enough to change that,
   but its benchmark harness and scenario registry were written with that
   complaint in mind.

## 9. How to reproduce everything here

```bash
pip install -e ".[all,dev]"

python -m pymapf.experimental.study --scale full --output docs/assets
python scripts/generate_gallery.py
pytest
```

The playground at [`docs/index.html`](index.html) runs the same solvers in a
browser, so any claim in §7 can be poked at directly.
