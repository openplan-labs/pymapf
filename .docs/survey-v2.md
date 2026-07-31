# Multi-Agent Path Finding and Decentralized Coordination: second edition

*Second edition of our 2026 update to “Survey of the Multi-Agent Pathfinding
Solutions” (Lejeune and Sarkar, 2021, DOI 10.13140/RG.2.2.14030.28486). The
[first edition](survey.md) remains the reference for everything implemented and
measured in this repository; this edition revises the **framing**, because the
field's centre of gravity moved while we were writing it.*

Every algorithm named here is cited in [`REFERENCES.md`](../REFERENCES.md).
Algorithms marked **[impl]** are implemented in this repository and were run to
produce their numbers; everything else is surveyed only, from published work.
The literature scan behind this edition, with identifiers and the depth to which
each was read, is in [`research-notes.md`](research-notes.md).

---

## 1. What is different about this edition

The first edition told a story with optimal MAPF at the centre: CBS and its
refinements at the top, suboptimal methods below as compromises, and learning
off to one side as a promising alternative that did not yet work. That story was
a fair reading of 2021–2024. It is now the wrong shape, in three specific ways.

**One-shot MAPF is no longer the problem being solved.** The deployed problem is
*lifelong* MAPF: agents are given a new goal the moment they reach the last one,
nothing ever terminates, and the objective is **throughput**, not sum-of-costs.
Sum-of-costs optimality — the thing CBS gives you, and the thing this
repository's §7 measures everything against — is not an approximation of the
lifelong objective. It is a different objective.

**There is a third lever nobody in the first edition pulled.** Every method in
that edition improved the *solver*. A line of work starting with guidance-graph
optimisation improves the **environment** instead: reweight and re-orient the
graph so that ordinary planners produce less congested traffic on it. It is
orthogonal to solver quality and, in a warehouse, far cheaper to deploy.

**Learning found its job, and it is not being the solver.** The first edition
recorded that learned MAPF underperformed search. That is still true of learned
policies used *as* solvers — we reproduce it below, at some cost to our own RL
layer. What changed is that the competitive systems stopped trying: they embed a
learned heuristic, priority, or neighbourhood *inside* a sound search, and let
the search keep responsibility for feasibility.

Sections 2–6 revise the survey around those three shifts. Section 7 reports what
this repository actually measured, unchanged in substance from the first edition
but reframed. Section 8 revises the open problems accordingly.

---

## 2. Lifelong MAPF, and why the objective change matters

In LMAPF an agent that arrives is immediately re-tasked. There is no terminal
state and no plan to be optimal *with respect to* — the system runs
indefinitely, and the number that matters is goals completed per unit time.

Three consequences, none of which is cosmetic:

1. **Optimality stops being well-defined.** You can be optimal over a horizon,
   but the horizon is a modelling choice rather than a property of the problem.
   The literature has largely stopped reporting suboptimality ratios for LMAPF
   and reports throughput instead.
2. **Replanning frequency becomes the binding constraint.** A planner that takes
   500 ms for a beautiful plan is worse than one that takes 50 ms for a mediocre
   one, because the mediocre one gets ten chances to correct itself.
3. **Congestion is a steady-state property.** In one-shot MAPF congestion is a
   transient to be routed around. In LMAPF it is the equilibrium the system
   settles into, which is what makes §4 possible at all.

Scaling LMAPF to realistic settings (arXiv:2404.16162) is the position paper
worth reading here: it enumerates what benchmark LMAPF still abstracts away —
kinematics, task assignment, robot failure — and is candid that these are not
details.

**Where this repository stands.** Everything here is one-shot. The RL
environment already draws a fresh instance on reset, so a lifelong mode is a
small change rather than a rewrite; it is the first item in §8.

## 3. Search: LaCAM as an engineering programme

The first edition treated LaCAM as an algorithm — configuration space, lazy
constraint generation, PIBT as successor generator — and reported that it solved
6 of 7 instances where PIBT livelocked and CBS proved a solution existed. That
remains our result. What has changed is that LaCAM became a *programme*, and
most of the subsequent gain is engineering rather than theory.

**LaCAM3** (Engineering LaCAM\*, AAMAS 2024) is the reference implementation the
field now benchmarks against: real-time, large-scale, near-optimal, and
explicitly an engineering paper.

**Local guidance** (arXiv:2510.19072) moves guidance *into* configuration
generation rather than applying it as a global heuristic over the search.
**Lifelong LaCAM with local guidance** (arXiv:2605.16855) carries that into §2's
setting.

**Lightweight traffic maps** (arXiv:2603.07891) accumulate congestion
information *during* the search and feed it back into node ordering, improving
anytime convergence and final quality.

That last one lands directly on a negative result of ours. Our §7.4 experiment
found LaCAM's in-search anytime relaxation won **0 of 28** paired instances and
had to be replaced with randomised restarts (which then won 26 of 28). We
reported that as "the g-relaxation had nothing to propagate". The traffic-map
work suggests the sharper reading: an anytime pass needs *new information* to
propagate, and congestion measured during the search is that information.
Relaxation alone re-derives what the first pass already knew.

## 4. The third lever: optimising the graph, not the planner

This is the genuinely new axis, and the one the first edition missed.

The idea: leave the planner alone and change the graph it runs on. Assign edge
weights — and, in the newer work, edge *directions* — that make congested
manoeuvres expensive, so that an ordinary planner routing greedily on the
modified graph produces globally better traffic.

- **Guidance graph optimisation for LMAPF** (arXiv:2402.01446) sets up the
  problem: learn edge weights that maximise throughput.
- **Edge directions and weights for mixed guidance graphs** (arXiv:2602.23468)
  adds direction, i.e. partial one-way systems, to the search space.
- **Multi-robot coordination and layout design for automated warehousing**
  (arXiv:2305.06436) pushes it one level further out and co-designs the physical
  layout with the coordination policy.

Two reasons this deserves a section rather than a footnote. It is **orthogonal**
to solver quality — a better guidance graph helps whatever planner you already
run, and the gains compose. And it is **the cheapest thing to deploy**: an
operator can change floor markings and aisle directions far more readily than
they can qualify a new planner.

**Where this repository stands.** `ExplicitGraph` already carries arbitrary edge
weights, so the representation exists and the planners already honour it. What
is missing is the optimiser. This is the second item in §8.

## 5. Learning, and what it is actually for

The 2021 survey covered learned MAPF (PRIMAL and successors) as a candidate
replacement for search. The first edition of this update repeated that framing
and found the candidates wanting. Both were asking the wrong question.

**The pattern in every competitive system since:** the learned component supplies
a *heuristic, a priority, or a neighbourhood*, and a sound search retains
responsibility for feasibility.

- **LaGAT / MAGAT+** (arXiv:2510.17382, AAAI) embeds a learned graph-attention
  heuristic inside LaCAM, and beats both pure search and pure learning in dense
  scenarios — the regime where each alone is weakest.
- **SILLM** (arXiv:2410.21415, ICRA 2025) imitates a search-based expert but
  keeps single-step collision resolution and global guidance at execution,
  reaching **10,000 agents** across six maps.
- **MAPF-World** (arXiv:2508.12087) learns a world model of the joint transition
  and plans against it.
- **Transformer heuristics for MAPF-LNS** learn *which neighbourhood to destroy*
  rather than what to do — the most surgical version of the pattern.

None of these lets a network emit the final joint action unchecked.

**Our own result is evidence for the same conclusion, from the negative side.**
We built a full RL layer — PettingZoo-parallel environment over our own
instances, IPPO and MAPPO, benchmarked against our own optimal CBS — precisely
so the comparison would be honest (§7.6). The learned policy emits the joint
action directly, and it is not competitive as a solver: 45% solved at 1.11×
optimal under argmax, 100% at 2.94× when sampled, against CBS at 100% and 1.00×
and PIBT at 100% and 1.07×. As a *heuristic* it would be a reasonable thing to
put inside a search. As a solver it is not close, and the literature agrees.

## 6. The competition as a forcing function

The **League of Robot Runners** (ICAPS) fixed the evaluation for LMAPF: a
common execution model, throughput as the score, and a clock. It has done for
this field roughly what a benchmark with prize money usually does — collapsed
the space of plausible-sounding methods down to the ones that survive contact
with a deadline.

Entries are worth reading precisely because they are not papers. The 2023
winning entry is public
([DiligentPanda/MAPF-LRR2023](https://github.com/DiligentPanda/MAPF-LRR2023)),
and it is a *combination* — guidance, a fast one-step rule, and an anytime
improvement loop — rather than a single idea, which is itself the finding.

## 7. What we measured

Unchanged in substance from the [first edition](survey.md), where the full
tables, methodology and reproduction instructions live. Reframed here against
the three shifts above.

**Optimal MAPF has a ceiling, and we hit it.** Plain CBS needs 6,470 expansions
and 5.3 s for cost 100 on an 8-agent warehouse instance; weighted CBS reaches
104 in 23 expansions and 18 ms. On the bottleneck instances CBS closed **1%**
inside five seconds while PIBT closed **100%** at 1.07× — the clearest
demonstration in our own numbers that optimality is the wrong target once
instances get hard.

**Suboptimal methods are the working answer.** LaCAM's completeness is real (6 of
7 instances where PIBT livelocked and CBS proved a solution existed). PIBT is
incomplete and livelocks on ~25% of warehouse instances, which our tests encode
rather than hide.

**Three negative results**, reported as negative: delay-based LNS had no effect;
LaCAM's in-search anytime relaxation won 0 of 28 (see §3 for what we now think
that was telling us); the true-distance heuristic left expansions unchanged on
our instances. One positive-with-caveat: congestion-PIBT was 6% cheaper but
identical for every α in [0.1, 1.0], because on a unit-cost grid a congestion
penalty can only break ties — which §3's preference-construction work addresses
properly, by changing the ordering rather than the cost.

**The learned layer** (§7.6 of the first edition) is summarised in §5. One result
from it is worth repeating here because it is about evaluation rather than
about MAPF: the same trained weights score 45% or 100% depending only on whether
actions are taken by argmax or sampled. The failures split 70/30 between
collision-free period-2 orbits, where the agents never touch and the conflict
rules are never invoked, and period-1 freezes that are genuine livelock. An
earlier draft asserted only the second mechanism; measuring all 80 instances
showed it accounts for under a third of the failures. Reporting a single
instance would have supported either story confidently.

## 8. Open problems, revised

The first edition's list was about making solvers better. This one is not.

1. **Throughput, not cost.** Until this repository has a lifelong mode it is
   measuring a quantity the field has moved past. The RL environment already
   re-draws instances on reset, so this is a wrapper, not a rewrite — and it is
   the single highest-value thing on this list.
2. **A guidance-graph optimiser.** `ExplicitGraph` already carries the weights;
   the optimiser is missing. Orthogonal to every solver already implemented, so
   the gains would compose with all of them.
3. **A learned heuristic inside LaCAM**, rather than a learned policy beside it.
   §5 is unanimous that this is the direction that works, and our own numbers
   agree from the negative side.
4. **Execution, not planning.** Every method here assumes plans execute exactly.
   Kinematics, delays and robot failure are where deployed systems actually
   lose, and arXiv:2404.16162 is right that they are not details.
5. **An evaluation protocol for learned policies.** Our greedy-versus-sampled
   result suggests the field's success rates may not be comparable across papers
   unless action selection is stated, and it frequently is not.

## 9. Reproducing this

Everything marked **[impl]**:

```bash
python scripts/run_experiments.py      # the experimental section
python scripts/train_rl.py             # train IPPO/MAPPO, benchmark vs CBS
python scripts/generate_gallery.py     # every figure in .docs/assets
```

Measurements land in `.docs/assets/experiments.{json,csv}` and
`.docs/assets/rl-benchmark.json`. The interactive playground
([`.docs/index.html`](index.html)) runs the same solvers in a browser under
Pyodide.

Nothing in §2, §4, §5 or §6 is implemented here. Where this edition cites a
result we did not run, it says so, and `research-notes.md` records how far each
citation was actually read.
