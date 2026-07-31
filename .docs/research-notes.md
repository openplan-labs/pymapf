# Research notes: what changed in MAPF, 2024–2026

Working notes behind [`survey-v2.md`](survey-v2.md). This file is the raw
material — an annotated scan of the literature that appeared after the first
2026 update was written, organised by the theme it belongs to rather than by
date.

**On verification.** Everything below was located by literature search and read
at abstract level; where a claim comes from an abstract rather than a full text,
it says so. Nothing here has been reproduced in this repository unless it is
marked **[impl]**, and the survey keeps that distinction. Several identifiers
are recent enough that they are preprints rather than proceedings versions.

---

## 1. The centre of gravity moved to *lifelong* MAPF

The single biggest shift. One-shot MAPF — fixed starts, fixed goals, plan once —
is now largely a benchmark rather than the application. The deployed problem is
**lifelong MAPF (LMAPF)**: an agent that reaches its goal is immediately given
another, so the system never stops and throughput, not sum-of-costs, is the
objective.

| Work | Identifier | Note |
|---|---|---|
| Scaling LMAPF to more realistic settings: research challenges and opportunities | arXiv:2404.16162 | Position paper. Names the gap between benchmark LMAPF and warehouse reality — kinematics, task assignment, failures. |
| Deploying Ten Thousand Robots: Scalable Imitation Learning for LMAPF (SILLM) | arXiv:2410.21415, ICRA 2025 | The scale result. Communication module, single-step collision resolution, global guidance; six maps, up to 10,000 agents. |
| Learning-guided prioritized planning for LMAPF in warehouse automation | arXiv:2603.23838 | Learned priorities inside prioritized planning, warehouse-specific. |
| Lifelong LaCAM with local guidance | arXiv:2605.16855 | Carries the local-guidance idea (§2) into the lifelong setting. |

**Why it matters for a survey.** Sum-of-costs optimality — the thing CBS gives
you and the thing our §7 measures against — is not the objective in LMAPF at
all. A survey that treats optimal MAPF as the centre and everything else as an
approximation of it now has the emphasis backwards.

## 2. LaCAM became an engineering programme

The first update covered LaCAM as an algorithm. Since then it has become a
*line*, and most of the progress is engineering rather than new theory.

| Work | Identifier | Note |
|---|---|---|
| Engineering LaCAM\*: towards real-time, large-scale, near-optimal MAPF | AAMAS 2024, DOI 10.5555/3635637.3663010 | "LaCAM3". The reference implementation everything else now compares against. |
| Local guidance for configuration-based MAPF | arXiv:2510.19072 | Guidance applied *within* configuration generation rather than as a global heuristic. |
| A lightweight traffic map for efficient anytime LaCAM\* | arXiv:2603.07891 | Captures congestion *during* the search and feeds it back; faster convergence and better final quality. |

The traffic-map result is directly relevant to us: our §7.4 experiment found
that LaCAM's in-search anytime relaxation won 0 of 28 paired instances and had
to be replaced with randomised restarts. A congestion signal accumulated during
search is exactly the missing ingredient that negative result was pointing at.

## 3. PIBT got a second life

PIBT is a one-step rule, which made it look like a solved component. Two 2025
papers reopened it.

| Work | Identifier | Note |
|---|---|---|
| Anytime single-step MAPF planning with Anytime PIBT | arXiv:2504.07841 | Makes the *single step* anytime — spend more time on one timestep and get a better joint action. |
| Lightweight and effective preference construction in PIBT | arXiv:2505.12623 | How an agent orders its candidate moves; cheap changes, large effect at scale. |

Also directly relevant to our own results. Our §7.4 congestion-PIBT variant was
6% cheaper but *identical* for every α in [0.1, 1.0], because on a unit-cost
grid a congestion penalty can only break ties. Preference construction is the
same lever pulled properly: change the ordering, not the cost.

## 4. A third lever: shape the graph, not the planner

The genuinely new idea, and the one the first update missed entirely. Instead of
improving the planner, **optimise the environment the planner runs on** — edge
weights and directions that discourage head-on flow — and then run an ordinary
planner on the modified graph.

| Work | Identifier | Note |
|---|---|---|
| Guidance graph optimization for lifelong MAPF | arXiv:2402.01446 | The formulation: learn edge weights that raise throughput. |
| Optimization of edge directions and weights for mixed guidance graphs in LMAPF | arXiv:2602.23468 | Adds *direction* to the search space — partial one-way systems. |
| Multi-robot coordination and layout design for automated warehousing | arXiv:2305.06436 | The same idea one level up: co-design the layout with the coordination. |

This is a different axis from everything in the first update, all of which
improved solvers. It is also the axis with the clearest industrial reading: a
warehouse operator can repaint floor markings far more easily than they can swap
a planner.

## 5. Learning stopped trying to replace search

The 2021 survey covered learned MAPF as an alternative to search (PRIMAL and
successors). The current work embeds learning *inside* search as a component,
which is a much better fit for what each is good at.

| Work | Identifier | Note |
|---|---|---|
| Graph attention-guided search for dense MAPF (LaGAT / MAGAT+) | arXiv:2510.17382, AAAI | A learned neural heuristic used as LaCAM's guidance. Beats both pure search and pure learning in dense scenarios. |
| MAPF-World: action world model for MAPF | arXiv:2508.12087 | A learned world model of the joint transition, used for planning. |
| SILLM | arXiv:2410.21415 | Imitation of a search-based expert, with search-based collision resolution retained at execution. |
| Transformer heuristics for MAPF-LNS | (see LNS line below) | Learned neighbourhood selection rather than learned policies. |
| Anytime MAPF using operation parallelism in LNS | arXiv:2402.01961 | Not learning, but the same "improve the loop, keep the solver" instinct. |

**The pattern:** in every successful case the learned component supplies a
*heuristic, a priority, or a neighbourhood* — and a sound search retains
responsibility for feasibility. None of the competitive systems lets a network
emit the final joint action unchecked.

Our own RL results (§7.6 of the survey) sit exactly here and support it from the
negative side: a policy that emits the joint action directly is 45% successful
under argmax and pays 2.94× optimal when sampled, while the search-based
baselines solve 100% at 1.00–1.07×. The learned policy is not competitive as a
solver; it would be a plausible *heuristic*.

## 6. The competition became a forcing function

The **League of Robot Runners** (ICAPS) has done for LMAPF roughly what the
DARPA challenges did for driving: fixed the evaluation, made throughput the
score, and forced entrants to handle a realistic execution model. The winning
2023 entry (team Pikachu) is public at
[DiligentPanda/MAPF-LRR2023](https://github.com/DiligentPanda/MAPF-LRR2023).

Competition entries are worth reading precisely because they are not papers:
they show which combination of the above actually survives contact with a clock.

---

## What this repository does and does not cover

Implemented and measured here **[impl]**: CBS, weighted CBS/ECBS-style focal
search, PIBT, LaCAM (with the anytime caveat in §7.4), MAPF-LNS, prioritized
planning, SIPP, space-time A\*, plus the decentralized swarm layer and the RL
layer.

Surveyed but **not** implemented: everything in §1, §4, §5 and §6 above, and the
LaCAM3 engineering line in §2. The gap that would be most worth closing, in
order:

1. **A lifelong wrapper.** Our environment already re-assigns goals on reset; an
   LMAPF mode is a small change and would let us measure throughput, which is
   what the field now optimises.
2. **A guidance graph.** `ExplicitGraph` already supports arbitrary weights, so
   the representation is there — the missing part is the optimiser.
3. **A learned heuristic inside LaCAM**, rather than a learned policy beside it.
   This is the direction §5 says works, and our own §7.6 numbers agree.
