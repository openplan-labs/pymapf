# References

Every algorithm in PyMAPF, with the work it comes from. The same citations
appear in the docstring of the module that implements each one, so they are
visible from `help()` and from the source.

Where an implementation deviates from its source — a simplification, a missing
refinement, a parameter we chose ourselves — the module docstring says so
explicitly. Those deviations are listed in [Implementation notes](#implementation-notes)
at the end.

## Contents

- [Problem definitions and benchmarks](#problem-definitions-and-benchmarks)
- [Single-agent search](#single-agent-search)
- [Optimal MAPF](#optimal-mapf)
- [Bounded-suboptimal MAPF](#bounded-suboptimal-mapf)
- [Fast suboptimal and anytime MAPF](#fast-suboptimal-and-anytime-mapf)
- [Learning-based MAPF](#learning-based-mapf) *(surveyed, not implemented)*
- [Variants and extensions](#variants-and-extensions) *(surveyed, not implemented)*
- [Decentralized flocking](#decentralized-flocking)
- [Decentralized coverage](#decentralized-coverage)
- [Reactive collision avoidance](#reactive-collision-avoidance)
- [Implementation notes](#implementation-notes)

---

## Problem definitions and benchmarks

| Work | Reference |
|---|---|
| MAPF definitions, variants, benchmark suite | Stern, R.; Sturtevant, N. R.; Felner, A.; Koenig, S.; Ma, H.; Walker, T. T.; Li, J.; Atzmon, D.; Cohen, L.; Kumar, T. K. S.; Boyarski, E.; and Barták, R. 2019. *Multi-Agent Pathfinding: Definitions, Variants, and Benchmarks.* SOCS 2019: 151–158. |
| Grid benchmark maps | Sturtevant, N. R. 2012. *Benchmarks for grid-based pathfinding.* IEEE Transactions on Computational Intelligence and AI in Games 4(2): 144–148. |
| NP-hardness of optimal MAPF | Yu, J.; and LaValle, S. M. 2013. *Structure and intractability of optimal multi-robot path planning on graphs.* AAAI 2013: 1443–1449. |
| Survey (this project's own, being extended) | Lejeune, E.; and Sarkar, S. 2021. *Survey of the Multi-Agent Pathfinding Solutions.* Unpublished. DOI 10.13140/RG.2.2.14030.28486 |
| Recent comprehensive survey | *Where Paths Collide: A Comprehensive Survey of Classic and Learning-Based Multi-Agent Pathfinding.* 2025. arXiv:2505.19219. |

## Single-agent search

Implemented in `pymapf/algorithms/search.py`, `space_time_astar.py`, `sipp.py`.

| Algorithm | Module | Reference |
|---|---|---|
| Dijkstra | `search.dijkstra` | Dijkstra, E. W. 1959. *A note on two problems in connexion with graphs.* Numerische Mathematik 1(1): 269–271. |
| A* | `search.astar` | Hart, P. E.; Nilsson, N. J.; and Raphael, B. 1968. *A formal basis for the heuristic determination of minimum cost paths.* IEEE Transactions on Systems Science and Cybernetics 4(2): 100–107. |
| Weighted A* | `search.weighted_astar` | Pohl, I. 1970. *Heuristic search viewed as path finding in a graph.* Artificial Intelligence 1(3–4): 193–204. |
| Focal search / A*ε | `search.focal_astar` | Pearl, J.; and Kim, J. H. 1982. *Studies in semi-admissible heuristics.* IEEE TPAMI 4(4): 392–399. |
| True-distance heuristic | `core.heuristics.true_distance` | Sturtevant, N. R.; Felner, A.; Barrer, M.; Schaeffer, J.; and Burch, N. 2009. *Memory-based heuristics for explicit state spaces.* IJCAI 2009: 609–614. |
| Space-time A* (reservation table) | `space_time_astar` | Silver, D. 2005. *Cooperative pathfinding.* AIIDE 2005: 117–122. |
| Safe Interval Path Planning | `sipp.sipp` | Phillips, M.; and Likhachev, M. 2011. *SIPP: Safe interval path planning for dynamic environments.* ICRA 2011: 5628–5635. |
| Jump point search | *not implemented* | Harabor, D.; and Grastien, A. 2011. *Online graph pruning for pathfinding on grid maps.* AAAI 2011: 1114–1119. |

## Optimal MAPF

| Algorithm | Module | Reference |
|---|---|---|
| **CBS** | `algorithms.cbs` | Sharon, G.; Stern, R.; Felner, A.; and Sturtevant, N. R. 2015. *Conflict-based search for optimal multi-agent pathfinding.* Artificial Intelligence 219: 40–66. |
| ICBS (conflict prioritisation, meta-agents) | partially, in `cbs` | Boyarski, E.; Felner, A.; Stern, R.; Sharon, G.; Tolpin, D.; Betzalel, O.; and Shimony, S. E. 2015. *ICBS: Improved conflict-based search algorithm for multi-agent pathfinding.* IJCAI 2015: 740–746. |
| Bypassing conflicts | *not implemented* | Boyarski, E.; Felner, A.; Sharon, G.; and Stern, R. 2015. *Don't split, try to work it out: Bypassing conflicts in multi-agent pathfinding.* SOCS 2015: 159–162. |
| CBSH (admissible high-level heuristics) | *not implemented* | Felner, A.; Li, J.; Boyarski, E.; Ma, H.; Cohen, L.; Kumar, T. K. S.; and Koenig, S. 2018. *Adding heuristics to conflict-based search for multi-agent path finding.* ICAPS 2018: 83–87. |
| Disjoint splitting | *not implemented* | Li, J.; Harabor, D.; Stuckey, P. J.; Ma, H.; and Koenig, S. 2019. *Disjoint splitting for multi-agent path finding with conflict-based search.* ICAPS 2019: 279–283. |
| Symmetry reasoning (corridor, rectangle, target) | *not implemented* | Li, J.; Harabor, D.; Stuckey, P. J.; and Koenig, S. 2021. *Pairwise symmetry reasoning for multi-agent path finding search.* Artificial Intelligence 301: 103574. |
| M\* / ODrM\* (subdimensional expansion) | *not implemented* | Wagner, G.; and Choset, H. 2011. *M\*: A complete multirobot path planning algorithm with performance bounds.* IROS 2011: 3260–3267. |
| Branch-and-cut-and-price | *not implemented* | Lam, E.; Le Bodic, P.; Harabor, D.; and Stuckey, P. J. 2022. *Branch-and-cut-and-price for multi-agent path finding.* Computers & Operations Research 144: 105809. |
| Increasing Cost Tree Search | *not implemented* | Sharon, G.; Stern, R.; Goldenberg, M.; and Felner, A. 2013. *The increasing cost tree search for optimal multi-agent pathfinding.* Artificial Intelligence 195: 470–495. |

## Bounded-suboptimal MAPF

| Algorithm | Module | Reference |
|---|---|---|
| **ECBS** (focal high level) | `algorithms.weighted_cbs` | Barer, M.; Sharon, G.; Stern, R.; and Felner, A. 2014. *Suboptimal variants of the conflict-based search algorithm for the multi-agent pathfinding problem.* SOCS 2014: 19–27. |
| EECBS (online-learned inadmissible h) | *not implemented* | Li, J.; Ruml, W.; and Koenig, S. 2021. *EECBS: A bounded-suboptimal search for multi-agent path finding.* AAAI 2021: 12353–12362. |

## Fast suboptimal and anytime MAPF

| Algorithm | Module | Reference |
|---|---|---|
| Prioritized planning | `algorithms.prioritized_planning` | Erdmann, M.; and Lozano-Pérez, T. 1987. *On multiple moving objects.* Algorithmica 2: 477–521. |
| Cooperative A* / WHCA* | `algorithms.prioritized_planning` | Silver, D. 2005. *Cooperative pathfinding.* AIIDE 2005: 117–122. |
| Priority orderings (PBS and friends) | *not implemented* | Ma, H.; Harabor, D.; Stuckey, P. J.; Li, J.; and Koenig, S. 2019. *Searching with consistent prioritization for multi-agent path finding.* AAAI 2019: 7643–7650. |
| **PIBT** | `algorithms.pibt` | Okumura, K.; Machida, M.; Défago, X.; and Tamura, Y. 2022. *Priority inheritance with backtracking for iterative multi-agent path finding.* Artificial Intelligence 310: 103752. (Earlier: IJCAI 2019: 535–542.) |
| PIBT preference construction | related work for `experimental.congestion_pibt` | Okumura, K.; and Nagai, R. 2025. *Lightweight and effective preference construction in PIBT for large-scale multi-agent pathfinding.* SOCS 2025. |
| **LaCAM** | `algorithms.lacam` | Okumura, K. 2023. *LaCAM: Search-based algorithm for quick multi-agent pathfinding.* AAAI 2023, 37(10): 11655–11662. |
| LaCAM\* (eventually optimal) | `algorithms.lacam` (`anytime=True`) | Okumura, K. 2023. *Improving LaCAM for scalable eventually optimal multi-agent pathfinding.* IJCAI 2023: 243–251. |
| LaCAM3 / Engineering LaCAM\* | *not implemented* | Okumura, K. 2024. *Engineering LaCAM\*: Towards real-time, large-scale, and near-optimal multi-agent pathfinding.* AAMAS 2024: 1501–1509. |
| **MAPF-LNS** | `algorithms.lns` | Li, J.; Chen, Z.; Harabor, D.; Stuckey, P. J.; and Koenig, S. 2021. *Anytime multi-agent path finding via large neighborhood search.* IJCAI 2021: 4127–4135. |
| MAPF-LNS2 / SIPPS | partially, in `lns` + `sipp` | Li, J.; Chen, Z.; Harabor, D.; Stuckey, P. J.; and Koenig, S. 2022. *MAPF-LNS2: Fast repairing for multi-agent path finding via large neighborhood search.* AAAI 2022: 10256–10265. |
| Large neighbourhood search (origin) | `algorithms.lns` | Shaw, P. 1998. *Using constraint programming and local search methods to solve vehicle routing problems.* CP 1998: 417–431. |
| Push and Swap / Push and Rotate | *not implemented* | Luna, R.; and Bekris, K. E. 2011. *Push and swap: Fast cooperative path-finding with completeness guarantees.* IJCAI 2011: 294–300. de Wilde, B.; ter Mors, A. W.; and Witteveen, C. 2014. *Push and rotate: A complete multi-agent pathfinding algorithm.* JAIR 51: 443–492. |
| Iterative refinement | inspiration for `experimental.restart_lacam` | Okumura, K.; Tamura, Y.; and Défago, X. 2021. *Iterative refinement for real-time multi-robot path planning.* IROS 2021: 9690–9697. |
| Restart strategies for randomised search | `experimental.restart_lacam` | Luby, M.; Sinclair, A.; and Zuckerman, D. 1993. *Optimal speedup of Las Vegas algorithms.* Information Processing Letters 47(4): 173–180. |

## Learning-based MAPF

Surveyed in `docs/survey.md`; not implemented (the core of this library is
dependency-free by design, and these need a trained model).

| Method | Reference |
|---|---|
| PRIMAL | Sartoretti, G.; Kerr, J.; Shi, Y.; Wagner, G.; Kumar, T. K. S.; Koenig, S.; and Choset, H. 2019. *PRIMAL: Pathfinding via reinforcement and imitation multi-agent learning.* IEEE RA-L 4(3): 2378–2385. |
| PRIMAL2 | Damani, M.; Luo, Z.; Wenzel, E.; and Sartoretti, G. 2021. *PRIMAL2: Pathfinding via reinforcement and imitation multi-agent learning — lifelong.* IEEE RA-L 6(2): 2666–2673. |
| DHC / distributed heuristic communication | Ma, Z.; Luo, Y.; and Ma, H. 2021. *Distributed heuristic multi-agent path finding with communication.* ICRA 2021: 8699–8705. |
| SCRIMP | Wang, Y.; Xiang, B.; Huang, S.; and Sartoretti, G. 2023. *SCRIMP: Scalable communication for reinforcement- and imitation-learning-based multi-agent pathfinding.* IROS 2023. |
| MAPF-GPT | Andreychuk, A.; et al. 2025. *MAPF-GPT: Imitation learning for multi-agent pathfinding at scale.* AAAI 2025. arXiv:2409.00134. |
| LNS2+RL | Yan, Z.; and Wu, C. 2025. *LNS2+RL: Combining multi-agent reinforcement learning with large neighborhood search in multi-agent path finding.* AAAI 2025. |

## Variants and extensions

Surveyed in `docs/survey.md`; not implemented.

| Variant | Reference |
|---|---|
| Lifelong MAPF (RHCR) | Li, J.; Tinka, A.; Kiesel, S.; Durham, J. W.; Kumar, T. K. S.; and Koenig, S. 2021. *Lifelong multi-agent path finding in large-scale warehouses.* AAAI 2021: 11272–11281. |
| Continuous-time MAPF (CCBS) | Andreychuk, A.; Yakovlev, K.; Boyarski, E.; and Stern, R. 2022. *Improving continuous-time conflict based search.* AAAI 2021 / Artificial Intelligence 305: 103662. |
| Anonymous / target assignment (TSWAP) | Okumura, K.; and Défago, X. 2022. *Solving simultaneous target assignment and path planning efficiently with time-independent execution.* ICAPS 2022: 270–278. |
| Robust/k-robust plans | Atzmon, D.; Stern, R.; Felner, A.; Wagner, G.; Barták, R.; and Zhou, N.-F. 2020. *Robust multi-agent path finding and executing.* JAIR 67: 549–579. |
| Execution under uncertainty (ADG) | Hönig, W.; Kiesel, S.; Tinka, A.; Durham, J. W.; and Ayanian, N. 2019. *Persistent and robust execution of MAPF schedules in warehouses.* IEEE RA-L 4(2): 1125–1131. |
| MAPF with kinematic constraints | Hönig, W.; Kumar, T. K. S.; Cohen, L.; Ma, H.; Xu, H.; Ayanian, N.; and Koenig, S. 2016. *Multi-agent path finding with kinematic constraints.* ICAPS 2016: 477–485. |

## Decentralized flocking

Implemented in `pymapf/decentralized/flocking.py`.

| Model | Function | Reference |
|---|---|---|
| Boids | `flocking.boids` | Reynolds, C. W. 1987. *Flocks, herds and schools: A distributed behavioral model.* SIGGRAPH 1987: 25–34. |
| Vicsek model | `flocking.vicsek` | Vicsek, T.; Czirók, A.; Ben-Jacob, E.; Cohen, I.; and Shochet, O. 1995. *Novel type of phase transition in a system of self-driven particles.* Physical Review Letters 75(6): 1226–1229. |
| Cucker–Smale | *surveyed* | Cucker, F.; and Smale, S. 2007. *Emergent behavior in flocks.* IEEE Transactions on Automatic Control 52(5): 852–862. |
| Gradient flocking (α-lattice) | `flocking.olfati_saber` | Olfati-Saber, R. 2006. *Flocking for multi-agent dynamic systems: Algorithms and theory.* IEEE Transactions on Automatic Control 51(3): 401–420. |
| Topological neighbourhoods | `flocking` (`topological_k`) | Ballerini, M.; et al. 2008. *Interaction ruling animal collective behavior depends on topological rather than metric distance.* PNAS 105(4): 1232–1237. |
| Optimised drone flocking | *surveyed* | Vásárhelyi, G.; Virágh, C.; Somorjai, G.; Nepusz, T.; Eiben, A. E.; and Vicsek, T. 2018. *Optimized flocking of autonomous drones in confined environments.* Science Robotics 3(20): eaat3536. |
| **Acceleration-based bird-inspired flocking** | `flocking.acceleration` | Iacone, L.; Lejeune, E.; Manoni, T.; Manfredi, S.; and Albani, D. 2024. *Decentralized acceleration-based bird-inspired flocking.* IEEE, 2024. Autonomous Robotics Research Centre, Technology Innovation Institute. |

## Decentralized coverage

Implemented in `pymapf/decentralized/coverage.py`.

| Method | Function | Reference |
|---|---|---|
| Lloyd's algorithm | `coverage.lloyd_step` | Lloyd, S. P. 1982. *Least squares quantization in PCM.* IEEE Transactions on Information Theory 28(2): 129–137. |
| Voronoi coverage control | `coverage.lloyd_step` | Cortés, J.; Martínez, S.; Karataş, T.; and Bullo, F. 2004. *Coverage control for mobile sensing networks.* IEEE Transactions on Robotics and Automation 20(2): 243–255. |
| Adaptive coverage | *surveyed* | Schwager, M.; Rus, D.; and Slotine, J.-J. 2009. *Decentralized, adaptive coverage control for networked robots.* IJRR 28(3): 357–375. |
| **Limited-range coverage for aerial teams** | `coverage` (`sensing_range`) | Bertoncelli, F.; Belal, M.; Albani, D.; Pratissoli, F.; and Sabattini, L. 2024. *On limited-range coverage control for large-scale teams of aerial drones: Deployment and study.* DARS 2022, Springer Proceedings in Advanced Robotics. |
| **Hemispherical surface coverage** | `coverage.spherical_lloyd_step` | Belal, M.; Manoni, T.; Albani, D.; and Sabattini, L. 2026. *Decentralized multi-robot coverage of hemispherical surfaces via fortune-based partitioning.* ANTS 2026. |

## Reactive collision avoidance

Implemented in `pymapf/decentralized/` (pre-existing modules).

| Method | Module | Reference |
|---|---|---|
| Velocity obstacles | `decentralized.velocity_obstacle` | Fiorini, P.; and Shiller, Z. 1998. *Motion planning in dynamic environments using velocity obstacles.* IJRR 17(7): 760–772. |
| Reciprocal velocity obstacles | *related* | van den Berg, J.; Lin, M.; and Manocha, D. 2008. *Reciprocal velocity obstacles for real-time multi-agent navigation.* ICRA 2008: 1928–1935. |
| ORCA | *related* | van den Berg, J.; Guy, S. J.; Lin, M.; and Manocha, D. 2011. *Reciprocal n-body collision avoidance.* Robotics Research (ISRR 2009), Springer: 3–19. |
| Nonlinear MPC for multi-robot motion | `decentralized.nmpc` | Kamel, M.; Alonso-Mora, J.; Siegwart, R.; and Nieto, J. 2017. *Robust collision avoidance for multiple micro aerial vehicles using nonlinear model predictive control.* IROS 2017: 236–243. |

---

## Implementation notes

Where this library departs from the work it cites — stated here so nobody has
to read the source to find out.

**CBS** expands best-first on `(cost, conflict count)` rather than cost alone.
The tie-break is standard practice (it is the "prefer fewer conflicts" idea
behind ICBS) but the full ICBS conflict classification — cardinal, semi-cardinal,
non-cardinal — is *not* implemented, and neither are bypass, disjoint splitting
or symmetry reasoning. On corridor-heavy maps this CBS is therefore markedly
slower than a modern optimal solver.

**Weighted CBS** implements the ECBS *high level* (focal search on the
constraint tree). It does not implement ECBS's bounded-suboptimal *low level*,
so the solver's effective focal set is smaller than the reference's; the
suboptimality bound still holds.

**LaCAM** implements the AAAI 2023 search: configuration space, lazy constraint
generation, PIBT as the successor generator. `anytime=True` continues the
search after the first solution and relaxes g-values over the explored graph,
following the idea of LaCAM\* (IJCAI 2023) — but it does not implement the full
cost-propagation scheme, and it should not be read as carrying LaCAM\*'s
eventual-optimality guarantee. The measured effect of that continuation is
reported in `docs/survey.md` § Experiments. None of the LaCAM3 engineering
(swap operations, monte-carlo generation, multi-threading) is present.

**MAPF-LNS** uses the destroy/repair loop and adaptive operator weights of the
IJCAI 2021 paper, with SIPP as the repair search. The soft-constraint SIPPS of
MAPF-LNS2 — which lets repair produce a *colliding* path scored by collision
count — is not implemented; repair here either finds a conflict-free path or
fails.

**Acceleration-based flocking** follows the model family described by Iacone et
al. (2024): self-propulsion, drag, pairwise potential, velocity alignment. The
paper's own parameter values were not accessible when this was written, so the
defaults in `FlockParams` were tuned here. Results from this implementation are
not a reproduction of that paper's results.

**Olfati-Saber flocking** implements the paper's action function φ_α, bump
function ρ_h and both navigational-feedback terms. One deviation is deliberate:
the interaction range is clamped to 1.2 × the reference distance, per the
paper's α-lattice construction. Left at this library's default sensing range
(2 × reference distance) the lattice collapses — measured, and documented in
`docs/survey.md` § Experiments.

**Coverage** uses grid quadrature rather than an exact Voronoi construction, so
cell boundaries are resolution-limited. Fixed points agree with the exact
algorithm as the sampling gets finer; the hemispherical variant is the
discrete counterpart of the fortune-based partitioning of Belal et al. (2026),
not an implementation of it.
