"""PyMAPF: a toolbox for multi-agent path finding and planning.

The :mod:`pymapf.core` package defines the algorithm-agnostic framework
(``GridMap``, ``MAPFProblem``, ``Solution``, ``MAPFSolver`` and a solver
registry). Concrete centralized algorithms live in :mod:`pymapf.algorithms` and
register themselves by name, so they can be obtained via :func:`get_solver`::

    import pymapf

    grid = pymapf.GridMap([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    problem = pymapf.MAPFProblem(grid, [
        pymapf.Agent("a", (0, 0), (2, 2)),
        pymapf.Agent("b", (2, 0), (0, 2)),
    ])
    solution = pymapf.solve(problem, "cbs")
    print(solution.paths, solution.sum_of_costs)

The reactive/decentralized planners (NMPC, Velocity Obstacles) remain available
under :mod:`pymapf.decentralized`.
"""

from .core import (
    Agent,
    Cell,
    ExplicitGraph,
    Conflict,
    Constraints,
    GridMap,
    MAPFProblem,
    MAPFSolver,
    Observer,
    SearchEvent,
    SearchTrace,
    Solution,
    available_solvers,
    find_first_conflict,
    get_solver,
    register_solver,
)

# Importing the algorithms package registers the built-in solvers by name.
from . import algorithms  # noqa: F401  (side-effect: populates the registry)
from . import scenarios  # noqa: F401
from .algorithms import (
    ConflictBasedSearch,
    LaCAM,
    LargeNeighborhoodSearch,
    PIBT,
    PrioritizedPlanning,
    WeightedCBS,
    astar,
    dijkstra,
    sipp,
    space_time_astar,
)
from .scenarios import Scenario, available_scenarios, build_scenario

__version__ = "0.5.0"


def solve(
    problem: MAPFProblem,
    algorithm: str = "cbs",
    observer: Observer = None,
    **kwargs,
):
    """Solve ``problem`` with a registered algorithm (default: CBS).

    ``observer`` receives :class:`SearchEvent` objects while the search runs
    (see :class:`SearchTrace`); ``kwargs`` are forwarded to the solver
    constructor (e.g. ``heuristic``).
    """
    return get_solver(algorithm, **kwargs).solve(problem, observer=observer)


__all__ = [
    "Agent",
    "Cell",
    "Conflict",
    "Constraints",
    "GridMap",
    "MAPFProblem",
    "MAPFSolver",
    "Observer",
    "Scenario",
    "SearchEvent",
    "SearchTrace",
    "Solution",
    "available_scenarios",
    "build_scenario",
    "ConflictBasedSearch",
    "PrioritizedPlanning",
    "WeightedCBS",
    "PIBT",
    "LaCAM",
    "LargeNeighborhoodSearch",
    "ExplicitGraph",
    "astar",
    "dijkstra",
    "sipp",
    "space_time_astar",
    "available_solvers",
    "find_first_conflict",
    "get_solver",
    "register_solver",
    "solve",
    "__version__",
]
