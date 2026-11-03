"""Core framework types shared by every centralized MAPF algorithm.

This module defines the small, stable vocabulary that makes PyMAPF usable as a
framework: a :class:`MAPFProblem` (the input), a :class:`Solution` (the output),
:class:`Constraints` (used by constraint-based low-level search), conflict
detection utilities, an abstract :class:`MAPFSolver`, and a name-based solver
registry so new algorithms can be discovered and swapped without touching call
sites.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from .grid import Cell, GridMap
from .trace import Observer


@dataclass(frozen=True)
class Agent:
    """A planning agent: a unique ``name`` with ``start`` and ``goal`` cells.

    Cells given as lists (``[0, 1]``, as JSON hands them over) are stored as
    tuples: every solver hashes cells, and a list cannot be hashed.
    """

    name: str
    start: Cell
    goal: Cell

    def __post_init__(self):
        for field_name in ("start", "goal"):
            value = getattr(self, field_name)
            if isinstance(value, list):
                object.__setattr__(self, field_name, tuple(value))


@dataclass
class MAPFProblem:
    """A multi-agent path finding instance."""

    grid: GridMap
    agents: List[Agent]
    allow_diagonals: bool = False

    def __post_init__(self):
        self.agents = list(self.agents)
        names = [a.name for a in self.agents]
        if len(names) != len(set(names)):
            raise ValueError("agent names must be unique")
        # Two agents cannot occupy one cell, at t = 0 or parked at the end, so
        # a shared start or goal makes the instance unsolvable. Say so here
        # rather than let each solver fail (or worse, "succeed") in its own way.
        for attribute in ("start", "goal"):
            claimed = {}
            for a in self.agents:
                cell = getattr(a, attribute)
                if cell in claimed:
                    raise ValueError(
                        "agents %r and %r share the %s %s"
                        % (claimed[cell], a.name, attribute, cell)
                    )
                claimed[cell] = a.name
        for a in self.agents:
            if not self.grid.is_free(a.start):
                raise ValueError(
                    "agent %r start %s is blocked/out of bounds" % (a.name, a.start)
                )
            if not self.grid.is_free(a.goal):
                raise ValueError(
                    "agent %r goal %s is blocked/out of bounds" % (a.name, a.goal)
                )


class Constraints:
    """Vertex and edge constraints for a single agent's low-level search.

    * A *vertex* constraint ``(cell, t)`` forbids occupying ``cell`` at time
      ``t``.
    * An *edge* constraint ``(u, v, t)`` forbids traversing from ``u`` to ``v``
      and arriving at ``v`` at time ``t`` (used to prevent agents swapping).
    """

    __slots__ = ("vertex", "edge")

    def __init__(self):
        self.vertex = set()  # {(cell, t)}
        self.edge = set()  # {(u, v, t)}

    def copy(self) -> "Constraints":
        c = Constraints()
        c.vertex = set(self.vertex)
        c.edge = set(self.edge)
        return c

    def add_vertex(self, cell: Cell, t: int) -> None:
        self.vertex.add((cell, t))

    def add_edge(self, u: Cell, v: Cell, t: int) -> None:
        self.edge.add((u, v, t))

    def blocks_vertex(self, cell: Cell, t: int) -> bool:
        return (cell, t) in self.vertex

    def blocks_edge(self, u: Cell, v: Cell, t: int) -> bool:
        return (u, v, t) in self.edge

    def last_vertex_time(self, cell: Cell) -> int:
        """Latest time a vertex constraint applies to ``cell`` (``-1`` if none).

        The low-level search uses this to know it may only "settle" on the goal
        once no further constraint can push it off.
        """
        times = [t for (c, t) in self.vertex if c == cell]
        return max(times) if times else -1


# A conflict between two agents in a candidate solution.
@dataclass(frozen=True)
class Conflict:
    kind: str  # "vertex" or "edge"
    a: str
    b: str
    t: int
    cell_a: Cell
    cell_b: Cell  # equals cell_a for vertex conflicts


@dataclass
class Solution:
    """Result of a solve: a path per agent plus cost metrics.

    ``paths[name]`` is a list of cells where ``paths[name][t]`` is the agent's
    location at timestep ``t`` (index 0 is the start).
    """

    paths: Dict[str, List[Cell]]
    algorithm: str = ""
    expansions: int = 0
    runtime: float = 0.0  # wall-clock seconds spent in ``solve``

    @property
    def makespan(self) -> int:
        """Time at which the last agent reaches its goal."""
        return max((len(p) - 1 for p in self.paths.values()), default=0)

    @property
    def sum_of_costs(self) -> int:
        """Sum over agents of ``len(path) - 1``.

        That is the standard sum of costs -- each agent's arrival time at its
        goal for good -- because every built-in solver ends a path the moment
        its agent settles (LaCAM and PIBT trim the trailing waits on the goal).
        A hand-built path padded with waits on its goal is charged for them.
        """
        return sum(len(p) - 1 for p in self.paths.values())

    def first_conflict(self) -> Optional[Conflict]:
        return find_first_conflict(self.paths)

    def is_valid(self, problem: Optional["MAPFProblem"] = None) -> bool:
        """Whether this is a valid plan.

        Without ``problem`` this only checks for collisions *between* agents
        (vertex and swap conflicts), as it always has. With ``problem`` it
        checks the whole definition; see :meth:`violations`.
        """
        if problem is None:
            return self.first_conflict() is None
        return not self.violations(problem)

    def violations(self, problem: "MAPFProblem") -> List[str]:
        """Everything that makes this an invalid plan for ``problem``.

        A plan can be free of collisions and still walk through walls, jump
        across the map, or never reach a goal; a solver that returned every
        agent's start cell would pass a conflict check. This checks the whole
        definition: exactly one path per agent, starting on its start and
        ending on its goal, every step a wait or a move to a neighbour on free
        cells (diagonals only when the problem allows them), and no vertex or
        swap conflict. Returns readable messages; empty means valid.
        """
        grid, diagonals = problem.grid, problem.allow_diagonals
        found: List[str] = []
        names = {a.name for a in problem.agents}
        for name in self.paths:
            if name not in names:
                found.append("path for unknown agent %r" % (name,))
        for agent in problem.agents:
            path = self.paths.get(agent.name)
            if not path:
                found.append("agent %r has no path" % agent.name)
                continue
            if path[0] != agent.start:
                found.append(
                    "agent %r starts at %s, not its start %s"
                    % (agent.name, path[0], agent.start)
                )
            if path[-1] != agent.goal:
                found.append(
                    "agent %r ends at %s, not its goal %s"
                    % (agent.name, path[-1], agent.goal)
                )
            for t, cell in enumerate(path):
                if not grid.is_free(cell):
                    found.append(
                        "agent %r is on blocked or missing cell %s at t=%d"
                        % (agent.name, cell, t)
                    )
                    break
                if t and cell != path[t - 1]:
                    if cell not in grid.neighbors(path[t - 1], diagonals):
                        found.append(
                            "agent %r jumps from %s to %s at t=%d"
                            % (agent.name, path[t - 1], cell, t)
                        )
                        break
        conflict = find_first_conflict({n: p for n, p in self.paths.items() if p})
        if conflict is not None:
            found.append(
                "%s conflict between %r and %r at t=%d on %s"
                % (conflict.kind, conflict.a, conflict.b, conflict.t, conflict.cell_a)
            )
        return found

    def position_at(self, agent: str, t: int) -> Cell:
        """Where ``agent`` is at time ``t`` (it parks on its goal afterwards)."""
        return _cell_at(self.paths[agent], t)

    def congestion(self) -> Dict[Cell, int]:
        """How many agent-timesteps each cell absorbs -- the traffic hot spots."""
        counts: Dict[Cell, int] = {}
        for path in self.paths.values():
            for cell in path:
                counts[cell] = counts.get(cell, 0) + 1
        return counts

    def as_dict(self) -> Dict:
        """JSON-serialisable view (paths as lists), used by the web playground."""
        return {
            "algorithm": self.algorithm,
            "expansions": self.expansions,
            "runtime": self.runtime,
            "makespan": self.makespan,
            "sum_of_costs": self.sum_of_costs,
            "paths": {
                name: [list(cell) for cell in path] for name, path in self.paths.items()
            },
        }


def _cell_at(path: List[Cell], t: int) -> Cell:
    """Position of an agent at time ``t`` (it waits on its goal after arrival)."""
    return path[t] if t < len(path) else path[-1]


def find_first_conflict(paths: Dict[str, List[Cell]]) -> Optional[Conflict]:
    """Return the earliest vertex or edge conflict between any pair of agents.

    Paths are implicitly padded: once an agent reaches its goal it is assumed to
    stay there, so a shorter path still occupies its goal at later timesteps.
    """
    names = list(paths)
    horizon = max((len(p) for p in paths.values()), default=0)
    for t in range(horizon):
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                pa, pb = paths[a], paths[b]
                a_now, b_now = _cell_at(pa, t), _cell_at(pb, t)
                # Vertex conflict: both agents occupy the same cell.
                if a_now == b_now:
                    return Conflict("vertex", a, b, t, a_now, b_now)
                # Edge conflict: agents swap cells between t and t+1.
                a_next, b_next = _cell_at(pa, t + 1), _cell_at(pb, t + 1)
                if a_now == b_next and b_now == a_next:
                    return Conflict("edge", a, b, t + 1, a_next, b_next)
    return None


def count_conflicts(paths: Dict[str, List[Cell]]) -> int:
    """Total number of pairwise conflicts in a joint plan.

    Where :func:`find_first_conflict` answers "is this plan valid?",
    this answers "how far from valid is it?" -- the quantity CBS uses to
    tie-break between nodes of equal cost.
    """
    names = list(paths)
    horizon = max((len(p) for p in paths.values()), default=0)
    total = 0
    for t in range(horizon):
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pa, pb = paths[names[i]], paths[names[j]]
                a_now, b_now = _cell_at(pa, t), _cell_at(pb, t)
                if a_now == b_now:
                    total += 1
                    continue
                if a_now == _cell_at(pb, t + 1) and b_now == _cell_at(pa, t + 1):
                    total += 1
    return total


class MAPFSolver(ABC):
    """Abstract base class for centralized MAPF algorithms.

    Subclasses implement :meth:`solve`; registering them with
    :func:`register_solver` makes them retrievable by name via
    :func:`get_solver`.
    """

    name = "abstract"

    @abstractmethod
    def solve(
        self, problem: MAPFProblem, observer: Optional[Observer] = None
    ) -> Optional[Solution]:
        """Return a :class:`Solution` or ``None`` if the instance is unsolved.

        ``observer`` is an optional callable receiving
        :class:`~pymapf.core.trace.SearchEvent` objects as the search runs; it
        is what the live views in :mod:`pymapf.viz` subscribe to.
        """
        raise NotImplementedError


_REGISTRY: Dict[str, Type[MAPFSolver]] = {}


def register_solver(name: str, replace: bool = False):
    """Class decorator that registers a solver under ``name``.

    Registering a *different* class under a taken name raises, unless
    ``replace=True``: silently shadowing a built-in is a bug, not a feature.
    Re-registering the same class -- same module and qualified name, which is
    what re-running a notebook cell or reloading a module does -- simply
    installs the new definition.
    """

    def decorator(cls: Type[MAPFSolver]) -> Type[MAPFSolver]:
        key = name.lower()
        existing = _REGISTRY.get(key)
        if existing is not None and not replace:
            same = (existing.__module__, existing.__qualname__) == (
                cls.__module__,
                cls.__qualname__,
            )
            if not same:
                raise ValueError(
                    "solver %r is already registered by %s.%s; pass "
                    "replace=True to override it"
                    % (name, existing.__module__, existing.__qualname__)
                )
        cls.name = key
        _REGISTRY[key] = cls
        return cls

    return decorator


def available_solvers() -> List[str]:
    return sorted(_REGISTRY)


def get_solver(name: str, **kwargs) -> MAPFSolver:
    """Instantiate a registered solver by name, forwarding ``kwargs``."""
    key = name.lower()
    try:
        cls = _REGISTRY[key]
    except KeyError as error:
        raise ValueError(
            "Unknown solver %r. Available: %s" % (name, ", ".join(available_solvers()))
        ) from error
    return cls(**kwargs)
