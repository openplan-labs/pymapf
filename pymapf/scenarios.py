"""Reproducible MAPF scenarios: named maps plus agent placements.

Benchmarks and demos need *the same instance every time*, which the legacy
``World`` (random walls on construction, no seed) could not provide. Everything
here is deterministic given a ``seed`` and depends only on the standard library,
so the exact same scenarios run in a notebook, in CI, and in the browser under
Pyodide.

Maps follow the shapes used in the MAPF literature: open rooms, random
obstacles, warehouse aisles, mazes, and a bottleneck corridor that forces
head-on interaction (the case where prioritized planning visibly gives up
optimality and CBS does not).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

from .core.grid import Cell, GridMap
from .core.solver import Agent, MAPFProblem

__all__ = [
    "Scenario",
    "SCENARIO_BUILDERS",
    "available_scenarios",
    "build_scenario",
    "empty_room",
    "random_obstacles",
    "warehouse",
    "maze",
    "bottleneck",
    "corner_swap",
    "from_ascii",
    "to_ascii",
]


@dataclass(frozen=True)
class Scenario:
    """A named, fully specified MAPF instance."""

    name: str
    grid: GridMap
    agents: List[Agent]
    allow_diagonals: bool = False
    description: str = ""
    meta: Dict[str, object] = field(default_factory=dict)

    def to_problem(self) -> MAPFProblem:
        return MAPFProblem(
            grid=self.grid,
            agents=list(self.agents),
            allow_diagonals=self.allow_diagonals,
        )

    @property
    def n_agents(self) -> int:
        return len(self.agents)

    def __repr__(self) -> str:
        return "Scenario(%r, %dx%d, agents=%d)" % (
            self.name,
            self.grid.height,
            self.grid.width,
            len(self.agents),
        )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _free_cells(occupancy: Sequence[Sequence[int]]) -> List[Cell]:
    return [
        (r, c)
        for r, row in enumerate(occupancy)
        for c, value in enumerate(row)
        if not value
    ]


def _reachable(occupancy: Sequence[Sequence[int]], source: Cell) -> set:
    """Flood-fill the free component containing ``source`` (4-connected)."""
    height, width = len(occupancy), len(occupancy[0])
    seen = {source}
    stack = [source]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            n = (r + dr, c + dc)
            if (
                0 <= n[0] < height
                and 0 <= n[1] < width
                and not occupancy[n[0]][n[1]]
                and n not in seen
            ):
                seen.add(n)
                stack.append(n)
    return seen


def _largest_component(occupancy: Sequence[Sequence[int]]) -> List[Cell]:
    """All cells of the biggest connected free region, sorted for determinism."""
    remaining = set(_free_cells(occupancy))
    best: set = set()
    while remaining:
        seed = min(remaining)
        component = _reachable(occupancy, seed)
        remaining -= component
        if len(component) > len(best):
            best = component
    return sorted(best)


def _sample_agents(
    occupancy: Sequence[Sequence[int]],
    n_agents: int,
    rng: random.Random,
    min_separation: int = 3,
) -> List[Agent]:
    """Place ``n_agents`` start/goal pairs inside one connected region.

    Starts and goals are drawn without replacement so the instance is always
    well formed, and each pair is re-drawn a few times to land at least
    ``min_separation`` apart -- a scenario where everyone starts on their goal
    makes for a poor demo.
    """
    cells = _largest_component(occupancy)
    if len(cells) < 2 * n_agents:
        raise ValueError(
            "map has %d reachable cells, need >= %d for %d agents"
            % (len(cells), 2 * n_agents, n_agents)
        )

    pool = list(cells)
    rng.shuffle(pool)
    starts = pool[:n_agents]
    goal_pool = pool[n_agents:]

    agents = []
    for i, start in enumerate(starts):
        best_index = 0
        for index, candidate in enumerate(goal_pool):
            distance = abs(candidate[0] - start[0]) + abs(candidate[1] - start[1])
            if distance >= min_separation:
                best_index = index
                break
            if distance > (
                abs(goal_pool[best_index][0] - start[0])
                + abs(goal_pool[best_index][1] - start[1])
            ):
                best_index = index
        goal = goal_pool.pop(best_index)
        agents.append(Agent(_agent_name(i), start, goal))
    return agents


_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _agent_name(index: int) -> str:
    """``A``..``Z`` then ``A2``, ``B2``... so names stay short and readable."""
    letter = _ALPHABET[index % len(_ALPHABET)]
    cycle = index // len(_ALPHABET)
    return letter if cycle == 0 else "%s%d" % (letter, cycle + 1)


# --------------------------------------------------------------------------
# map builders
# --------------------------------------------------------------------------


def empty_room(height: int = 12, width: int = 12, n_agents: int = 4, seed: int = 0):
    """A bordered empty room -- the baseline instance."""
    occupancy = [
        [
            1 if (r in (0, height - 1) or c in (0, width - 1)) else 0
            for c in range(width)
        ]
        for r in range(height)
    ]
    rng = random.Random(seed)
    return Scenario(
        name="empty_room",
        grid=GridMap(occupancy),
        agents=_sample_agents(occupancy, n_agents, rng),
        description="Open %dx%d room; interaction comes only from the agents."
        % (height, width),
        meta={"seed": seed},
    )


def random_obstacles(
    height: int = 16,
    width: int = 16,
    n_agents: int = 6,
    density: float = 0.18,
    seed: int = 0,
):
    """Uniform random walls at ``density``, then agents in the largest region."""
    if not 0.0 <= density < 0.9:
        raise ValueError("density must be in [0, 0.9)")
    rng = random.Random(seed)
    occupancy = [
        [
            (
                1
                if (r in (0, height - 1) or c in (0, width - 1))
                else int(rng.random() < density)
            )
            for c in range(width)
        ]
        for r in range(height)
    ]
    return Scenario(
        name="random_obstacles",
        grid=GridMap(occupancy),
        agents=_sample_agents(occupancy, n_agents, rng),
        description="%dx%d grid, %.0f%% random obstacles." % (height, width, 100 * density),
        meta={"seed": seed, "density": density},
    )


def warehouse(
    shelf_rows: int = 3,
    shelf_cols: int = 4,
    shelf_height: int = 2,
    shelf_width: int = 3,
    n_agents: int = 8,
    seed: int = 0,
):
    """Regular shelf blocks separated by single-cell aisles.

    The structure of a fulfilment-centre map: plenty of free space, but every
    route between two aisles is a corridor, so agents meet head-on constantly.
    """
    height = 2 + shelf_rows * (shelf_height + 1) + 1
    width = 2 + shelf_cols * (shelf_width + 1) + 1
    occupancy = [[0] * width for _ in range(height)]
    for c in range(width):
        occupancy[0][c] = occupancy[height - 1][c] = 1
    for r in range(height):
        occupancy[r][0] = occupancy[r][width - 1] = 1

    for br in range(shelf_rows):
        for bc in range(shelf_cols):
            top = 2 + br * (shelf_height + 1)
            left = 2 + bc * (shelf_width + 1)
            for r in range(top, min(top + shelf_height, height - 1)):
                for c in range(left, min(left + shelf_width, width - 1)):
                    occupancy[r][c] = 1

    rng = random.Random(seed)
    return Scenario(
        name="warehouse",
        grid=GridMap(occupancy),
        agents=_sample_agents(occupancy, n_agents, rng, min_separation=6),
        description="Warehouse aisles: %d x %d shelf blocks." % (shelf_rows, shelf_cols),
        meta={"seed": seed},
    )


def maze(height: int = 15, width: int = 15, n_agents: int = 4, seed: int = 0):
    """A perfect maze (recursive backtracker) -- one route between any two cells.

    With a unique route, *every* interaction is a hard one: agents cannot route
    around each other, they must wait or back off.
    """
    # Carve on odd coordinates so walls stay one cell thick.
    height = height if height % 2 else height + 1
    width = width if width % 2 else width + 1
    occupancy = [[1] * width for _ in range(height)]
    rng = random.Random(seed)

    start = (1, 1)
    occupancy[1][1] = 0
    stack = [start]
    while stack:
        r, c = stack[-1]
        candidates = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr, nc = r + dr, c + dc
            if 1 <= nr < height - 1 and 1 <= nc < width - 1 and occupancy[nr][nc]:
                candidates.append((nr, nc, dr, dc))
        if not candidates:
            stack.pop()
            continue
        nr, nc, dr, dc = rng.choice(candidates)
        occupancy[r + dr // 2][c + dc // 2] = 0
        occupancy[nr][nc] = 0
        stack.append((nr, nc))

    return Scenario(
        name="maze",
        grid=GridMap(occupancy),
        agents=_sample_agents(occupancy, n_agents, rng, min_separation=8),
        description="Perfect maze %dx%d: a single route between any two cells."
        % (height, width),
        meta={"seed": seed},
    )


def bottleneck(room: int = 5, corridor: int = 5, n_agents: int = 4, seed: int = 0):
    """Two rooms joined by a one-cell corridor, agents swapping sides.

    The textbook instance where priorities are not enough: half the agents must
    yield, and a greedy order can deadlock where CBS finds the optimal
    interleaving.
    """
    height = 2 * room + 1
    width = 2 * room + corridor + 2
    occupancy = [[1] * width for _ in range(height)]
    mid = height // 2

    left_cols = range(1, room + 1)
    right_cols = range(room + corridor + 1, width - 1)
    for r in range(1, height - 1):
        for c in left_cols:
            occupancy[r][c] = 0
        for c in right_cols:
            occupancy[r][c] = 0
    for c in range(room + 1, room + corridor + 1):
        occupancy[mid][c] = 0

    left = sorted(
        (r, c) for r in range(1, height - 1) for c in left_cols
    )
    right = sorted(
        (r, c) for r in range(1, height - 1) for c in right_cols
    )
    rng = random.Random(seed)
    rng.shuffle(left)
    rng.shuffle(right)

    agents = []
    per_side = max(1, n_agents // 2)
    for i in range(per_side):
        agents.append(Agent(_agent_name(len(agents)), left[i], right[i]))
    for i in range(n_agents - per_side):
        agents.append(Agent(_agent_name(len(agents)), right[per_side + i], left[per_side + i]))

    return Scenario(
        name="bottleneck",
        grid=GridMap(occupancy),
        agents=agents,
        description="Two rooms, one %d-cell corridor: every agent must cross it."
        % corridor,
        meta={"seed": seed},
    )


def corner_swap(size: int = 9, n_agents: int = 4, seed: int = 0):
    """Agents on the corners/edges that must all cross the centre.

    Fully symmetric and free of randomness, so it is the clearest visual
    demonstration of conflict resolution.
    """
    occupancy = [
        [1 if (r in (0, size - 1) or c in (0, size - 1)) else 0 for c in range(size)]
        for r in range(size)
    ]
    mid = size // 2
    last = size - 2
    ring = [
        ((1, 1), (last, last)),
        ((last, last), (1, 1)),
        ((1, last), (last, 1)),
        ((last, 1), (1, last)),
        ((mid, 1), (mid, last)),
        ((mid, last), (mid, 1)),
        ((1, mid), (last, mid)),
        ((last, mid), (1, mid)),
    ]
    if n_agents > len(ring):
        raise ValueError("corner_swap supports at most %d agents" % len(ring))
    agents = [
        Agent(_agent_name(i), start, goal) for i, (start, goal) in enumerate(ring[:n_agents])
    ]
    return Scenario(
        name="corner_swap",
        grid=GridMap(occupancy),
        agents=agents,
        description="Symmetric swap: every path crosses the centre of the room.",
        meta={"seed": seed},
    )


# --------------------------------------------------------------------------
# ASCII interchange -- handy for docs, tests and the web playground
# --------------------------------------------------------------------------

WALL_CHARS = "#@T"


def from_ascii(text: str, name: str = "ascii", allow_diagonals: bool = False):
    """Parse an ASCII map into a :class:`Scenario`.

    ``#`` (or ``@``/``T``) is a wall, ``.``/space is free. A lowercase letter
    marks an agent's start and the matching uppercase letter marks its goal::

        ########
        #a....A#
        #..##..#
        #B....b#
        ########
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty ASCII map")
    width = max(len(line) for line in lines)
    lines = [line.ljust(width) for line in lines]

    occupancy = [[0] * width for _ in lines]
    starts: Dict[str, Cell] = {}
    goals: Dict[str, Cell] = {}
    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            if char in WALL_CHARS:
                occupancy[r][c] = 1
            elif char.islower():
                starts[char.upper()] = (r, c)
            elif char.isupper():
                goals[char] = (r, c)

    missing = set(starts) ^ set(goals)
    if missing:
        raise ValueError(
            "every agent needs a start (lowercase) and a goal (uppercase); "
            "unmatched: %s" % ", ".join(sorted(missing))
        )
    agents = [Agent(key, starts[key], goals[key]) for key in sorted(starts)]
    return Scenario(
        name=name,
        grid=GridMap(occupancy),
        agents=agents,
        allow_diagonals=allow_diagonals,
        description="Parsed from an ASCII map.",
    )


def to_ascii(scenario: Scenario) -> str:
    """Render a scenario back to the :func:`from_ascii` format."""
    grid = scenario.grid
    canvas = [
        ["#" if not grid.is_free((r, c)) else "." for c in range(grid.width)]
        for r in range(grid.height)
    ]
    for agent in scenario.agents:
        canvas[agent.start[0]][agent.start[1]] = agent.name[0].lower()
        canvas[agent.goal[0]][agent.goal[1]] = agent.name[0].upper()
    return "\n".join("".join(row) for row in canvas)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

SCENARIO_BUILDERS: Dict[str, Callable[..., Scenario]] = {
    "empty_room": empty_room,
    "random_obstacles": random_obstacles,
    "warehouse": warehouse,
    "maze": maze,
    "bottleneck": bottleneck,
    "corner_swap": corner_swap,
}


def available_scenarios() -> List[str]:
    return sorted(SCENARIO_BUILDERS)


def build_scenario(name: str, **kwargs) -> Scenario:
    """Build a registered scenario by name, forwarding keyword arguments."""
    try:
        builder = SCENARIO_BUILDERS[name]
    except KeyError:
        raise ValueError(
            "Unknown scenario %r. Available: %s"
            % (name, ", ".join(available_scenarios()))
        )
    return builder(**kwargs)
