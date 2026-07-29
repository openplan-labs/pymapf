"""Arbitrary graphs, so the solvers are not tied to grids.

Grid maps are the MAPF benchmark convention, but the algorithms in this library
only ever ask a map two things: *is this vertex free?* and *what is adjacent to
it?*. :class:`ExplicitGraph` answers exactly those questions for any graph --
a roadmap, a warehouse topology, a metro network, a PRM sampled from a
continuous space -- and is deliberately duck-type compatible with
:class:`~pymapf.core.grid.GridMap`, so every solver, the space-time A* low
level and the benchmark harness accept one with no changes.

    graph = ExplicitGraph({"a": ["b", "c"], "b": ["a", "d"], "c": ["d"], "d": []})
    problem = MAPFProblem(graph, [Agent("r1", "a", "d")])
    solution = pymapf.solve(problem, "cbs", heuristic=zero_heuristic)

Because a general graph has no geometry, the usual grid heuristics
(``manhattan`` and friends) do not apply; use
:func:`~pymapf.core.heuristics.true_distance` (a backward Dijkstra, exact and
admissible on any graph) or supply coordinates via ``positions``.
"""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

Node = Hashable


class ExplicitGraph:
    """A graph given by its adjacency, usable anywhere a ``GridMap`` is.

    Args:
        adjacency: ``{node: [neighbour, ...]}``. Edges are treated as directed
            exactly as given; call :meth:`undirected` to symmetrise them.
        positions: optional ``{node: (y, x)}`` layout, used by the geometric
            heuristics and by :mod:`pymapf.viz` when drawing the graph.
        blocked: nodes that exist in the adjacency but are not traversable
            (a closed aisle, a reserved dock).
    """

    def __init__(
        self,
        adjacency: Mapping[Node, Iterable[Node]],
        positions: Optional[Mapping[Node, Tuple[float, float]]] = None,
        blocked: Iterable[Node] = (),
    ):
        self._adjacency: Dict[Node, List[Node]] = {
            node: list(neighbours) for node, neighbours in adjacency.items()
        }
        # Nodes that only ever appear as a neighbour still exist.
        for neighbours in list(self._adjacency.values()):
            for neighbour in neighbours:
                self._adjacency.setdefault(neighbour, [])

        self._blocked = frozenset(blocked)
        self.positions = dict(positions or {})
        if not self._adjacency:
            raise ValueError("graph must have at least one node")

    # -- construction -------------------------------------------------------
    @classmethod
    def undirected(
        cls,
        edges: Iterable[Tuple[Node, Node]],
        positions: Optional[Mapping[Node, Tuple[float, float]]] = None,
        blocked: Iterable[Node] = (),
    ) -> "ExplicitGraph":
        """Build from an edge list, adding both directions of every edge."""
        adjacency: Dict[Node, List[Node]] = {}
        for u, v in edges:
            adjacency.setdefault(u, []).append(v)
            adjacency.setdefault(v, []).append(u)
        return cls(adjacency, positions=positions, blocked=blocked)

    @classmethod
    def from_grid(cls, grid, allow_diagonals: bool = False) -> "ExplicitGraph":
        """Explode a :class:`~pymapf.core.grid.GridMap` into a general graph.

        Useful for measuring the cost of the abstraction, and for algorithms
        that want to rewire a grid (one-way aisles, for instance).
        """
        adjacency = {}
        positions = {}
        for r in range(grid.height):
            for c in range(grid.width):
                if not grid.is_free((r, c)):
                    continue
                adjacency[(r, c)] = list(grid.neighbors((r, c), allow_diagonals))
                positions[(r, c)] = (float(r), float(c))
        return cls(adjacency, positions=positions)

    # -- the map interface every solver uses --------------------------------
    def is_free(self, node: Node) -> bool:
        return node in self._adjacency and node not in self._blocked

    def in_bounds(self, node: Node) -> bool:
        return node in self._adjacency

    def neighbors(self, node: Node, allow_diagonals: bool = False) -> List[Node]:
        """Free neighbours of ``node``.

        ``allow_diagonals`` is accepted and ignored: adjacency in a general
        graph is whatever the caller declared, and keeping the signature makes
        this a drop-in for ``GridMap``.
        """
        return [n for n in self._adjacency.get(node, ()) if self.is_free(n)]

    @property
    def free_cells(self) -> int:
        """Number of traversable nodes (the solvers' horizon bound)."""
        return len(self._adjacency) - len(self._blocked)

    # -- extras -------------------------------------------------------------
    @property
    def nodes(self) -> List[Node]:
        return list(self._adjacency)

    @property
    def edges(self) -> List[Tuple[Node, Node]]:
        return [(u, v) for u, vs in self._adjacency.items() for v in vs]

    @property
    def height(self) -> int:
        """Bounding-box height of the layout (1 when there is no layout)."""
        rows = [p[0] for p in self.positions.values()]
        return int(max(rows) - min(rows)) + 1 if rows else 1

    @property
    def width(self) -> int:
        cols = [p[1] for p in self.positions.values()]
        return int(max(cols) - min(cols)) + 1 if cols else 1

    def degree(self, node: Node) -> int:
        return len(self._adjacency.get(node, ()))

    def is_undirected(self) -> bool:
        return all(
            u in self._adjacency.get(v, ())
            for u, vs in self._adjacency.items()
            for v in vs
        )

    def __contains__(self, node: Node) -> bool:
        return node in self._adjacency

    def __len__(self) -> int:
        return len(self._adjacency)

    def __repr__(self) -> str:
        return "ExplicitGraph(nodes=%d, edges=%d, blocked=%d)" % (
            len(self._adjacency),
            sum(len(v) for v in self._adjacency.values()),
            len(self._blocked),
        )


def roadmap(
    nodes: Sequence[Node],
    edges: Iterable[Tuple[Node, Node]],
    positions: Optional[Mapping[Node, Tuple[float, float]]] = None,
) -> ExplicitGraph:
    """Convenience builder for an undirected roadmap with isolated nodes kept."""
    adjacency: Dict[Node, List[Node]] = {node: [] for node in nodes}
    for u, v in edges:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)
    return ExplicitGraph(adjacency, positions=positions)
