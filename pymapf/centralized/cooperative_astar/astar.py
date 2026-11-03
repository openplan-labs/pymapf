from .node import Node
from typing import Tuple, List
from termcolor import colored
from ..world import World
from .state import State
import logging
from .agent import Agent


class AStar:
    """
    >>> wd = World(10, 10, 0.2)
    >>> astar = AStar((0, 0), (len(wd.grid) - 1, len(wd.grid[0]) - 1), wd)
    >>> (astar.start.pos_y + wd.delta[3][0], astar.start.pos_x + wd.delta[3][1])
    (0, 1)
    >>> [x.pos for x in astar.get_successors(astar.start)]
    [(1, 0), (0, 1)]
    >>> (astar.start.pos_y + wd.delta[2][0], astar.start.pos_x + wd.delta[2][1])
    (1, 0)
    >>> astar.retrace_path(astar.start)
    [(0, 0)]
    >>> astar.search()  # doctest: +NORMALIZE_WHITESPACE
    [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3),
     (4, 3), (4, 4), (5, 4), (5, 5), (6, 5), (6, 6)]
    """

    def __init__(self, agent: Agent, world: World, global_paths):
        self.start = Node(
            agent.init_pos[1],
            agent.init_pos[0],
            0,
            agent.goal_pos[1],
            agent.goal_pos[0],
            0,
            False,
            True,
            None,
        )
        self.target = Node(
            agent.goal_pos[1],
            agent.goal_pos[0],
            -1,
            agent.goal_pos[1],
            agent.goal_pos[0],
            float("inf"),
            False,
            False,
            None,
        )
        self.world = world
        self.agent = agent

        self.nodes = dict()
        self.nodes[self.start.state] = self.start

        self.global_paths = global_paths
        self.opened_nodes = 1

        self.reached = False

    def search(self) -> List[Tuple[int]]:
        # Space-time A*: a node is a cell *at a time step*, so every node is
        # keyed by its ``State`` (x, y, t), the start included.  Keying some
        # nodes by cell instead let a later arrival overwrite an already
        # closed node and reopen it.
        opened_nodes = 0
        while self.opened_nodes > 0:
            # Open Nodes are sorted using __lt__
            current_key = min(
                [n for n in self.nodes if self.nodes[n].is_open],
                key=(lambda k: self.nodes[k].f_cost),
            )
            current_node = self.nodes[current_key]

            # Stop at the goal only if nobody planned earlier still has to
            # pass through it: we are going to stay there.
            if current_node.pos == self.target.pos and self.agent.can_stay(
                current_node.state, self.global_paths
            ):
                logging.debug("Found path for agent [%s]" % str(self.agent.ident))
                self.agent.opened_nodes = opened_nodes
                self.agent.path = self.retrace_path(current_node)
                return self.agent.path

            if current_node.t > 2 * self.start.h_cost:
                break

            current_node.is_closed = True
            current_node.is_open = False
            self.opened_nodes -= 1

            successors = self.get_successors(current_node)
            opened_nodes += len(successors)

            for child_node in successors:
                known = self.nodes.get(child_node.state)
                if known is None:
                    self.nodes[child_node.state] = child_node
                    self.opened_nodes += 1
                elif known.is_open and child_node.g_cost < known.g_cost:
                    # Same cell, same time, cheaper way there: replace it.
                    self.nodes[child_node.state] = child_node
                # Otherwise it is closed, or already open at least as cheaply.
        logging.warning("Path not found for agent [%s]" % str(self.agent.ident))
        return [self.start.state]

    def get_successors(self, parent: Node) -> List[Node]:
        """
        Returns a list of successors (both in the world and free spaces)
        """
        successors = []
        for action in self.world.delta:
            pos_x = parent.pos_x + action[0]
            pos_y = parent.pos_y + action[1]
            if not (0 <= pos_x < self.world.length and 0 <= pos_y < self.world.height):
                continue

            if self.world.grid[pos_y][pos_x] != 0:
                continue

            if self.agent.in_conflict(
                State(parent.pos_x, parent.pos_y, parent.t),
                State(pos_x, pos_y, parent.t + 1),
                self.global_paths,
            ):
                continue

            successors.append(
                Node(
                    pos_x,
                    pos_y,
                    parent.t + 1,
                    self.target.pos_x,
                    self.target.pos_y,
                    parent.g_cost + action[2],
                    False,
                    True,
                    parent,
                )
            )
        return successors

    def retrace_path(self, node: Node) -> List[State]:
        """
        Retrace the path from parents to parents until start node
        """
        current_node = node
        path = []
        while current_node is not None:
            path.append(current_node.state)
            current_node = current_node.parent
        path.reverse()
        return path
