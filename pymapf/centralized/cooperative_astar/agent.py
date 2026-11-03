"""
An agent is defined by an Id, a start position and a goal position.
To do: add radius
"""

import logging

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self, ident, init_pos, goal_pos, allow_diagonals=False, nodes_dict=None
    ):
        self.ident = ident
        self.init_pos = init_pos
        self.goal_pos = goal_pos
        self.nodes_dict = nodes_dict
        self.plan = []
        self.conflicts_found = 0
        self.opened_nodes = 0
        self.allow_diagonals = allow_diagonals

    @staticmethod
    def position_at(path, t):
        """Cell ``(x, y)`` occupied at time ``t`` by an agent following ``path``.

        ``path[k]`` is the state at time ``k``.  Once its path ends, an agent
        stays where it finished (its goal, or its start if it has not been
        planned yet or its search failed), forever.
        """
        state = path[min(t, len(path) - 1)]
        return state.x, state.y

    def in_conflict(self, current_state, future_state, other_agents_paths):
        """Would moving ``current_state -> future_state`` collide with anybody?

        Checked against every other agent's reserved path:

        * vertex conflict: both agents in the same cell at ``t + 1``;
        * edge (swap) conflict: the two agents trade cells between ``t`` and
          ``t + 1``, passing through each other;
        * with diagonals, a crossing conflict: both agents cut the same 2x2
          square along its two opposite diagonals at the same time.
        """
        t = current_state.t
        here = (current_state.x, current_state.y)
        there = (future_state.x, future_state.y)
        for key, val in other_agents_paths.items():
            if key == self.ident or not val:
                continue

            other_now = self.position_at(val, t)
            other_next = self.position_at(val, t + 1)

            conflict = other_next == there or (
                other_next == here and other_now == there
            )
            diagonal = here[0] != there[0] and here[1] != there[1]
            if not conflict and self.allow_diagonals and diagonal:
                # Moving diagonally from ``here`` to ``there``: the other agent
                # crosses us if it goes between the two remaining corners.
                corners = {(here[0], there[1]), (there[0], here[1])}
                conflict = (
                    other_now in corners
                    and other_next in corners
                    and other_now != other_next
                )

            if conflict:
                logger.debug(
                    "Found conflict between agent %s and agent %s" % (self.ident, key)
                )
                self.conflicts_found += 1
                return True
        return False

    def can_stay(self, state, other_agents_paths):
        """Can the agent stop at ``state`` and stay there for good?

        Arriving at the goal is not enough: an agent planned earlier may still
        pass through this cell later on, and would run into us while we wait.
        """
        cell = (state.x, state.y)
        for key, val in other_agents_paths.items():
            if key == self.ident or not val:
                continue
            # From ``state.t`` until the other agent's path ends (after which it
            # stays put, which the last time step covers).
            for t in range(state.t, max(state.t, len(val) - 1) + 1):
                if self.position_at(val, t) == cell:
                    return False
        return True

    def __str__(self):
        return "Id: %d | Init: [%d;%d] | Goal: [%d;%d]" % (
            self.ident,
            self.init_pos[0],
            self.init_pos[1],
            self.goal_pos[0],
            self.goal_pos[1],
        )
