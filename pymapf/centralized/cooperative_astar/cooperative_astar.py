from .agent import Agent
from .astar import AStar
import time
from ..common import TIME
from ..world import World
from .state import State
from ..animator import Animator
import coloredlogs
import logging


class CooperativeAStar:
    def __init__(self, world, heuristic=0, log_level="WARNING"):
        self.world = world
        self.allow_diagonals = world.allow_diagonals
        self.heuristic = 0
        self.agents = dict()
        self.paths = dict()
        self.searches_sim_times = []

        # Flags
        self.simulation_complete = False
        self.__init_logger(log_level)
        coloredlogs.install(level=log_level)

    def register_agent(self, ident, start, goal):
        if ident in self.agents:
            logging.warning("Agent ID already registered, ignoring...")
            return
        # A shared start collides at t = 0 and a shared goal can only be held
        # by one agent: neither instance has a conflict-free solution.
        for other in self.agents.values():
            if tuple(other.init_pos) == tuple(start) or tuple(other.goal_pos) == tuple(goal):
                logging.warning(
                    "Agent %s shares a start or goal with agent %s, ignoring..."
                    % (ident, other.ident)
                )
                return
        self.agents[ident] = Agent(
            ident, start, goal, allow_diagonals=self.allow_diagonals
        )
        self.paths[ident] = [State(start[1], start[0], 0)]

    def run_simulation(self):
        for _, agent in self.agents.items():
            astar = AStar(agent, self.world, self.paths)
            self.paths[agent.ident] = astar.search()
            try:
                self.searches_sim_times.append(self.paths[agent.ident][-1].t)
            except BaseException as e:
                logging.debug(e)

        self.simulation_complete = True

    def visualize(self, save_file):
        if not self.simulation_complete:
            logging.warning("Simulation isn't yet complete: can't visualize.")
            return
        anim = Animator(
            self.world, self.paths, self.agents, max(self.searches_sim_times)
        )
        anim.save(save_file)
        anim.show()

    def __init_logger(self, log_level):
        import os

        if not os.path.exists("logs"):
            os.makedirs("logs")
        logging.basicConfig(
            filename="logs/main.log",
            format="%(levelname)s:%(message)s",
            filemode="w",
            level=log_level,
        )
