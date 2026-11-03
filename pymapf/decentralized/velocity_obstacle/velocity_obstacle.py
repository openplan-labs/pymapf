"""
Decentralized planning using velocity obstacles
author: Erwin Lejeune (erwin.lejeune15@gmail.com)
"""

from ..obstacle import Obstacle
from .velocity_agent import VelocityAgent
import numpy as np
import matplotlib as mpl
from random import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# import threading
from matplotlib.patches import Circle
import logging

logger = logging.getLogger(__name__)


class MultiAgentVelocityObstacle:
    def __init__(self, simulation_time=8.0, timestep=0.1, log_level="WARNING"):
        self.simulation_time = simulation_time
        self.timestep = timestep
        self.number_of_timesteps = int(simulation_time / timestep)

        # Agents
        self.agents = dict()
        self.global_state_history = dict()

        # Obstacles
        self.obstacles_objects = []

        # Flags
        self.simulation_complete = False
        # ``log_level`` sets the level of this planner's own loggers (this
        # package) only.  The library does not configure the root logger or
        # write log files: attach a handler in your application to see them.
        logging.getLogger(__package__).setLevel(log_level)

    def register_agent(self, ident, start, goal, radius=0.5, vmax=2, vmin=0.2):
        if ident in self.agents:
            logger.warning("Failed to register agent: ID has already been registered.")
            return
        agent = VelocityAgent(
            ident,
            start,
            goal,
            self.number_of_timesteps,
            self.timestep,
            radius,
            vmax,
            vmin,
        )
        self.agents[ident] = agent

    def register_obstacle(self, velocity, theta, initial_position):
        obstacle = Obstacle(
            False,
            velocity,
            theta,
            initial_position,
            self.simulation_time,
            self.number_of_timesteps,
        )
        self.obstacles_objects.append(obstacle)
        if not hasattr(self, "obstacles"):
            self.obstacles = obstacle.state
        else:
            self.obstacles = np.dstack((self.obstacles, obstacle.state))

    def run_simulation(self):
        try:
            obstacles = self.obstacles
        except BaseException as e:
            logger.debug(e)
            obstacles = []
        # What every agent knows about the others: their position and velocity
        # at the end of the previous step, as ``[x, y, vx, vy]``.  Everyone
        # decides simultaneously from the same snapshot, so the result does not
        # depend on the order in which the agents were registered (and the
        # first agent is not blind to the rest, as it was when this list was
        # rebuilt from scratch inside the step).
        last_seen = {
            key: np.concatenate((agent.start, np.zeros(2)))
            for key, agent in self.agents.items()
        }
        for i in range(self.number_of_timesteps):
            snapshot = dict(last_seen)
            for key, agent in self.agents.items():
                others = [obs for other, obs in snapshot.items() if other != key]
                last_seen[key] = self.__agent_step(key, agent, i, obstacles, others)
        self.simulation_complete = True

    def visualize(self, saved_file, map_length, map_height):
        if not self.simulation_complete:
            logger.warning("Failed to visualize: simulation must be complete")
            return

        self.__plot(saved_file, map_length, map_height)

    def __agent_step(self, key, agent, i, obstacles, other_agents_lst):
        # ``simulate_step`` returns ``(history, state, velocity)`` where
        # ``state`` is ``[x, y, vx, vy]``: unpack it in that order, or the
        # other agents end up seeing a velocity where the position should be.
        state_history, state, vel = agent.simulate_step(i, obstacles, other_agents_lst)
        self.global_state_history[key] = state_history
        return self.__agent_to_obstacle(vel, state[:2])

    def __agent_to_obstacle(self, velocity, pos):
        return np.concatenate((pos, velocity))

    def __plot(self, saved_file, map_length, map_height):
        fig = plt.figure()
        ax = fig.add_subplot(
            111, autoscale_on=False, xlim=(0, map_length), ylim=(0, map_height)
        )
        ax.set_aspect("equal")
        ax.grid()

        lines = []
        agents_list = []
        for key, agent in self.agents.items():
            rgb = [random(), random(), random()]
            agents_list.append(
                Circle(
                    (
                        self.global_state_history[key][0, 0],
                        self.global_state_history[key][0, 1],
                    ),
                    agent.radius,
                    facecolor=rgb,
                    edgecolor="black",
                )
            )
            (line,) = ax.plot([], [], c=rgb)
            lines.append(line)
        obstacle_list = []
        try:
            for obstacle in range(np.shape(self.obstacles)[2]):
                obstacle_list.append(
                    Circle(
                        (0, 0),
                        self.obstacles_objects[obstacle].radius,
                        facecolor="red",
                        edgecolor="black",
                    )
                )
        except BaseException as e:
            logger.debug(e)

        def init():
            for agent in agents_list:
                ax.add_patch(agent)
                line.set_data([], [])
                lines.append(line)
            for obstacle in obstacle_list:
                ax.add_patch(obstacle)

            return agents_list + lines + obstacle_list

        def animate(i):
            k = 0
            for key, _ in self.agents.items():
                agents_list[k].center = (
                    self.global_state_history[key][0, i],
                    self.global_state_history[key][1, i],
                )
                lines[k].set_data(
                    self.global_state_history[key][0, :i],
                    self.global_state_history[key][1, :i],
                )
                k += 1
            for j, _ in enumerate(obstacle_list):
                obstacle_list[j].center = (
                    self.obstacles[0, i, j],
                    self.obstacles[1, i, j],
                )
            return agents_list + lines + obstacle_list

        init()
        step = self.simulation_time / self.number_of_timesteps
        for i in range(self.number_of_timesteps):
            animate(i)
            plt.pause(step)

        # Save animation
        if not saved_file:
            return

        ani = animation.FuncAnimation(
            fig,
            animate,
            np.arange(1, self.number_of_timesteps),
            interval=200,
            blit=True,
            init_func=init,
        )

        ani.save(saved_file + str(".gif"), writer="ffmpeg", fps=10)
