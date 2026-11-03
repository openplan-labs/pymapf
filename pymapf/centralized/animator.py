#!/usr/bin/env python3
import matplotlib
from random import random
from matplotlib.patches import Circle, Rectangle, Arrow
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import math
import logging

logger = logging.getLogger(__name__)

Colors = ["skyblue", "blue", "orange"]


class Animator:
    def __init__(self, world, paths, agents, simulation_time):
        self.world = world
        self.paths = paths
        self.coop_agents = agents

        aspect = self.world.length / self.world.height
        self.fig = plt.figure(frameon=False, figsize=(4 * aspect, 4))
        self.ax = self.fig.add_subplot(111, aspect="equal")
        self.fig.subplots_adjust(
            left=0, right=1, bottom=0, top=1, wspace=None, hspace=None
        )
        self.patches = []
        self.artists = []
        self.agents = dict()
        self.agents_labels = dict()

        xmin = -0.5
        ymin = -0.5
        xmax = self.world.length - 0.5
        ymax = self.world.height - 0.5
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        self.__initialize_obstacles(xmin, ymin, xmax, ymax)
        self.__initialize_agents()
        self.simulation_time = simulation_time
        self.__animate()

    def __initialize_obstacles(self, xmin, ymin, xmax, ymax):
        self.patches.append(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                facecolor="none",
                edgecolor="black",
            )
        )

        try:
            for pos in self.world.walls:
                self.patches.append(
                    Rectangle(
                        (pos[1] - 0.5, pos[0] - 0.5),
                        1,
                        1,
                        facecolor="black",
                        edgecolor="black",
                    )
                )
        except BaseException as e:
            logger.debug(e)

    def __initialize_agents(self):
        for _, agent in self.coop_agents.items():
            self.patches.append(
                Rectangle(
                    (agent.goal_pos[1] - 0.25, agent.goal_pos[0] - 0.25),
                    0.5,
                    0.5,
                    facecolor=Colors[0],
                    edgecolor="black",
                    alpha=0.5,
                )
            )
            self.agents[agent.ident] = Circle(
                (agent.init_pos[0], agent.init_pos[1]),
                0.3,
                facecolor=Colors[0],
                edgecolor="black",
            )
            self.agents[agent.ident].original_face_color = Colors[0]
            self.patches.append(self.agents[agent.ident])
            self.agents_labels[agent.ident] = self.ax.text(
                agent.init_pos[0], agent.init_pos[1], agent.ident
            )
            self.agents_labels[agent.ident].set_horizontalalignment("center")
            self.agents_labels[agent.ident].set_verticalalignment("center")
            self.artists.append(self.agents_labels[agent.ident])

    def __animate(self):
        self.anim = animation.FuncAnimation(
            self.fig,
            self.__animations,
            init_func=self.__initialize_animation,
            frames=int(self.simulation_time + 1) * 10,
            interval=250,
            blit=False,
        )

    def save(self, file_name):
        self.anim.save(file_name + ".gif", "ffmpeg", fps=5)
        logger.debug("Saved file as %s" % file_name + ".gif")

    def show(self):
        plt.show()

    def __initialize_animation(self):
        for p in self.patches:
            self.ax.add_patch(p)
        for a in self.artists:
            self.ax.add_artist(a)
        return self.patches + self.artists

    def __animations(self, i):
        for agent_name, path in self.paths.items():
            try:
                pos = (path[i].x, path[i].y)
            except BaseException as e:
                logger.debug(e)
                pos = (path[-1].x, path[-1].y)
            p = (pos[0], pos[1])
            self.agents[agent_name].center = p
            self.agents_labels[agent_name].set_position(p)

        # reset all colors
        for _, agent in self.agents.items():
            agent.set_facecolor(agent.original_face_color)

        # check drive-drive collisions
        agents_array = [agent for _, agent in self.agents.items()]
        for i, _ in enumerate(agents_array):
            for j in range(i + 1, len(agents_array)):
                d1 = agents_array[i]
                d2 = agents_array[j]
                pos1 = np.array(d1.center)
                pos2 = np.array(d2.center)
                if np.linalg.norm(pos1 - pos2) < 1:
                    d1.set_facecolor("red")
                    d2.set_facecolor("red")
                    print("COLLISION! (agent-agent) ({}, {})".format(i, j))

        return self.patches + self.artists
