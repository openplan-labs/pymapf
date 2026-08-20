import numpy as np
import time
import logging


class VelocityAgent:
    def __init__(
        self,
        ident,
        start,
        goal,
        number_of_timesteps,
        timestep,
        radius=0.5,
        vmax=2,
        vmin=0.2,
    ):
        # Agent Initialization
        self.ident = ident
        self.start = np.array([start.x, start.y])
        self.goal = np.array([goal.x, goal.y])

        # MAPF Velocity Obstacles Consts
        self.number_of_timesteps = number_of_timesteps
        self.timestep = timestep

        # Agent Consts
        self.radius = radius
        self.vmax = vmax
        self.vmin = vmin

        # Current State
        self.current_state = self.start
        self.state_history = np.empty((4, self.number_of_timesteps))
        self.total_computation_runtime = 0

    def simulate_step(self, step, obstacles, other_agents):
        # Predict Obstacles and Agents Positions in the Future
        stamp = time.time()
        v_desired = self.__compute_desired_velocity(self.current_state[:2], self.goal)
        control_vel = self.__compute_velocity(
            self.current_state, obstacles, step, v_desired, other_agents
        )
        self.current_state = self.__update_state(self.current_state, control_vel)
        self.state_history[:4, step] = self.current_state
        self.total_computation_runtime += time.time() - stamp
        return self.state_history, self.current_state, control_vel

    def __compute_desired_velocity(self, current_pos, goal_pos):
        disp_vec = (goal_pos - current_pos)[:2]
        norm = np.linalg.norm(disp_vec)
        if norm < self.radius / 5:
            return np.zeros(2)
        disp_vec = disp_vec / norm
        np.shape(disp_vec)
        desired_vel = self.vmax * disp_vec
        return desired_vel

    def __compute_velocity(self, state, obstacles, step, v_desired, other_agents):
        try:
            obstacles = obstacles[:, step, :]
            pos_agent = state[:2]
            number_of_obstacles = np.shape(obstacles)[1]

        except BaseException as e:
            logging.debug(e)
            pos_agent = state[:2]
            number_of_obstacles = 0

        a = np.empty(((number_of_obstacles + len(other_agents)) * 2, 2))
        b = np.empty(((number_of_obstacles + len(other_agents)) * 2))

        # Handle none agents Obstacles
        for i in range(number_of_obstacles):
            obstacle = obstacles[:, i]
            pos_obs = obstacle[:2]
            vel_obs = obstacle[2:]
            dispBA = pos_agent - pos_obs
            distBA = np.linalg.norm(dispBA)
            thetaBA = np.arctan2(dispBA[1], dispBA[0])

            if 2.2 * self.radius > distBA:
                distBA = 2.2 * self.radius

            phi_obst = np.arcsin(2.2 * self.radius / distBA)
            phi_left = thetaBA + phi_obst
            phi_right = thetaBA - phi_obst

            # VO
            translation = vel_obs
            a_temp, b_temp = self.__create_constraints(translation, phi_left, "left")
            a[i * 2, :] = a_temp
            b[i * 2] = b_temp
            a_temp, b_temp = self.__create_constraints(translation, phi_right, "right")
            a[i * 2 + 1, :] = a_temp
            b[i * 2 + 1] = b_temp

        # Handle other agents
        k = number_of_obstacles
        for obs_agent in other_agents:
            obs_agent_position = obs_agent[:2]
            obs_agent_velocity = obs_agent[2:]
            dispBA = pos_agent - obs_agent_position
            distBA = np.linalg.norm(dispBA)
            thetaBA = np.arctan2(dispBA[1], dispBA[0])

            if 2.2 * self.radius > distBA:
                distBA = 2.2 * self.radius

            phi_obst = np.arcsin(2.2 * self.radius / distBA)
            phi_left = thetaBA + phi_obst
            phi_right = thetaBA - phi_obst

            # VO
            translation = obs_agent_velocity
            a_temp, b_temp = self.__create_constraints(translation, phi_left, "left")
            a[k * 2, :] = a_temp
            b[k * 2] = b_temp
            a_temp, b_temp = self.__create_constraints(translation, phi_right, "right")
            a[k * 2 + 1, :] = a_temp
            b[k * 2 + 1] = b_temp
            k += 1

        # Create search-space
        th = np.linspace(0, 2 * np.pi, 20)
        vel = np.linspace(0, self.vmax, 5)
        vv, thth = np.meshgrid(vel, th)
        vx_sample = (vv * np.cos(thth)).flatten()
        vy_sample = (vv * np.sin(thth)).flatten()
        v_sample = np.stack((vx_sample, vy_sample))

        try:
            v_satisfying_constraints = self.__check_constraints(v_sample, a, b)
            # Objective function
            size = np.shape(v_satisfying_constraints)[1]
            diffs = v_satisfying_constraints - (
                (v_desired).reshape(2, 1) @ np.ones(size).reshape(1, size)
            )
            norm = np.linalg.norm(diffs, axis=0)
            min_index = np.where(norm == np.amin(norm))[0][0]
            cmd_vel = v_satisfying_constraints[:, min_index]
        except BaseException as e:
            logging.debug(e)
            cmd_vel = np.array([0, 0])
        return cmd_vel

    def __check_constraints(self, v_sample, Amat, bvec):
        length = np.shape(bvec)[0]

        for i in range(int(length / 2)):
            v_sample = self.__check_inside(
                v_sample, Amat[2 * i : 2 * i + 2, :], bvec[2 * i : 2 * i + 2]
            )

        return v_sample

    def __check_inside(self, v, Amat, bvec):
        v_out = []
        for i in range(np.shape(v)[1]):
            if not ((Amat @ v[:, i] < bvec).all()):
                v_out.append(v[:, i])
        return np.array(v_out).T

    def __create_constraints(self, translation, angle, side):
        # create line
        origin = np.array([0, 0, 1])
        # NumPy 2.0 removed np.cross support for 2-dimensional vectors, so pad
        # the direction with an explicit z=0 (the 1.x behaviour, spelled out).
        point = np.array([np.cos(angle), np.sin(angle), 0.0])
        line = np.cross(origin, point)
        line = self.__translate_line(line, translation)

        if side == "left":
            line *= -1

        A = line[:2]
        b = -line[2]

        return A, b

    def __translate_line(self, line, translation):
        matrix = np.eye(3)
        matrix[2, :2] = -translation[:2]
        return matrix @ line

    def __update_state(self, x, v):
        new_state = np.empty((4))
        new_state[:2] = x[:2] + v * self.timestep
        new_state[-2:] = v
        return new_state
