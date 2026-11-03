import numpy as np
from scipy.optimize import minimize, Bounds
import time


class NMPCAgent:
    def __init__(
        self,
        ident,
        start,
        goal,
        number_of_timesteps,
        nmpc_timestep,
        timestep,
        qc=5.0,
        kappa=4.0,
        radius=0.5,
        vmax=2,
        vmin=0.2,
        horizon_length=4,
        seed=0,
    ):
        # Initializations
        self.ident = ident
        self.start = np.array([start.x, start.y])
        self.goal = np.array([goal.x, goal.y])

        # Consts from NMPC Class
        self.number_of_timesteps = number_of_timesteps
        self.nmpc_timestep = nmpc_timestep
        self.timestep = timestep

        # Agent Constraints
        self.horizon_length = horizon_length
        self.vmax = vmax
        self.vmin = vmin
        self.qc = qc
        self.kappa = kappa
        self.upper_bound = [(1 / np.sqrt(2)) * self.vmax] * self.horizon_length * 2
        self.lower_bound = [-(1 / np.sqrt(2)) * self.vmax] * self.horizon_length * 2
        self.radius = radius

        # The optimiser starts from a random control sequence (it breaks the
        # symmetry of head-on encounters); seed it so runs are reproducible.
        self.rng = np.random.default_rng(seed)

        # Current State
        self.current_state = self.start
        self.state_history = np.empty((4, self.number_of_timesteps))
        self.total_computation_runtime = 0

    def simulate_step(self, step, obstacles, other_agents):
        # Predict Obstacles and Agents Positions in the Future
        stamp = time.time()
        obstacle_prediction = self.__predict_obstacle_positions(
            obstacles, step, other_agents
        )
        xref = self.__compute_xref()
        vel, _ = self.__compute_velocity(self.current_state, obstacle_prediction, xref)
        self.current_state = self.__update_state(self.current_state, vel)
        self.state_history[:2, step] = self.current_state
        self.total_computation_runtime += time.time() - stamp
        return self.state_history, vel, self.current_state

    def __compute_velocity(self, robot_state, obstacle_predictions, xref):
        u0 = self.rng.random(2 * self.horizon_length)

        def cost_fn(u):
            return self.__total_cost(u, robot_state, obstacle_predictions, xref)

        bounds = Bounds(self.lower_bound, self.upper_bound)

        res = minimize(cost_fn, u0, method="SLSQP", bounds=bounds)
        velocity = res.x[:2]
        return velocity, res.x

    def __compute_xref(self):
        dir_vec = self.goal - self.current_state
        norm = np.linalg.norm(dir_vec)
        if norm < 0.1:
            new_goal = self.current_state
        else:
            dir_vec = dir_vec / norm
            new_goal = (
                self.current_state
                + dir_vec * self.vmax * self.nmpc_timestep * self.horizon_length
            )
        return np.linspace(self.current_state, new_goal, self.horizon_length).reshape(
            (2 * self.horizon_length)
        )

    def __total_cost(self, u, robot_state, obstacle_predictions, xref):
        x_robot = self.__update_state(robot_state, u)
        c1 = self.__tracking_cost(x_robot, xref)
        c2 = self.__total_collision_cost(x_robot, obstacle_predictions)
        total = c1 + c2
        return total

    def __tracking_cost(self, x, xref):
        return np.linalg.norm(x - xref)

    def __total_collision_cost(self, robot, obstacles):
        total_cost = 0
        for i in range(self.horizon_length):
            for j, _ in enumerate(obstacles):
                obstacle = obstacles[j]
                rob = robot[2 * i : 2 * i + 2]
                obs = obstacle[2 * i : 2 * i + 2]
                total_cost += self.__collision_cost(rob, obs)
        return total_cost

    def __collision_cost(self, x0, x1):
        """
        Cost of collision between two robot_state
        """
        d = np.linalg.norm(x0 - x1)
        cost = self.qc / (1 + np.exp(self.kappa * (d - 2 * self.radius)))
        return cost

    def __predict_obstacle_positions(self, obstacles, step, other_agents):
        obstacle_predictions = []
        # ``obstacles`` is the manager's ``(4, timesteps, n_obstacles)`` stack
        # of ``[x, y, vx, vy]`` trajectories (or an empty list when there are
        # none).  Take every obstacle's state *at this step* and roll it
        # forward at constant velocity over the horizon.
        if np.size(obstacles):
            for obstacle in np.asarray(obstacles)[:, step, :].T:
                obstacle_position = obstacle[:2]
                obstacle_vel = obstacle[2:]
                u = np.vstack([np.eye(2)] * self.horizon_length) @ obstacle_vel
                obstacle_prediction = self.__update_state(obstacle_position, u)
                obstacle_predictions.append(obstacle_prediction)

        for agent in other_agents:
            agent_position = agent[:2]
            agent_vel = agent[2:]
            u = np.vstack([np.eye(2)] * self.horizon_length) @ agent_vel
            obstacle_agent_prediction = self.__update_state(agent_position, u)
            obstacle_predictions.append(obstacle_agent_prediction)

        return obstacle_predictions

    def __update_state(self, x0, u):
        """
        Computes the states of the system after applying a sequence of control signals u on
        initial state x0
        """
        N = int(len(u) / 2)
        lower_triangular_ones_matrix = np.tril(np.ones((N, N)))
        kron = np.kron(lower_triangular_ones_matrix, np.eye(2))

        new_state = np.vstack([np.eye(2)] * int(N)) @ x0 + kron @ u * self.nmpc_timestep

        return new_state

    def __hash__(self):
        # Consistent with ``__eq__``, which compares identifiers only.
        return hash(self.ident)

    def __eq__(self, other):
        return self.ident == other.ident

    def __str__(self):
        return (
            "=======AGENT=======\nID = %s\nSTART_POSITION = %s\nGOAL_POSITION = %s\nRADIUS = %s\nVMIN = %s\nVMAX = %s\nHORIZON_LENGTH = %s\nQC = %s\nKappa = %s\n==================="
            % (
                str(self.ident),
                str(self.start),
                str(self.goal),
                str(self.radius),
                str(self.vmin),
                str(self.vmax),
                str(self.horizon_length),
                str(self.qc),
                str(self.kappa),
            )
        )
