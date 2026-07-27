"""Decentralized flocking: from Reynolds' boids to acceleration-based control.

Multi-agent path finding assumes a discrete graph, a central planner and
complete information. A drone swarm has none of those: each vehicle sees a few
neighbours, runs its own control loop at 50 Hz, and there is no plan to follow.
This module holds that side of the problem -- reactive, decentralized,
continuous-space collective motion -- in the same shape as the rest of the
library: a small set of named, swappable controllers plus a simulator that
measures them.

Every controller here has the same signature::

    acceleration = controller(state, index, neighbours, params)

so they can be compared on identical initial conditions, which is the entire
point of :func:`simulate`.

The four models
---------------

``boids``
    Reynolds' three rules -- separation, alignment, cohesion -- as steering
    accelerations. The origin of the entire field.

``vicsek``
    Pure heading alignment with noise, at constant speed. The minimal model of
    a phase transition from disorder to collective motion.

``olfati_saber``
    Gradient-based flocking with a smooth potential over the sigma-norm, plus
    velocity consensus and a navigational feedback term. The first framework
    with stability proofs, and the reference point for control-theoretic work.

``acceleration``
    Acceleration-based bird-inspired flocking: self-propulsion and drag give
    each agent a preferred cruise speed, while pairwise attraction/repulsion
    and velocity alignment shape the group. Because the control variable is
    acceleration rather than velocity or position, the commands map directly
    onto what a multirotor's inner loop can execute, and the flock keeps
    momentum through turns instead of snapping to a new reference.

    This follows the model family of Iacone et al. (2024); their published
    parameter values were not available when this was written, so the defaults
    below were tuned here for stable flight at 10 Hz and should be treated as a
    starting point rather than a reproduction of the paper's results.

References
----------
* Reynolds, C. W. 1987. *Flocks, herds and schools: A distributed behavioral
  model.* SIGGRAPH 1987: 25-34.
* Vicsek, T.; Czirok, A.; Ben-Jacob, E.; Cohen, I.; and Shochet, O. 1995.
  *Novel type of phase transition in a system of self-driven particles.*
  Physical Review Letters 75(6): 1226-1229.
* Cucker, F.; and Smale, S. 2007. *Emergent behavior in flocks.* IEEE
  Transactions on Automatic Control 52(5): 852-862.
* Olfati-Saber, R. 2006. *Flocking for multi-agent dynamic systems: Algorithms
  and theory.* IEEE Transactions on Automatic Control 51(3): 401-420.
* Vasarhelyi, G.; Viragh, C.; Somorjai, G.; Nepusz, T.; Eiben, A. E.; and
  Vicsek, T. 2018. *Optimized flocking of autonomous drones in confined
  environments.* Science Robotics 3(20): eaat3536.
* Iacone, L.; Lejeune, E.; Manoni, T.; Manfredi, S.; and Albani, D. 2024.
  *Decentralized acceleration-based bird-inspired flocking.* IEEE, 2024.
  (Autonomous Robotics Research Centre, Technology Innovation Institute.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "FlockParams",
    "FlockState",
    "boids",
    "vicsek",
    "olfati_saber",
    "acceleration",
    "CONTROLLERS",
    "get_controller",
    "simulate",
    "FlockMetrics",
]


@dataclass
class FlockParams:
    """Every knob the controllers share, so they can be swapped mid-comparison.

    Distances are metres, speeds m/s, accelerations m/s^2.
    """

    # neighbourhood
    sensing_range: float = 6.0
    topological_k: Optional[int] = None  # if set, use the k nearest instead of a radius

    # geometry
    separation_distance: float = 1.5  # desired minimum spacing
    reference_distance: float = 3.0  # preferred neighbour spacing

    # gains
    separation_gain: float = 6.0
    cohesion_gain: float = 1.2
    alignment_gain: float = 2.5
    migration_gain: float = 0.8

    # acceleration-based model
    cruise_speed: float = 4.0
    propulsion_gain: float = 1.6
    drag_gain: float = 0.4
    potential_gain: float = 3.0
    potential_softening: float = 0.35

    # Olfati-Saber's gradient term is scaled separately: its action function
    # already saturates at +-(a+b)/2, so reusing the acceleration model's gain
    # keeps the total command permanently against the acceleration limit, which
    # destroys the balance between attraction, repulsion and consensus.
    gradient_gain: float = 0.6

    # limits and noise
    max_speed: float = 8.0
    max_acceleration: float = 6.0
    noise: float = 0.0

    # Share of the acceleration budget the waypoint term may claim; the rest is
    # always available for separation.
    migration_authority: float = 0.35
    # The c2 term of Olfati-Saber's navigational feedback: velocity feedback
    # toward the reference velocity. Without it the gradient term overshoots and
    # the flock passes through itself, because the sigma-space direction n_ij
    # vanishes at zero separation and cannot push the pair apart again.
    navigation_damping: float = 0.9
    migration_point: Optional[Tuple[float, ...]] = None
    obstacles: Sequence[Tuple[Tuple[float, ...], float]] = ()  # (centre, radius)
    obstacle_gain: float = 12.0
    seed: int = 0


@dataclass
class FlockState:
    """Positions and velocities of the whole flock at one instant."""

    positions: np.ndarray  # (n, d)
    velocities: np.ndarray  # (n, d)

    @property
    def n(self) -> int:
        return self.positions.shape[0]

    @property
    def dimension(self) -> int:
        return self.positions.shape[1]

    def copy(self) -> "FlockState":
        return FlockState(self.positions.copy(), self.velocities.copy())


def _neighbours(state: FlockState, index: int, params: FlockParams) -> np.ndarray:
    """Indices of the agents ``index`` can see.

    Metric by default (a sensing radius); topological when ``topological_k`` is
    set, which is what starlings actually do -- each bird tracks a fixed number
    of neighbours regardless of density, and that is what keeps a real flock
    cohesive under compression (Ballerini et al. 2008, PNAS 105(4): 1232-1237).
    """
    offsets = state.positions - state.positions[index]
    distances = np.linalg.norm(offsets, axis=1)
    distances[index] = np.inf
    if params.topological_k:
        order = np.argsort(distances)
        return order[: params.topological_k]
    return np.flatnonzero(distances <= params.sensing_range)


def _limit(vector: np.ndarray, magnitude: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > magnitude and norm > 0:
        return vector * (magnitude / norm)
    return vector


def _migration(state: FlockState, index: int, params: FlockParams) -> np.ndarray:
    """Bounded navigational feedback toward the migration point.

    The bound is the whole point. An unsaturated ``gain * (target - position)``
    term grows with distance, so a waypoint 60 m away produces a command an
    order of magnitude larger than separation; the final clamp on the total
    command then scales the collision avoidance down to nothing and the flock
    flies to its goal in a heap. Capping migration at a fraction of the
    acceleration budget keeps separation authoritative at every distance.
    """
    if params.migration_point is None:
        return np.zeros(state.dimension)
    target = np.asarray(params.migration_point, dtype=float)
    return _limit(
        params.migration_gain * (target - state.positions[index]),
        params.migration_authority * params.max_acceleration,
    )


def _obstacle_repulsion(state: FlockState, index: int, params: FlockParams) -> np.ndarray:
    push = np.zeros(state.dimension)
    for centre, radius in params.obstacles:
        offset = state.positions[index] - np.asarray(centre, dtype=float)
        distance = float(np.linalg.norm(offset))
        margin = distance - radius
        if margin < params.sensing_range and distance > 1e-9:
            push += params.obstacle_gain * offset / (distance * max(margin, 0.2) ** 2)
    return push


# --------------------------------------------------------------------------
# controllers
# --------------------------------------------------------------------------


def boids(state: FlockState, index: int, params: FlockParams) -> np.ndarray:
    """Reynolds (1987): separation + alignment + cohesion as accelerations."""
    neighbours = _neighbours(state, index, params)
    acceleration_command = np.zeros(state.dimension)
    if len(neighbours):
        offsets = state.positions[neighbours] - state.positions[index]
        distances = np.linalg.norm(offsets, axis=1, keepdims=True)
        distances = np.maximum(distances, 1e-6)

        # separation: push away, weighted by 1/d
        close = distances[:, 0] < params.separation_distance
        if np.any(close):
            acceleration_command -= params.separation_gain * np.sum(
                offsets[close] / distances[close] ** 2, axis=0
            )
        # cohesion: steer toward the local centroid
        acceleration_command += params.cohesion_gain * (
            offsets.mean(axis=0)
        )
        # alignment: match the local mean velocity
        acceleration_command += params.alignment_gain * (
            state.velocities[neighbours].mean(axis=0) - state.velocities[index]
        )

    acceleration_command += _migration(state, index, params)
    acceleration_command += _obstacle_repulsion(state, index, params)
    return _limit(acceleration_command, params.max_acceleration)


def vicsek(state: FlockState, index: int, params: FlockParams) -> np.ndarray:
    """Vicsek et al. (1995): align heading with neighbours, constant speed.

    Expressed as an acceleration so it can share the integrator with the other
    controllers: the command is whatever turns the current velocity toward the
    local mean heading within one step.
    """
    neighbours = _neighbours(state, index, params)
    velocities = state.velocities[np.append(neighbours, index)]
    mean = velocities.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-9:
        heading = state.velocities[index]
    else:
        heading = mean / norm * params.cruise_speed
    command = params.alignment_gain * (heading - state.velocities[index])
    command += _obstacle_repulsion(state, index, params)
    return _limit(command, params.max_acceleration)


def _sigma_norm(vector: np.ndarray, epsilon: float = 0.1) -> float:
    """Olfati-Saber's smooth norm: differentiable everywhere, including zero."""
    return (math.sqrt(1 + epsilon * float(vector @ vector)) - 1) / epsilon


def _bump(z: float, h: float = 0.2) -> float:
    """Olfati-Saber's rho_h: 1 inside, smoothly to 0 at the sensing edge.

    This is what makes the neighbourhood *smooth* -- an agent entering or
    leaving sensing range does not produce a step change in anyone's command,
    which is precisely the property the paper's stability proof needs.
    """
    if z < 0:
        return 0.0
    if z < h:
        return 1.0
    if z <= 1.0:
        return 0.5 * (1 + math.cos(math.pi * (z - h) / (1 - h)))
    return 0.0


def _action(z: float, d_alpha: float, a: float = 5.0, b: float = 5.0) -> float:
    """The action function phi_alpha: repulsive below ``d_alpha``, attractive above.

    ``phi(z) = 0.5[(a+b) sigma_1(z+c) + (a-b)]`` with ``sigma_1(z) = z/sqrt(1+z^2)``
    and ``c = |a-b|/sqrt(4ab)``, as in Olfati-Saber (2006), eq. (15)-(16).
    """
    c = abs(a - b) / math.sqrt(4 * a * b) if a * b > 0 else 0.0
    z_shift = z - d_alpha
    sigma_1 = (z_shift + c) / math.sqrt(1 + (z_shift + c) ** 2)
    return 0.5 * ((a + b) * sigma_1 + (a - b))


def olfati_saber(state: FlockState, index: int, params: FlockParams) -> np.ndarray:
    """Olfati-Saber (2006): gradient flocking with the paper's action function.

    Three terms, exactly as in the reference: a gradient term over the smooth
    pairwise potential, velocity consensus weighted by the same smooth
    adjacency, and navigational feedback (both the position and the velocity
    term).

    One property is worth knowing before deploying this: the gradient direction
    ``n_ij`` is the sigma-space one, whose magnitude *vanishes* as two agents
    approach each other. The repulsion therefore peaks at intermediate range
    and fades at contact, so a pair driven together by other terms -- or by a
    saturating acceleration limit -- can merge and stay merged. The
    acceleration-based controller in this module does not have that failure
    mode, because its short-range term is an unbounded ``1/d^2`` repulsion.
    """
    neighbours = _neighbours(state, index, params)
    command = np.zeros(state.dimension)
    epsilon = 0.1

    # Interaction and sensing ranges expressed in sigma-norm space. The paper's
    # alpha-lattice is defined for an interaction range r ~ 1.2 d; measured on
    # this simulator, r/d = 2 makes the lattice collapse outright (every agent
    # sees enough distant neighbours that summed attraction beats local
    # repulsion), while r/d = 1.2 reproduces an exact lattice at spacing d. The
    # ratio is therefore part of the algorithm, not a free parameter.
    interaction_range = min(params.sensing_range, 1.2 * params.reference_distance)
    d_alpha = _sigma_norm(
        np.array([params.reference_distance] + [0.0] * (state.dimension - 1)), epsilon
    )
    r_alpha = _sigma_norm(
        np.array([interaction_range] + [0.0] * (state.dimension - 1)), epsilon
    )

    for j in neighbours:
        offset = state.positions[j] - state.positions[index]
        sigma_distance = _sigma_norm(offset, epsilon)
        # n_ij: the sigma-space gradient direction (bounded, unlike q_j - q_i).
        direction = offset / math.sqrt(1 + epsilon * float(offset @ offset))
        weight = _bump(sigma_distance / r_alpha) if r_alpha > 0 else 0.0

        command += (
            params.gradient_gain * weight * _action(sigma_distance, d_alpha) * direction
        )
        command += (
            params.alignment_gain
            * weight
            * (state.velocities[j] - state.velocities[index])
        )

    # Navigational feedback, both terms: position toward the waypoint (c1) and
    # velocity toward the reference velocity, here zero (c2).
    command += _migration(state, index, params)
    command -= params.navigation_damping * state.velocities[index]
    command += _obstacle_repulsion(state, index, params)
    return _limit(command, params.max_acceleration)


def acceleration(state: FlockState, index: int, params: FlockParams) -> np.ndarray:
    """Acceleration-based bird-inspired flocking (Iacone et al. 2024, model family).

    Four terms, all acting on acceleration:

    * **self-propulsion** drives the agent toward its cruise speed along its
      current heading, so the flock keeps flying rather than settling;
    * **drag** removes energy proportionally to speed, which bounds the group
      velocity without a hard clamp and damps the oscillations that pure
      spring-like attraction produces;
    * **pairwise potential** attracts beyond the reference distance and repels
      sharply inside it, softened near zero so a near-collision produces a
      large but finite command;
    * **velocity alignment** over the same neighbourhood.

    The self-propulsion/drag pair is what distinguishes this from
    velocity-reference flocking: the controller never commands a velocity the
    vehicle cannot achieve, and momentum carries the flock through turns.
    """
    neighbours = _neighbours(state, index, params)
    velocity = state.velocities[index]
    speed = float(np.linalg.norm(velocity))

    if speed > 1e-6:
        heading = velocity / speed
    else:  # start-up: pick a deterministic direction from the agent index
        heading = np.zeros(state.dimension)
        heading[index % state.dimension] = 1.0

    # self-propulsion toward cruise speed, and drag against the current velocity
    command = params.propulsion_gain * (params.cruise_speed - speed) * heading
    command -= params.drag_gain * velocity

    for j in neighbours:
        offset = state.positions[j] - state.positions[index]
        distance = float(np.linalg.norm(offset))
        if distance < 1e-9:
            continue
        direction = offset / distance
        # Smooth well at `reference_distance`: attractive outside, repulsive in.
        gap = distance - params.reference_distance
        softened = params.potential_softening + max(distance - params.separation_distance, 0.0)
        magnitude = params.potential_gain * gap / (softened ** 2 + 1.0)
        if distance < params.separation_distance:
            magnitude -= params.separation_gain / max(distance, 0.2) ** 2
        command += magnitude * direction
        command += params.alignment_gain * (state.velocities[j] - velocity) / max(1, len(neighbours))

    command += _migration(state, index, params)
    command += _obstacle_repulsion(state, index, params)
    return _limit(command, params.max_acceleration)


CONTROLLERS: Dict[str, Callable[[FlockState, int, FlockParams], np.ndarray]] = {
    "boids": boids,
    "vicsek": vicsek,
    "olfati_saber": olfati_saber,
    "acceleration": acceleration,
}


def get_controller(name):
    if callable(name):
        return name
    try:
        return CONTROLLERS[name]
    except KeyError:
        raise ValueError(
            "Unknown flocking controller %r. Available: %s"
            % (name, ", ".join(sorted(CONTROLLERS)))
        )


# --------------------------------------------------------------------------
# simulation and metrics
# --------------------------------------------------------------------------


@dataclass
class FlockMetrics:
    """What "good flocking" means, measured rather than eyeballed.

    ``order`` is the Vicsek polarisation -- the normalised mean heading, 1.0
    for a perfectly aligned flock. Safety is reported twice on purpose: the
    steady-state minimum separation is what the controller achieves, while the
    transient minimum includes the start-up phase, where a dense spawn can put
    agents inside each other's separation distance before any control has acted.
    Reporting only the second number blames the controller for the initial
    condition; reporting only the first hides real start-up risk.
    """

    order: List[float] = field(default_factory=list)
    cohesion: List[float] = field(default_factory=list)
    min_distance: List[float] = field(default_factory=list)
    speed: List[float] = field(default_factory=list)
    collisions: int = 0
    steady_collisions: int = 0

    def summary(self) -> Dict[str, float]:
        half = max(0, len(self.order) // 2)
        tail = slice(half, None)
        return {
            "order": float(np.mean(self.order[tail])) if self.order else 0.0,
            "cohesion": float(np.mean(self.cohesion[tail])) if self.cohesion else 0.0,
            "min_distance": (
                float(np.min(self.min_distance[half:])) if self.min_distance else 0.0
            ),
            "min_distance_transient": (
                float(np.min(self.min_distance)) if self.min_distance else 0.0
            ),
            "mean_speed": float(np.mean(self.speed[tail])) if self.speed else 0.0,
            "collisions": self.collisions,
            "steady_collisions": self.steady_collisions,
        }


def _measure(
    state: FlockState, params: FlockParams, metrics: FlockMetrics, steady: bool = False
) -> None:
    speeds = np.linalg.norm(state.velocities, axis=1)
    total = np.linalg.norm(state.velocities.sum(axis=0))
    denominator = float(speeds.sum())
    metrics.order.append(float(total / denominator) if denominator > 1e-9 else 0.0)
    centre = state.positions.mean(axis=0)
    metrics.cohesion.append(float(np.mean(np.linalg.norm(state.positions - centre, axis=1))))
    metrics.speed.append(float(speeds.mean()))

    if state.n > 1:
        deltas = state.positions[:, None, :] - state.positions[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        np.fill_diagonal(distances, np.inf)
        closest = float(distances.min())
        metrics.min_distance.append(closest)
        if closest < params.separation_distance:
            metrics.collisions += 1
            if steady:
                metrics.steady_collisions += 1


def simulate(
    controller="acceleration",
    n_agents: int = 20,
    steps: int = 400,
    dt: float = 0.1,
    dimension: int = 2,
    params: Optional[FlockParams] = None,
    initial: Optional[FlockState] = None,
    spawn_radius: float = 8.0,
) -> Tuple[List[FlockState], FlockMetrics]:
    """Run a flock and record its trajectory and metrics.

    Returns ``(history, metrics)`` where ``history[t]`` is the state at step
    ``t`` -- the same shape :mod:`pymapf.viz` animates for MAPF solutions, so
    the swarm side gets the same plots for free.
    """
    params = params or FlockParams()
    rng = np.random.default_rng(params.seed)
    control = get_controller(controller)

    if initial is None:
        state = FlockState(
            _lattice_spawn(n_agents, dimension, params, rng, spawn_radius),
            rng.normal(0, 0.5, size=(n_agents, dimension)),
        )
    else:
        state = initial.copy()

    history = [state.copy()]
    metrics = FlockMetrics()
    _measure(state, params, metrics)

    for _ in range(steps):
        commands = np.zeros_like(state.velocities)
        for index in range(state.n):
            commands[index] = control(state, index, params)
        if params.noise:
            commands += rng.normal(0, params.noise, size=commands.shape)

        # Semi-implicit Euler: velocity first, then position. It is the
        # integrator a flight controller effectively runs, and it does not
        # inject the energy that explicit Euler does.
        state.velocities += commands * dt
        speeds = np.linalg.norm(state.velocities, axis=1, keepdims=True)
        scale = np.minimum(1.0, params.max_speed / np.maximum(speeds, 1e-9))
        state.velocities *= scale
        state.positions += state.velocities * dt

        history.append(state.copy())
        _measure(state, params, metrics, steady=len(history) > steps // 2)

    return history, metrics


def _lattice_spawn(
    n_agents: int,
    dimension: int,
    params: FlockParams,
    rng,
    spawn_radius: float,
) -> np.ndarray:
    """Spawn on a jittered lattice spaced by the reference distance.

    A uniform random spawn puts some pairs closer than the separation distance
    before the controller has run a single step, so every controller then starts
    in violation and the safety metric measures the spawn instead of the
    control law. A lattice starts every run legal.
    """
    spacing = max(params.reference_distance, params.separation_distance * 1.2)
    per_side = int(math.ceil(n_agents ** (1.0 / dimension)))
    coordinates = np.indices((per_side,) * dimension).reshape(dimension, -1).T[:n_agents]
    positions = coordinates.astype(float) * spacing
    positions -= positions.mean(axis=0)
    jitter = rng.normal(0, spacing * 0.05, size=positions.shape)
    return positions + jitter
