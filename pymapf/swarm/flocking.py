"""Flocking behaviors, as swappable objects.

Ten control laws spanning forty years of the field, all with the same
interface, all runnable on identical initial conditions. That is the point: the
interesting question is never "does this flock?" but "what does this one do
that the previous one did not?", and that question needs a controlled
comparison.

The lineage
-----------

``boids``
    Reynolds (1987). Separation, alignment, cohesion as steering forces. The
    origin of everything below.

``vicsek``
    Vicsek et al. (1995). Heading alignment with noise at constant speed. No
    attraction at all, which is why it aligns perfectly and disperses.

``cucker_smale``
    Cucker and Smale (2007). Velocity consensus with an influence that decays
    as a power of distance -- the model with the cleanest convergence theory.

``olfati_saber``
    Olfati-Saber (2006). Gradient of a smooth potential plus velocity
    consensus; converges to an alpha-lattice, with stability proofs.

``proximal``
    Vasarhelyi et al. (2018)-style proximal control: pairwise repulsion, a
    velocity-alignment term with a distance-dependent maximum, and explicit
    speed regulation. The line of work behind real outdoor drone flocks.

``active_elastic``
    Ferrante et al. (2012, 2013). The swarm as an *elastic solid*: each pair is
    a spring at rest at the reference distance, and self-propulsion cascades
    energy into the lowest-energy mode, which is uniform translation. Alignment
    is never computed -- it emerges. That makes it the one model here that needs
    no velocity sensing at all, which is exactly why it works on cheap robots.

``acceleration``
    Iacone, Lejeune, Manoni, Manfredi and Albani (2024). Self-propulsion and
    drag set a cruise speed; pairwise potential and alignment shape the group.
    Commanding acceleration rather than velocity is what keeps the request
    inside what a multirotor's inner loop can execute.

``gaussian_kernel``
    Manoni, Albani et al. (2022). The interaction *weight* is a Gaussian kernel
    of distance, and its width arbitrates between local and global coherence.
    Implemented as a behavior paired with
    :class:`~pymapf.swarm.neighborhood.GaussianKernelNeighborhood`, so the
    weighting is available to any other behavior too.

``distributed_3d``
    Albani, Manoni, Saska and Ferrante (2022). Proximal control made
    *anisotropic*, because a multirotor is: climbing is expensive, and a drone
    below another sits in its downwash. The vertical axis gets its own weighting
    and its own safety term, and the swarm settles into a flat wide lattice
    instead of a ball.

``minimalistic``
    Amorim, Nascimento, Chaudhary, Ferrante and Saska (2024). The floor of the
    field: range and bearing to neighbours, nothing else -- no GPS, no compass,
    no communication, no velocity sensing -- and a cohesive flock still emerges
    and agrees on a direction nobody transmitted. Active-elastic dynamics with a
    Lennard-Jones coupling in place of the spring.

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
  and theory.* IEEE TAC 51(3): 401-420.
* Ferrante, E.; Turgut, A. E.; Huepe, C.; Stranieri, A.; Pinciroli, C.; and
  Dorigo, M. 2012. *Self-organized flocking with a mobile robot swarm: a novel
  motion control method.* Adaptive Behavior 20(6): 460-477.
* Ferrante, E.; Turgut, A. E.; Dorigo, M.; and Huepe, C. 2013.
  *Elasticity-based mechanism for the collective motion of self-propelled
  particles with spring-like interactions.* Physical Review Letters 111(26):
  268302.
* Vasarhelyi, G.; Viragh, C.; Somorjai, G.; Nepusz, T.; Eiben, A. E.; and
  Vicsek, T. 2018. *Optimized flocking of autonomous drones in confined
  environments.* Science Robotics 3(20): eaat3536.
* Manoni, T.; Albani, D.; et al. 2022. *Adaptive arbitration of aerial swarm
  interactions through a Gaussian kernel for coherent group motion.* Frontiers
  in Robotics and AI 9: 1006786.
* Albani, D.; Manoni, T.; Saska, M.; and Ferrante, E. 2022. *Distributed Three
  Dimensional Flocking of Autonomous Drones.* ICRA 2022: 6904-6911.
* Iacone, L.; Lejeune, E.; Manoni, T.; Manfredi, S.; and Albani, D. 2024.
  *Decentralized acceleration-based bird-inspired flocking.* IROS 2024.
* Amorim, T.; Nascimento, T.; Chaudhary, A.; Ferrante, E.; and Saska, M. 2024.
  *A Minimalistic 3D Self-Organized UAV Flocking Approach for Desert
  Exploration.* Journal of Intelligent & Robotic Systems 110: 75.
* Ballerini, M.; et al. 2008. *Interaction ruling animal collective behavior
  depends on topological rather than metric distance.* PNAS 105(4): 1232-1237.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .base import Behavior, SwarmState, limit, register_behavior
from .neighborhood import GaussianKernelNeighborhood, TopologicalNeighborhood

__all__ = [
    "Boids",
    "Vicsek",
    "CuckerSmale",
    "OlfatiSaber",
    "ProximalControl",
    "ActiveElastic",
    "AccelerationFlocking",
    "GaussianKernelFlocking",
    "MinimalisticFlocking",
    "DistributedThreeDimensional",
]


@register_behavior("boids")
class Boids(Behavior):
    """Reynolds' three rules as accelerations."""

    def __init__(
        self,
        separation_gain: float = 6.0,
        cohesion_gain: float = 1.2,
        alignment_gain: float = 2.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.separation_gain = separation_gain
        self.cohesion_gain = cohesion_gain
        self.alignment_gain = alignment_gain

    def command(self, state: SwarmState, index: int) -> np.ndarray:
        neighbours = self.neighbors(state, index)
        command = np.zeros(state.dimension)
        if len(neighbours):
            offsets = state.positions[neighbours] - state.positions[index]
            distances = np.maximum(np.linalg.norm(offsets, axis=1, keepdims=True), 1e-6)
            close = distances[:, 0] < self.params.separation_distance
            if np.any(close):
                command -= self.separation_gain * np.sum(
                    offsets[close] / distances[close] ** 2, axis=0
                )
            command += self.cohesion_gain * offsets.mean(axis=0)
            command += self.alignment_gain * (
                state.velocities[neighbours].mean(axis=0) - state.velocities[index]
            )
        return self.finalise(command, state, index)


@register_behavior("vicsek")
class Vicsek(Behavior):
    """Constant-speed heading alignment (Vicsek et al. 1995).

    Expressed as an acceleration so it shares the integrator with everything
    else: the command is whatever turns the current velocity toward the local
    mean heading within one step.
    """

    def __init__(self, alignment_gain: float = 2.5, **kwargs):
        super().__init__(**kwargs)
        self.alignment_gain = alignment_gain

    def command(self, state: SwarmState, index: int) -> np.ndarray:
        neighbours = self.neighbors(state, index)
        velocities = state.velocities[np.append(neighbours, index).astype(int)]
        mean = velocities.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        heading = (
            state.velocities[index]
            if norm < 1e-9
            else mean / norm * self.params.cruise_speed
        )
        return self.finalise(
            self.alignment_gain * (heading - state.velocities[index]), state, index
        )


@register_behavior("cucker_smale")
class CuckerSmale(Behavior):
    """Velocity consensus with a power-law influence (Cucker and Smale 2007).

    ``a_i = K * sum_j (1 + |x_i - x_j|^2)^(-beta) (v_j - v_i)``

    The theory is the reason to have it: for ``beta <= 1/2`` the flock converges
    to a common velocity from *any* initial condition. It has no repulsion, so
    it is a consensus law rather than a complete flocking controller -- compose
    it with a spacing behavior for anything that must not collide.
    """

    def __init__(self, strength: float = 8.0, beta: float = 0.4, **kwargs):
        super().__init__(**kwargs)
        self.strength = strength
        self.beta = beta

    def command(self, state: SwarmState, index: int) -> np.ndarray:
        neighbours = self.neighbors(state, index)
        command = np.zeros(state.dimension)
        if len(neighbours):
            offsets = state.positions[neighbours] - state.positions[index]
            squared = np.sum(offsets ** 2, axis=1)
            influence = self.strength / np.power(1.0 + squared, self.beta)
            differences = state.velocities[neighbours] - state.velocities[index]
            command = (influence[:, None] * differences).sum(axis=0) / max(1, len(neighbours))
        return self.finalise(command, state, index)


def _sigma_norm(vector: np.ndarray, epsilon: float = 0.1) -> float:
    """Olfati-Saber's smooth norm: differentiable everywhere, including zero."""
    return (math.sqrt(1 + epsilon * float(vector @ vector)) - 1) / epsilon


def _bump(z: float, h: float = 0.2) -> float:
    if z < 0:
        return 0.0
    if z < h:
        return 1.0
    if z <= 1.0:
        return 0.5 * (1 + math.cos(math.pi * (z - h) / (1 - h)))
    return 0.0


def _action(z: float, d_alpha: float, a: float = 5.0, b: float = 5.0) -> float:
    c = abs(a - b) / math.sqrt(4 * a * b) if a * b > 0 else 0.0
    shifted = z - d_alpha
    sigma_1 = (shifted + c) / math.sqrt(1 + (shifted + c) ** 2)
    return 0.5 * ((a + b) * sigma_1 + (a - b))


@register_behavior("olfati_saber")
class OlfatiSaber(Behavior):
    """Gradient flocking with the paper's action and bump functions.

    Two properties are worth knowing before deploying it:

    * The interaction range is **part of the algorithm**: the alpha-lattice is
      defined for ``r ~ 1.2 d``. At ``r/d = 2`` the lattice collapses outright
      (summed attraction from distant neighbours beats local repulsion), so this
      class clamps the range rather than letting a caller silently break it.
    * The sigma-space gradient direction vanishes at zero separation, so
      repulsion peaks at intermediate range and *fades* at contact. A pair driven
      together by other terms can merge and stay merged.
    """

    def __init__(
        self,
        gradient_gain: float = 0.6,
        alignment_gain: float = 2.5,
        navigation_damping: float = 0.9,
        epsilon: float = 0.1,
        range_ratio: float = 1.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gradient_gain = gradient_gain
        self.alignment_gain = alignment_gain
        self.navigation_damping = navigation_damping
        self.epsilon = epsilon
        self.range_ratio = range_ratio

    def command(self, state: SwarmState, index: int) -> np.ndarray:
        params = self.params
        neighbours = self.neighbors(state, index)
        command = np.zeros(state.dimension)

        interaction_range = min(
            params.sensing_range, self.range_ratio * params.reference_distance
        )
        zeros = [0.0] * (state.dimension - 1)
        d_alpha = _sigma_norm(np.array([params.reference_distance] + zeros), self.epsilon)
        r_alpha = _sigma_norm(np.array([interaction_range] + zeros), self.epsilon)

        for j in neighbours:
            offset = state.positions[j] - state.positions[index]
            sigma_distance = _sigma_norm(offset, self.epsilon)
            direction = offset / math.sqrt(1 + self.epsilon * float(offset @ offset))
            weight = _bump(sigma_distance / r_alpha) if r_alpha > 0 else 0.0
            command += self.gradient_gain * weight * _action(sigma_distance, d_alpha) * direction
            command += self.alignment_gain * weight * (
                state.velocities[j] - state.velocities[index]
            )

        # Navigational feedback, both terms: position (c1) and velocity (c2).
        command += self.migration(state, index)
        command -= self.navigation_damping * state.velocities[index]
        command += self.obstacle_avoidance(state, index)
        return limit(command, params.max_acceleration)


@register_behavior("proximal")
class ProximalControl(Behavior):
    """Proximal-control flocking in the Vasarhelyi et al. (2018) style.

    Three explicitly separated terms, which is what makes this family tunable
    for real vehicles:

    * a **proximal potential** -- a Lennard-Jones-style force that is repulsive
      inside the reference distance and attractive outside it, which is what
      "proximal control" means in the swarm-robotics line of work: an agent
      needs only range-and-bearing to its neighbours, no shared frame;
    * **velocity alignment**, but only in excess of a distance-dependent
      allowance -- neighbours far apart are permitted to differ, so the swarm is
      not over-constrained. The allowance grows like ``sqrt`` of the free
      distance, the braking profile a vehicle with a fixed deceleration limit
      can actually achieve;
    * **speed regulation** toward the cruise speed.
    """

    def __init__(
        self,
        repulsion_gain: float = 4.0,
        alignment_gain: float = 2.0,
        speed_gain: float = 1.0,
        slack: float = 0.5,
        exponent: float = 2.0,
        max_deceleration: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.repulsion_gain = repulsion_gain
        self.alignment_gain = alignment_gain
        self.speed_gain = speed_gain
        self.slack = slack
        self.exponent = exponent
        self.max_deceleration = max_deceleration

    def equilibrium_sigma(self) -> float:
        """The ``sigma`` that puts the potential's minimum at the reference distance.

        A Lennard-Jones potential written with length parameter ``sigma`` has its
        force zero at ``2^(1/m) sigma``, not at ``sigma`` -- a factor of 1.41 for
        the default exponent. Passing the reference distance in as ``sigma``
        directly therefore builds a controller whose rest spacing is 41% wider
        than the one it was configured with, and in open space that difference
        compounds: the lattice expands, the outer agents cross the interaction
        range, and the flock sheds them. Solving for the minimum instead makes
        ``reference_distance`` mean what it says.
        """
        return self.params.reference_distance / (2.0 ** (1.0 / self.exponent))

    def proximal(self, distance: float) -> float:
        """Lennard-Jones-style magnitude: <0 repels, >0 attracts, 0 at rest.

        ``-4 e / d * (2 (s/d)^2m - (s/d)^m)``, with ``s`` chosen by
        :meth:`equilibrium_sigma` so the zero crossing lands on the reference
        distance. Bounded so a near-collision produces a large but finite
        command.
        """
        ratio = (self.equilibrium_sigma() / max(distance, 1e-6)) ** self.exponent
        magnitude = -4 * self.repulsion_gain / max(distance, 1e-6) * (2 * ratio ** 2 - ratio)
        return float(np.clip(magnitude, -4 * self.params.max_acceleration, self.params.max_acceleration))

    def _allowance(self, distance: float) -> float:
        """Velocity difference tolerated at this separation (smooth ramp)."""
        deceleration = self.max_deceleration or self.params.max_acceleration
        free = max(0.0, distance - self.params.separation_distance)
        if free <= 0:
            return 0.0
        return math.sqrt(2 * deceleration * free) * self.slack

    def command(self, state: SwarmState, index: int) -> np.ndarray:
        params = self.params
        neighbours = self.neighbors(state, index)
        command = np.zeros(state.dimension)

        for j in neighbours:
            offset = state.positions[j] - state.positions[index]
            distance = float(np.linalg.norm(offset))
            if distance < 1e-9:
                continue
            direction = offset / distance

            # Negative magnitude pushes away, positive pulls in.
            command += self.proximal(distance) * direction

            difference = state.velocities[j] - state.velocities[index]
            excess = float(np.linalg.norm(difference)) - self._allowance(distance)
            if excess > 0:
                command += self.alignment_gain * excess * difference / max(
                    float(np.linalg.norm(difference)), 1e-9
                )

        # Speed regulation toward the cruise speed along the current heading.
        velocity = state.velocities[index]
        speed = float(np.linalg.norm(velocity))
        if speed > 1e-6:
            command += self.speed_gain * (params.cruise_speed - speed) * (velocity / speed)

        return self.finalise(command, state, index)


@register_behavior("active_elastic")
class ActiveElastic(Behavior):
    """The swarm as an active elastic solid (Ferrante et al. 2012, 2013).

    Each pair is a linear spring at rest at the reference distance, and each
    agent self-propels along its own heading. No agent ever measures anyone's
    velocity or heading: alignment is not computed, it *emerges*, because
    self-propulsion energy cascades into the lowest-energy elastic mode and that
    mode is uniform translation.

    That property is the practical point. Heading sensing is the expensive part
    of a flocking robot -- it needs a compass, a shared frame, or inter-agent
    communication -- and this model needs none of it. Each agent measures only
    the *relative positions* of its neighbours.

    The model is first order, so this behavior returns a **velocity**, not an
    acceleration (``output = "velocity"``); the simulator integrates it
    accordingly. Following the 2013 formulation, with ``F`` the total spring
    force and ``n`` the agent's heading:

        speed along n  :  v + alpha * (F . n)
        heading rate   :  beta * (F . n_perp)

    In 3D the perpendicular component is taken in the plane spanned by the
    heading and the force, which reduces to the planar rule when the motion is
    planar.
    """

    output = "velocity"

    def __init__(
        self,
        spring_constant: float = 8.0,
        forward_speed: float = 2.0,
        alpha: float = 0.6,
        beta: float = 2.0,
        dt: float = 0.1,
        allow_reverse: bool = True,
        **kwargs,
    ):
        # Springs are two-sided but *bounded*: at zero separation a linear
        # spring pushes with only k, while ten stretched neighbours pull with
        # ~10k. With a wide metric radius the lattice is therefore crushed --
        # the same failure mode as Olfati-Saber at r/d = 2. A topological rule
        # fixes it by capping how many neighbours can pull at once, and it is
        # also what the robots in the original work actually had.
        kwargs.setdefault("neighborhood", TopologicalNeighborhood(k=3))
        super().__init__(**kwargs)
        self.spring_constant = spring_constant
        self.forward_speed = forward_speed
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.allow_reverse = allow_reverse
        self._headings: Optional[np.ndarray] = None

    def reset(self, state: SwarmState) -> None:
        speeds = np.linalg.norm(state.velocities, axis=1, keepdims=True)
        headings = np.where(speeds > 1e-6, state.velocities / np.maximum(speeds, 1e-9), 0.0)
        # Agents that start at rest need *some* heading; a deterministic fan
        # keeps runs reproducible without pointing them all the same way.
        for i in range(state.n):
            if np.linalg.norm(headings[i]) < 1e-6:
                angle = 2 * math.pi * i / max(1, state.n)
                vector = np.zeros(state.dimension)
                vector[0] = math.cos(angle)
                vector[1] = math.sin(angle)
                headings[i] = vector
        self._headings = headings

    def _heading(self, state: SwarmState, index: int) -> np.ndarray:
        if self._headings is None or len(self._headings) != state.n:
            self.reset(state)
        return self._headings[index]

    def elastic_force(self, state: SwarmState, index: int) -> np.ndarray:
        """Sum of spring forces from the neighbours -- the only thing sensed.

        The ``1/l`` scaling is from the paper: it makes the response depend on
        the *relative* extension, so the same gains work at any chosen spacing.
        """
        neighbours = self.neighbors(state, index)
        rest = max(self.params.reference_distance, 1e-6)
        force = np.zeros(state.dimension)
        for j in neighbours:
            offset = state.positions[j] - state.positions[index]
            distance = float(np.linalg.norm(offset))
            if distance < 1e-9:
                continue
            extension = distance - rest  # positive = stretched = attractive
            force += (self.spring_constant / rest) * extension * (offset / distance)
        return force

    def command(self, state: SwarmState, index: int) -> np.ndarray:
        heading = self._heading(state, index)
        force = self.elastic_force(state, index)
        force = force + self.migration(state, index) + self.obstacle_avoidance(state, index)

        along = float(force @ heading)
        perpendicular = force - along * heading

        # Rotate the heading by beta * (F . n_perp) * dt, in the plane spanned by
        # the heading and the force.
        magnitude = float(np.linalg.norm(perpendicular))
        if magnitude > 1e-9:
            axis = perpendicular / magnitude
            angle = self.beta * magnitude * self.dt
            angle = float(np.clip(angle, -0.5, 0.5))  # one step cannot spin an agent
            turned = math.cos(angle) * heading + math.sin(angle) * axis
            norm = float(np.linalg.norm(turned))
            if norm > 1e-9:
                self._headings[index] = turned / norm
                heading = self._headings[index]

        speed = self.forward_speed + self.alpha * along
        # A differential-drive robot can reverse, and forbidding it is what
        # leaves an agent stuck facing into a crowd it is being pushed out of.
        floor = -0.5 * self.forward_speed if self.allow_reverse else 0.0
        speed = float(np.clip(speed, floor, self.params.max_speed))
        return speed * heading


@register_behavior("acceleration")
class AccelerationFlocking(Behavior):
    """Acceleration-based bird-inspired flocking (Iacone et al. 2024).

    Self-propulsion drives each agent toward its cruise speed along its current
    heading, drag removes energy proportionally to speed, a softened pairwise
    potential attracts beyond the reference distance and repels sharply inside
    it, and velocity alignment averages over the neighbourhood.

    The self-propulsion/drag pair is the distinguishing feature: the flock keeps
    momentum through turns and keeps *flying* with no waypoint to chase, which
    is what a velocity-reference controller cannot do.

    The paper's own parameter values were not accessible when this was written;
    the defaults here were tuned for stable flight at 10 Hz and are a starting
    point, not a reproduction of its results.
    """

    def __init__(
        self,
        propulsion_gain: float = 1.6,
        drag_gain: float = 0.4,
        potential_gain: float = 3.0,
        potential_softening: float = 0.35,
        separation_gain: float = 6.0,
        alignment_gain: float = 2.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.propulsion_gain = propulsion_gain
        self.drag_gain = drag_gain
        self.potential_gain = potential_gain
        self.potential_softening = potential_softening
        self.separation_gain = separation_gain
        self.alignment_gain = alignment_gain

    def command(self, state: SwarmState, index: int) -> np.ndarray:
        params = self.params
        velocity = state.velocities[index]
        speed = float(np.linalg.norm(velocity))

        if speed > 1e-6:
            heading = velocity / speed
        else:  # start-up: a deterministic direction from the agent index
            heading = np.zeros(state.dimension)
            heading[index % state.dimension] = 1.0

        command = self.propulsion_gain * (params.cruise_speed - speed) * heading
        command -= self.drag_gain * velocity

        neighbours = self.neighbors(state, index)
        weights = self.neighborhood.weights(state, index, params)
        for offset_index, j in enumerate(neighbours):
            offset = state.positions[j] - state.positions[index]
            distance = float(np.linalg.norm(offset))
            if distance < 1e-9:
                continue
            direction = offset / distance
            weight = float(weights[offset_index]) if len(weights) else 1.0

            gap = distance - params.reference_distance
            softened = self.potential_softening + max(distance - params.separation_distance, 0.0)
            magnitude = self.potential_gain * gap / (softened ** 2 + 1.0)
            if distance < params.separation_distance:
                magnitude -= self.separation_gain / max(distance, 0.2) ** 2
            command += weight * magnitude * direction
            command += (
                self.alignment_gain
                * weight
                * (state.velocities[j] - velocity)
                / max(1, len(neighbours))
            )

        return self.finalise(command, state, index)


@register_behavior("gaussian_kernel")
class GaussianKernelFlocking(AccelerationFlocking):
    """Gaussian-kernel arbitration of swarm interactions (Manoni et al. 2022).

    Same control terms as :class:`AccelerationFlocking`, but every neighbour's
    contribution is weighted by ``exp(-d^2 / 2 sigma^2)``. The kernel width is
    the arbitration knob: small ``sigma`` gives locally coherent sub-groups,
    large ``sigma`` a single coherent flock, and because the weight is smooth,
    an agent crossing the sensing boundary never causes a discontinuity in
    anyone's command.

    ``adaptive=True`` sets ``sigma`` from the swarm's own spread each step, so
    the group re-arbitrates as it compresses and expands rather than holding a
    width tuned for one density.
    """

    def __init__(self, sigma: Optional[float] = None, adaptive: bool = False, **kwargs):
        kwargs.setdefault("neighborhood", GaussianKernelNeighborhood(sigma=sigma))
        super().__init__(**kwargs)
        self.sigma = sigma
        self.adaptive = adaptive

    def command(self, state: SwarmState, index: int) -> np.ndarray:
        if self.adaptive and isinstance(self.neighborhood, GaussianKernelNeighborhood):
            # Median nearest-neighbour distance is a robust density estimate:
            # it does not chase a single outlier the way the mean does.
            spread = np.median(
                [np.min(state.distances_from(i)) for i in range(state.n)]
            )
            self.neighborhood.sigma = max(
                self.params.separation_distance, float(spread) * 1.5
            )
        return super().command(state, index)


@register_behavior("minimalistic")
class MinimalisticFlocking(ActiveElastic):
    """Minimalistic 3D self-organized flocking (Amorim et al. 2024).

    The question this model answers is how *little* a drone needs in order to
    flock. The answer, from nine UAVs flown over a desert with no GPS, no
    external localisation and no radio exchange of headings, is: relative
    **range and bearing to neighbours, and nothing else**. No velocity sensing,
    no compass, no shared frame, no communication -- and the group still
    converges to a cohesive flock travelling in a common direction, which it
    picks itself. That emergent direction is the whole result: with no agent
    ever transmitting where it is going, agreement can only come from the
    dynamics, and it does.

    Mechanically this is the active-elastic mechanism of
    :class:`ActiveElastic` -- self-propulsion plus a passive coupling, with
    alignment as an emergent property rather than a control term -- with the
    linear spring replaced by a **Lennard-Jones proximal potential**:

        ``f(d) = -4 e / d * (2 (sigma/d)^(2m) - (sigma/d)^m)``

    Two things change as a result, and both matter on real hardware. The
    potential is *bounded in attraction*: a neighbour that drifts far away pulls
    with a force that decays instead of growing without limit, so a straggler
    cannot drag the flock apart the way a linear spring does. And it is
    *unbounded in repulsion* as separation goes to zero, so the safety margin is
    enforced by the potential itself rather than by a separate avoidance rule --
    which is exactly what you want when the failure mode is a mid-air collision.

    The trade is that the flock is softer. A Lennard-Jones well is shallow away
    from its minimum, so cohesion is looser and convergence slower than the
    spring model; that is the cost of not letting one distant agent dominate.

    Args:
        epsilon: well depth -- the strength of the proximal coupling.
        exponent: ``m`` above. Larger makes the well narrower and stiffer.
        cutoff: neighbours beyond this multiple of the reference distance are
            ignored entirely. The potential is already negligible there, and
            truncating it keeps the interaction strictly local, as on the
            robots.

    Connectivity, and the one place this model is fragile
    -----------------------------------------------------

    The default neighbourhood is topological with ``k = 8``, not the ``k = 3``
    inherited from :class:`ActiveElastic`, and the difference is not cosmetic. A
    bounded attraction needs more incident edges than a spring does to hold a
    group together: once the interaction graph pinches, the two halves stop
    pulling on each other and the bounded force never gets them back. Over ten
    seeds with twenty agents in the plane, the flock fragmented in 7 runs at
    ``k = 6``, 3 at ``k = 8`` with the original cutoff, and 1 at the defaults
    here; mean order rose from 0.76 to 0.94.

    In three dimensions -- which is what the paper is about -- none of this
    bites: twenty agents form a compact blob whose topological graph never
    pinches, and every seed converges (order 0.95 to 0.98, cohesion 3.2, no
    collisions). The planar fragility is worth stating plainly rather than
    tuning away: it is the price of a decaying attraction, and it is why the
    method is presented in 3D.

    For reference, starlings were measured to track six to seven neighbours
    (Ballerini et al. 2008), so the topological rule itself is well supported;
    the specific ``k`` here was chosen by measurement, not by the birds.

    References:
        Amorim, T.; Nascimento, T.; Chaudhary, A.; Ferrante, E.; and Saska, M.
        2024. *A Minimalistic 3D Self-Organized UAV Flocking Approach for
        Desert Exploration.* Journal of Intelligent & Robotic Systems 110: 75.

        Ballerini, M.; et al. 2008. *Interaction ruling animal collective
        behavior depends on topological rather than metric distance.* PNAS
        105(4): 1232-1237.
    """

    def __init__(
        self,
        epsilon: float = 3.0,
        exponent: float = 2.0,
        cutoff: float = 4.0,
        **kwargs,
    ):
        kwargs.setdefault("spring_constant", 0.0)  # replaced by the potential
        kwargs.setdefault("neighborhood", TopologicalNeighborhood(k=8))
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.exponent = exponent
        self.cutoff = cutoff

    def equilibrium_sigma(self) -> float:
        """``sigma`` placing the potential's minimum at the reference distance."""
        return self.params.reference_distance / (2.0 ** (1.0 / self.exponent))

    def proximal(self, distance: float) -> float:
        """Signed magnitude of the proximal force: <0 repels, >0 attracts."""
        sigma = max(self.equilibrium_sigma(), 1e-6)
        ratio = (sigma / max(distance, 1e-6)) ** self.exponent
        magnitude = (
            -4.0 * self.epsilon / max(distance, 1e-6) * (2.0 * ratio ** 2 - ratio)
        )
        # Finite even at contact: an unbounded command is not executable, and a
        # NaN propagates through the whole swarm.
        bound = 4.0 * self.params.max_acceleration
        return float(np.clip(magnitude, -bound, bound))

    def elastic_force(self, state: SwarmState, index: int) -> np.ndarray:
        """Proximal force in place of the spring force -- the only thing sensed."""
        limit_distance = self.cutoff * max(self.params.reference_distance, 1e-6)
        force = np.zeros(state.dimension)
        for j in self.neighbors(state, index):
            offset = state.positions[j] - state.positions[index]
            distance = float(np.linalg.norm(offset))
            if distance < 1e-9 or distance > limit_distance:
                continue
            force += self.proximal(distance) * (offset / distance)
        return force


@register_behavior("distributed_3d")
class DistributedThreeDimensional(ProximalControl):
    """Distributed 3D flocking for multirotors (Albani, Manoni et al. 2022).

    Lifting a planar flocking law into 3D by swapping ``dimension=2`` for
    ``dimension=3`` quietly assumes the vehicle is isotropic. A multirotor is
    not. Climbing costs far more than translating, the vertical axis has the
    tightest speed limit of the three, and -- the part that has no planar
    analogue at all -- a drone directly beneath another sits in its **downwash**,
    a disturbance that no amount of horizontal separation removes.

    So this model treats the vertical axis as its own control problem:

    * the proximal potential is evaluated on an **anisotropic** distance, with
      vertical offsets weighted by ``vertical_scale``. Above 1 the swarm reads
      vertical separation as larger than it is, so a vertically-adjacent pair
      settles at ``reference_distance / vertical_scale`` instead of
      ``reference_distance`` and the lattice comes out flat and wide -- the
      shape a multirotor swarm should hold. Because that weighting also softens
      vertical repulsion, an unweighted short-range repulsion on the true
      distance runs alongside it, so the shaping never eats the safety margin;
    * vertical commands are scaled by ``vertical_gain`` (below 1), so the
      controller spends its authority where it is cheap;
    * a neighbour within ``downwash_radius`` horizontally *and* below this agent
      contributes an explicit upward push, because the rotor wake is a real
      force and treating it as an unmodelled disturbance is how a stacked pair
      descends into each other.

    Everything else -- the alignment allowance, the speed regulation -- is
    inherited from :class:`ProximalControl`, which is the point: the 3D
    extension is a set of axis-aware corrections on top of a planar law, not a
    different algorithm.

    Reference:
        Albani, D.; Manoni, T.; Saska, M.; and Ferrante, E. 2022. *Distributed
        Three Dimensional Flocking of Autonomous Drones.* ICRA 2022: 6904-6911.
    """

    def __init__(
        self,
        vertical_scale: float = 2.0,
        vertical_gain: float = 0.5,
        downwash_radius: float = 1.0,
        downwash_gain: float = 3.0,
        safety_margin: float = 1.4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vertical_scale = vertical_scale
        self.vertical_gain = vertical_gain
        self.downwash_radius = downwash_radius
        self.downwash_gain = downwash_gain
        self.safety_margin = safety_margin

    def effective_vertical_scale(self) -> float:
        """``vertical_scale``, capped so flattening cannot outrun the safety margin.

        A weighting of ``s`` puts the vertical equilibrium at
        ``reference_distance / s``. Ask for enough flattening and that number
        drops below ``separation_distance`` -- the controller is then actively
        holding the swarm at a spacing it also calls a collision, and no amount
        of repulsion gain resolves a contradiction between two terms of the same
        law. So the request is honoured only down to
        ``safety_margin * separation_distance`` and silently capped past it: a
        flatter-than-safe lattice is not a tuning choice worth offering.
        """
        floor = self.safety_margin * max(self.params.separation_distance, 1e-6)
        ceiling = max(1.0, self.params.reference_distance / floor)
        return float(min(self.vertical_scale, ceiling))

    def _weighted(self, offset: np.ndarray) -> np.ndarray:
        """Offset with the vertical component stretched for the potential."""
        scale = self.effective_vertical_scale()
        if len(offset) < 3 or scale == 1.0:
            return offset
        weighted = offset.copy()
        weighted[2] *= scale
        return weighted

    def downwash(self, state: SwarmState, index: int) -> np.ndarray:
        """Upward push away from any neighbour this agent is sitting on top of."""
        command = np.zeros(state.dimension)
        if state.dimension < 3:
            return command
        for j in self.neighbors(state, index):
            offset = state.positions[j] - state.positions[index]
            horizontal = float(np.linalg.norm(offset[:2]))
            if horizontal < self.downwash_radius and offset[2] < 0:
                # The neighbour is below: this agent's wake hits it, so climb.
                depth = max(0.0, self.params.separation_distance + offset[2])
                command[2] += self.downwash_gain * depth
        return command

    def command(self, state: SwarmState, index: int) -> np.ndarray:
        params = self.params
        command = np.zeros(state.dimension)

        for j in self.neighbors(state, index):
            offset = state.positions[j] - state.positions[index]
            distance = float(np.linalg.norm(offset))
            if distance < 1e-9:
                continue
            # Anisotropic: the potential is evaluated on the *weighted* distance
            # but applied along the true direction. A purely vertical pair reads
            # its separation as ``vertical_scale`` times larger, so its
            # equilibrium lands at ``reference / vertical_scale`` -- closer --
            # and the lattice settles flat, which is the shape a multirotor
            # swarm should hold.
            #
            # The weighting must apply in the near regime too, because that is
            # where the flattening happens: carving it out "for safety" removes
            # the effect entirely. Safety is restored below by a separate
            # repulsion on the *true* distance, which nothing scales.
            effective = float(np.linalg.norm(self._weighted(offset)))
            command += self.proximal(effective) * (offset / distance)

            if distance < params.separation_distance:
                command -= (
                    self.repulsion_gain
                    * (params.separation_distance - distance)
                    / max(distance, 0.2)
                    * (offset / distance)
                )

            difference = state.velocities[j] - state.velocities[index]
            excess = float(np.linalg.norm(difference)) - self._allowance(distance)
            if excess > 0:
                command += self.alignment_gain * excess * difference / max(
                    float(np.linalg.norm(difference)), 1e-9
                )

        velocity = state.velocities[index]
        speed = float(np.linalg.norm(velocity))
        if speed > 1e-6:
            command += self.speed_gain * (params.cruise_speed - speed) * (velocity / speed)

        # Throttle the vertical axis first, then add downwash: climbing is
        # expensive and worth rationing, but escaping another drone's rotor wake
        # is not the place to economise.
        if state.dimension >= 3:
            command[2] *= self.vertical_gain
        command += self.downwash(state, index)
        return self.finalise(command, state, index)
