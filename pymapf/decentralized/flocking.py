"""Functional façade over :mod:`pymapf.swarm` (kept for backwards compatibility).

The flocking models moved to :mod:`pymapf.swarm.flocking`, where each one is a
:class:`~pymapf.swarm.base.Behavior` object: composable, individually
configurable, and usable with any neighbourhood strategy. This module keeps the
older function-and-parameter-bag interface working, and is a thin delegation --
there is exactly one implementation of every model.

New code should prefer::

    from pymapf.swarm import SwarmSimulator
    result = SwarmSimulator("acceleration", n_agents=20).run(steps=300)

See :mod:`pymapf.swarm.flocking` for the full model list and citations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..swarm.base import SwarmParams, SwarmState, get_behavior
from ..swarm.simulator import SwarmMetrics, SwarmSimulator

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

# The new types are drop-in for the old names.
FlockState = SwarmState
FlockMetrics = SwarmMetrics


@dataclass
class FlockParams(SwarmParams):
    """:class:`~pymapf.swarm.base.SwarmParams` plus the old per-model gains.

    The gains now live on the behavior objects that use them; they are accepted
    here so existing calls keep working, and forwarded to the right behavior.
    """

    separation_gain: float = 6.0
    cohesion_gain: float = 1.2
    alignment_gain: float = 2.5
    propulsion_gain: float = 1.6
    drag_gain: float = 0.4
    potential_gain: float = 3.0
    potential_softening: float = 0.35
    gradient_gain: float = 0.6
    navigation_damping: float = 0.9
    topological_k: Optional[int] = None


_GAINS_BY_MODEL = {
    "boids": ("separation_gain", "cohesion_gain", "alignment_gain"),
    "vicsek": ("alignment_gain",),
    "olfati_saber": ("gradient_gain", "alignment_gain", "navigation_damping"),
    "acceleration": (
        "propulsion_gain",
        "drag_gain",
        "potential_gain",
        "potential_softening",
        "separation_gain",
        "alignment_gain",
    ),
}


def _build(name: str, params: Optional[FlockParams]):
    params = params or FlockParams()
    kwargs = {
        gain: getattr(params, gain)
        for gain in _GAINS_BY_MODEL.get(name, ())
        if hasattr(params, gain)
    }
    if getattr(params, "topological_k", None):
        from ..swarm.neighborhood import TopologicalNeighborhood

        kwargs["neighborhood"] = TopologicalNeighborhood(k=params.topological_k)
    return get_behavior(name, params=params, **kwargs)


def _controller(name: str) -> Callable:
    def call(
        state: SwarmState, index: int, params: Optional[FlockParams] = None
    ) -> np.ndarray:
        behavior = _build(name, params)
        behavior.reset(state)
        return behavior.command(state, index)

    call.__name__ = name
    call.__doc__ = "Functional wrapper for the %r behavior." % name
    return call


boids = _controller("boids")
vicsek = _controller("vicsek")
olfati_saber = _controller("olfati_saber")
acceleration = _controller("acceleration")

CONTROLLERS: Dict[str, Callable] = {
    "boids": boids,
    "vicsek": vicsek,
    "olfati_saber": olfati_saber,
    "acceleration": acceleration,
}


def get_controller(name):
    """Resolve a controller by name (or pass a callable through)."""
    if callable(name) and not isinstance(name, str):
        return name
    try:
        return CONTROLLERS[name]
    except KeyError as error:
        raise ValueError(
            "Unknown flocking controller %r. Available: %s"
            % (name, ", ".join(sorted(CONTROLLERS)))
        ) from error


def simulate(
    controller="acceleration",
    n_agents: int = 20,
    steps: int = 400,
    dt: float = 0.1,
    dimension: int = 2,
    params: Optional[FlockParams] = None,
    initial: Optional[SwarmState] = None,
    spawn_radius: float = 8.0,
) -> Tuple[List[SwarmState], SwarmMetrics]:
    """Run a flock; returns ``(history, metrics)`` as it always did."""
    params = params or FlockParams()
    name = (
        controller
        if isinstance(controller, str)
        else getattr(controller, "__name__", "acceleration")
    )
    behavior = _build(name, params)
    simulator = SwarmSimulator(
        behavior,
        n_agents=n_agents,
        dimension=dimension,
        params=params,
        dt=dt,
        initial=initial,
    )
    result = simulator.run(steps=steps)
    return result.history, result.metrics
