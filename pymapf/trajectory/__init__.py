"""Joint trajectory optimisation in continuous space.

The rest of this library plans *paths* on graphs and times them afterwards.
This package plans **trajectories**: polynomials of time in R^n, optimised
for the whole fleet at once, with no grid anywhere::

    from pymapf.trajectory import Vehicle, plan_joint_trajectories

    fleet = [Vehicle("a", (0.0, 0.0), (10.0, 10.0), radius=0.3),
             Vehicle("b", (10.0, 0.0), (0.0, 10.0), radius=0.3)]
    plan = plan_joint_trajectories(fleet, v_max=2.0, a_max=4.0,
                                   obstacles=[((5.0, 5.0), 1.2)])

    plan.trajectories["a"](3.2)             # position, metres
    plan.trajectories["a"](3.2, 1)          # velocity
    plan.min_separation()                   # (distance, time, pair)
    plan.is_safe(), plan.max_speed()

Seed it from a discrete plan when the routing decision matters::

    from pymapf.trajectory import waypoints_from_solution

    solution = pymapf.solve(scenario.to_problem(), "cbs")
    plan = plan_joint_trajectories(fleet, waypoints=waypoints_from_solution(solution))

Requires numpy: ``pip install "pymapf[trajectory]"``. See
:mod:`pymapf.trajectory.joint` for the formulation and
:mod:`pymapf.trajectory.qp` for the solver it is built on.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised by the import itself
    import numpy  # noqa: F401
except ModuleNotFoundError as error:  # pragma: no cover
    raise ModuleNotFoundError(
        "pymapf.trajectory needs numpy, which is optional: the discrete "
        "solvers, pymapf.kinodynamic and pymapf.aerial run without it. "
        "Install the extra with `pip install pymapf[trajectory]` (or "
        "`pip install numpy`)."
    ) from error

from .joint import (
    JointPlan,
    Vehicle,
    plan_joint_trajectories,
    waypoints_from_solution,
)
from .polynomial import PiecewisePolynomial, cost_matrix, derivative_row
from .qp import solve_qp

__all__ = [
    "JointPlan",
    "PiecewisePolynomial",
    "Vehicle",
    "cost_matrix",
    "derivative_row",
    "plan_joint_trajectories",
    "solve_qp",
    "waypoints_from_solution",
]
