"""Decentralized swarm control: flocking, formation, coverage and distribution.

The centralized planners in :mod:`pymapf.algorithms` assume a graph, a central
solver and full information. This package is the other half: continuous space,
local sensing, no coordinator. It is built on the same conventions -- an
abstract base class per algorithm family, a name registry, and swappable
strategy objects -- so a controller is chosen with a string and compared with a
loop::

    from pymapf.swarm import SwarmSimulator, available_behaviors

    for name in available_behaviors():
        result = SwarmSimulator(name, n_agents=20).run(steps=300)
        print(name, result.metrics.summary())

Four families:

**Flocking** (:mod:`~pymapf.swarm.flocking`) -- move as a group. Ten control
laws from Reynolds' boids to the 2024 minimalistic model that flocks on relative
range and bearing alone, all composable via
:class:`~pymapf.swarm.base.CompositeBehavior` and all able to use any
:class:`~pymapf.swarm.neighborhood.Neighborhood` strategy.

**Formation** (:mod:`~pymapf.swarm.formation`) -- hold a *shape*, not just a
heading. Organised by what each agent is allowed to measure -- relative
position, range, or bearing -- because that is what decides which symmetry the
controller can fix. Includes the rigidity test that says whether a target shape
is holdable at all.

**Coverage** (:mod:`~pymapf.swarm.coverage`) -- spread out to watch an area.
Lloyd descent and its variants, over a pluggable
:class:`~pymapf.swarm.domain.Domain`, so the same controller deploys a team on a
plane, a sphere, a hemisphere or an annulus.

**Distribution control** (:mod:`~pymapf.swarm.distribution`) -- shape the swarm
into a target *density* rather than assigning individual goals, which is what
scales to hundreds of agents.
"""

from .base import (
    Behavior,
    CompositeBehavior,
    SwarmParams,
    SwarmState,
    available_behaviors,
    get_behavior,
    limit,
    register_behavior,
)
from .neighborhood import (
    ConeNeighborhood,
    GaussianKernelNeighborhood,
    MetricNeighborhood,
    Neighborhood,
    TopologicalNeighborhood,
    get_neighborhood,
)

# Importing the algorithm modules registers them by name.
from .flocking import (  # noqa: F401
    AccelerationFlocking,
    ActiveElastic,
    Boids,
    CuckerSmale,
    DistributedThreeDimensional,
    GaussianKernelFlocking,
    MinimalisticFlocking,
    OlfatiSaber,
    ProximalControl,
    Vicsek,
)
from .coverage import (  # noqa: F401
    AdaptiveCoverage,
    CoverageController,
    CoverageResult,
    CoverageSimulator,
    LimitedRangeCoverage,
    LloydCoverage,
    MixtureCoverage,
    TimeVaryingCoverage,
    available_coverage,
    get_coverage,
    register_coverage,
)
from .density import (
    DensityField,
    GaussianDensity,
    GaussianMixtureDensity,
    SampledDensity,
    TimeVaryingDensity,
    UniformDensity,
    get_density,
)
from .distribution import DensityMatching, MixtureAssignment  # noqa: F401
from .formation import (  # noqa: F401
    BearingFormation,
    CircleFormation,
    CubeFormation,
    CustomFormation,
    DisplacementFormation,
    DistanceFormation,
    FormationShape,
    GridFormation,
    LeaderFollower,
    LineFormation,
    SphereFormation,
    VFormation,
    assign_slots,
    available_shapes,
    formation_error,
    get_shape,
    is_infinitesimally_rigid,
    register_shape,
)
from .domain import (
    AnnulusDomain,
    DiskDomain,
    Domain,
    HemisphereDomain,
    MeshDomain,
    PlanarDomain,
    SphereDomain,
    get_domain,
)
from .simulator import SwarmMetrics, SwarmResult, SwarmSimulator, simulate

__all__ = [
    # core
    "SwarmState",
    "SwarmParams",
    "Behavior",
    "CompositeBehavior",
    "register_behavior",
    "get_behavior",
    "available_behaviors",
    "limit",
    # neighbourhoods
    "Neighborhood",
    "MetricNeighborhood",
    "TopologicalNeighborhood",
    "ConeNeighborhood",
    "GaussianKernelNeighborhood",
    "get_neighborhood",
    # behaviors
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
    "DensityMatching",
    "MixtureAssignment",
    # formation control
    "DisplacementFormation",
    "DistanceFormation",
    "BearingFormation",
    "LeaderFollower",
    "FormationShape",
    "LineFormation",
    "VFormation",
    "CircleFormation",
    "GridFormation",
    "CubeFormation",
    "SphereFormation",
    "CustomFormation",
    "get_shape",
    "register_shape",
    "available_shapes",
    "assign_slots",
    "formation_error",
    "is_infinitesimally_rigid",
    # domains
    "Domain",
    "PlanarDomain",
    "DiskDomain",
    "SphereDomain",
    "HemisphereDomain",
    "AnnulusDomain",
    "MeshDomain",
    "get_domain",
    # densities
    "DensityField",
    "UniformDensity",
    "GaussianDensity",
    "GaussianMixtureDensity",
    "TimeVaryingDensity",
    "SampledDensity",
    "get_density",
    # coverage
    "CoverageController",
    "LloydCoverage",
    "LimitedRangeCoverage",
    "AdaptiveCoverage",
    "MixtureCoverage",
    "TimeVaryingCoverage",
    "CoverageSimulator",
    "CoverageResult",
    "register_coverage",
    "get_coverage",
    "available_coverage",
    # simulation
    "SwarmSimulator",
    "SwarmResult",
    "SwarmMetrics",
    "simulate",
]
