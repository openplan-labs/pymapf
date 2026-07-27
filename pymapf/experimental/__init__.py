"""Experimental variants: ideas being tested, not yet recommendations.

Everything in this package is a deliberate deviation from a published
algorithm, implemented so it can be *measured* against the original rather than
argued about. Each variant states its hypothesis, what it changes, and -- once
:mod:`pymapf.experimental.study` has been run -- what actually happened,
including when the answer was "no better".

Nothing here is imported by :mod:`pymapf` by default. Importing this package
registers the variants under names prefixed with ``x-`` so they can never be
confused with the reference implementations::

    import pymapf.experimental          # registers x-pibt-congestion, ...
    pymapf.solve(problem, "x-pibt-congestion")

Run the study with::

    python -m pymapf.experimental.study --output docs/assets
"""

from .congestion_pibt import CongestionPIBT, congestion_map
from .delay_lns import DelayLNS
from .restart_lacam import RestartLaCAM

__all__ = [
    "CongestionPIBT",
    "congestion_map",
    "DelayLNS",
    "RestartLaCAM",
]
