"""Search instrumentation: the hook that makes solvers observable.

A solver's inner loop is where all the interesting behaviour lives -- which
node CBS expanded, which conflict it found, which agent it re-planned -- but
none of it is visible from a returned :class:`~pymapf.core.solver.Solution`.
This module adds a *push* interface for that information:

* :class:`SearchEvent` is a single, immutable observation;
* an **observer** is any callable taking one :class:`SearchEvent`;
* :class:`SearchTrace` is the batteries-included observer that records
  everything so it can be replayed later (used by the animated search views in
  :mod:`pymapf.viz`).

Observing is strictly opt-in: passing no observer costs one ``if`` per event.

Event vocabulary (``kind`` -> payload keys):

``root``            ``{"agents": [name, ...], "cost": int}``
``expand``          ``{"node": int, "cost": int, "open": int}``
``conflict``        ``{"type": str, "a": str, "b": str, "t": int, "cell": Cell}``
``branch``          ``{"agent": str, "constraint": str, "cost": int}``
``agent_planned``   ``{"agent": str, "path": [Cell, ...], "cost": int}``
``solved``          ``{"cost": int, "makespan": int, "expansions": int}``
``failed``          ``{"reason": str}``
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

Observer = Callable[["SearchEvent"], None]


@dataclass(frozen=True)
class SearchEvent:
    """One observation emitted by a solver during :meth:`solve`."""

    kind: str
    step: int
    payload: Dict[str, Any] = field(default_factory=dict)
    elapsed: float = 0.0

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "SearchEvent(%s, step=%d, %r)" % (self.kind, self.step, self.payload)


class SearchTrace:
    """An observer that records events, with a few convenience aggregates.

    Use it directly as the ``observer`` argument of
    :meth:`~pymapf.core.solver.MAPFSolver.solve`::

        trace = SearchTrace()
        solution = pymapf.solve(problem, "cbs", observer=trace)
        print(len(trace), trace.conflicts)
    """

    def __init__(self, max_events: Optional[int] = None):
        self.events: List[SearchEvent] = []
        self.max_events = max_events
        self._start = time.perf_counter()

    def __call__(self, event: SearchEvent) -> None:
        if self.max_events is not None and len(self.events) >= self.max_events:
            return
        self.events.append(event)

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    def of_kind(self, kind: str) -> List[SearchEvent]:
        return [e for e in self.events if e.kind == kind]

    @property
    def conflicts(self) -> List[SearchEvent]:
        return self.of_kind("conflict")

    @property
    def expansions(self) -> int:
        return len(self.of_kind("expand"))

    @property
    def duration(self) -> float:
        """Wall-clock seconds covered by the recorded events."""
        return self.events[-1].elapsed if self.events else 0.0

    def cost_curve(self) -> List[int]:
        """Cost of each expanded node, in expansion order.

        For CBS this is monotonically non-decreasing (best-first on
        sum-of-costs), which makes it a compact picture of how hard the
        instance was.
        """
        return [e.get("cost", 0) for e in self.of_kind("expand")]

    def summary(self) -> Dict[str, Any]:
        solved = self.of_kind("solved")
        return {
            "events": len(self.events),
            "expansions": self.expansions,
            "conflicts": len(self.conflicts),
            "branches": len(self.of_kind("branch")),
            "solved": bool(solved),
            "cost": solved[-1].get("cost") if solved else None,
            "duration": self.duration,
        }


class _Emitter:
    """Internal helper solvers use to emit events without None-checks."""

    __slots__ = ("_observer", "_step", "_start")

    def __init__(self, observer: Optional[Observer]):
        self._observer = observer
        self._step = 0
        self._start = time.perf_counter()

    def __bool__(self) -> bool:
        return self._observer is not None

    def __call__(self, kind: str, **payload: Any) -> None:
        if self._observer is None:
            return
        self._step += 1
        self._observer(
            SearchEvent(
                kind=kind,
                step=self._step,
                payload=payload,
                elapsed=time.perf_counter() - self._start,
            )
        )
