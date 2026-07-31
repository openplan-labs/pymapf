"""Minimal action and observation spaces.

Gymnasium's spaces are the obvious thing to reach for, and this module
deliberately does not. The rest of PyMAPF's core is pure standard library so it
runs in CI, on a robot and in the browser, and pulling a heavyweight dependency
in just to describe "an integer in ``[0, 5)``" would give that up for nothing.

So these are duck types: same attribute names, same ``sample``/``contains``
semantics, same ``shape``/``dtype``/``n``, which is all any RL library actually
touches. When gymnasium *is* installed, :func:`to_gymnasium` converts them, so
the environment drops into the wider ecosystem without this module ever
importing it.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["Space", "Discrete", "Box", "to_gymnasium"]


class Space:
    """Base class: something you can sample from and test membership of."""

    shape: Tuple[int, ...] = ()
    dtype: np.dtype = np.dtype(np.float32)

    def __init__(self, seed: Optional[int] = None):
        self._random = np.random.default_rng(seed)

    def seed(self, seed: Optional[int] = None) -> None:
        self._random = np.random.default_rng(seed)

    def sample(self):
        raise NotImplementedError

    def contains(self, item) -> bool:
        raise NotImplementedError

    def __contains__(self, item) -> bool:
        return self.contains(item)


class Discrete(Space):
    """The integers ``[0, n)`` -- one action per move."""

    def __init__(self, n: int, seed: Optional[int] = None):
        super().__init__(seed)
        if n <= 0:
            raise ValueError("Discrete needs a positive size, got %d" % n)
        self.n = int(n)
        self.shape = ()
        self.dtype = np.dtype(np.int64)

    def sample(self) -> int:
        return int(self._random.integers(self.n))

    def contains(self, item) -> bool:
        try:
            value = int(item)
        except (TypeError, ValueError):
            return False
        # A bool is an int in Python, but "True" is not a considered action.
        if isinstance(item, (bool, np.bool_)):
            return False
        return 0 <= value < self.n

    def __repr__(self) -> str:
        return "Discrete(%d)" % self.n

    def __eq__(self, other) -> bool:
        return isinstance(other, Discrete) and other.n == self.n


class Box(Space):
    """A bounded, real-valued array -- the observation tensors."""

    def __init__(
        self,
        low: float,
        high: float,
        shape: Sequence[int],
        dtype=np.float32,
        seed: Optional[int] = None,
    ):
        super().__init__(seed)
        self.shape = tuple(int(size) for size in shape)
        self.dtype = np.dtype(dtype)
        self.low = np.full(self.shape, low, dtype=self.dtype)
        self.high = np.full(self.shape, high, dtype=self.dtype)

    def sample(self) -> np.ndarray:
        return self._random.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, item) -> bool:
        array = np.asarray(item)
        return (
            array.shape == self.shape
            and bool(np.all(array >= self.low))
            and bool(np.all(array <= self.high))
        )

    def __repr__(self) -> str:
        return "Box(%s, %s, shape=%s)" % (
            float(self.low.flat[0]),
            float(self.high.flat[0]),
            self.shape,
        )

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Box)
            and other.shape == self.shape
            and np.array_equal(other.low, self.low)
            and np.array_equal(other.high, self.high)
        )


def to_gymnasium(space: Space):
    """Convert to the real gymnasium space, if gymnasium is installed.

    Raises :class:`ImportError` if it is not -- the caller asked for a
    gymnasium object specifically, so silently returning the duck type would be
    the wrong kind of helpful.
    """
    try:
        from gymnasium import spaces as gym_spaces
    except ImportError as error:  # pragma: no cover - exercised only with gymnasium
        raise ImportError(
            "gymnasium is not installed; `pip install pymapf[rl]` or use the "
            "spaces in pymapf.rl.spaces directly, which expose the same API"
        ) from error

    if isinstance(space, Discrete):
        return gym_spaces.Discrete(space.n)
    if isinstance(space, Box):
        return gym_spaces.Box(
            low=float(space.low.flat[0]),
            high=float(space.high.flat[0]),
            shape=space.shape,
            dtype=space.dtype,
        )
    raise TypeError("cannot convert %r to a gymnasium space" % type(space).__name__)
